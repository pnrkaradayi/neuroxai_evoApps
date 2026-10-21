"""Bölgesel (atlas tabanlı) öznitelikler ve bunların füzyona katkısı.

CNN yamaları hipokampus çevresini görür; FDG'de AD'ye özgü hipometabolizma bölgeleri (posterior singulat, prekuneus,
angular girus, inferior temporal) bu alanın dışında kalır. Bu modül önceden işlenmiş MNI hacimlerinden, öğrenme
gerektirmeyen (dolayısıyla sızıntısız) bölgesel ölçüler çıkarır:
  mri_z_<bölge>    : T1 robust z ortalaması (atrofi ile BOS artışı yoğunluğu düşürür)
  mri_bos_<bölge>  : z < -1.5 (BOS benzeri) voksel oranı — afin kayıtta atrofi göstergesi
  fdg_<bölge>      : FDG SUVR ortalaması (serebellum referanslı)
  d_*              : aynı hastanın önceki vizidine göre yıllık değişim (önceki vizit yoksa NaN; XGBoost NaN'ı işler)
  yas, kadin, egitim : demografik değişkenler (PTDEMOG). Tanının dayandığı testler (MMSE, CDR) bilerek alınmaz.

Komutlar:
  .venv\\Scripts\\python -m neuroxai.bolgesel cikar          -> <islenmis>/bolgesel/ADNI_bolgesel.csv
  .venv\\Scripts\\python -m neuroxai.bolgesel deney --ad ana  -> <sonuc>/ana/bolgesel/ozet.json
      (deney.py'nin kat embedding'leri + bölgesel öznitelikler; CNN yeniden eğitilmez)
"""
import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .ortak import AYARLAR, ENVANTER, ISLENMIS, SONUC, sablon, uyanik_tut, vizit_dizini

DIZIN = ISLENMIS / "bolgesel"
BOS_ESIK = -1.5
BLOKLAR = ("mri_roi", "fdg_roi", "delta", "demog")


def _aal():
    S = sablon()
    aal = sitk.GetArrayFromImage(sitk.ReadImage(S["yollar"]["aal"])).astype(np.int64)
    maske = sitk.GetArrayFromImage(sitk.ReadImage(S["yollar"]["maske"])) > 0
    etiket = sorted(int(k) for k in S["aal_etiketler"] if int(k) > 0 and (aal == int(k)).any())
    ad = {int(k): v for k, v in S["aal_etiketler"].items()}
    return aal, maske, etiket, ad


def _vizit_olcu(v: dict) -> dict:
    aal, maske, etiket, ad = _aal_onbellek()
    d = vizit_dizini(v["veri_seti"], v["hasta"], v["vizit_kodu"])
    sonuc = {"hasta": v["hasta"], "vizit_kodu": v["vizit_kodu"], "gun": v["gun"]}
    duz = aal.ravel()
    n = np.bincount(duz, minlength=duz.max() + 1).astype(float)
    t1 = d / "t1_mni.nii.gz"
    if t1.exists():
        z = sitk.GetArrayFromImage(sitk.ReadImage(str(t1))).ravel()
        ort = np.bincount(duz, weights=z, minlength=len(n)) / np.maximum(n, 1)
        bos = np.bincount(duz, weights=(z < BOS_ESIK).astype(float), minlength=len(n)) / np.maximum(n, 1)
        for e in etiket:
            sonuc[f"mri_z_{ad[e]}"] = round(float(ort[e]), 4)
            sonuc[f"mri_bos_{ad[e]}"] = round(float(bos[e]), 4)
    fdg = d / "fdg_mni.nii.gz"
    if fdg.exists() and v.get("fdg_kaynak"):
        s = sitk.GetArrayFromImage(sitk.ReadImage(str(fdg))).ravel()
        ort = np.bincount(duz, weights=s, minlength=len(n)) / np.maximum(n, 1)
        for e in etiket:
            sonuc[f"fdg_{ad[e]}"] = round(float(ort[e]), 4)
    return sonuc


_ONBELLEK = {}


def _aal_onbellek():
    if "aal" not in _ONBELLEK:
        _ONBELLEK["aal"] = _aal()
    return _ONBELLEK["aal"]


def demografi() -> dict:
    yol = sorted((Path(AYARLAR["veri_kok"]) / "metadata" / "adni").glob("PTDEMOG_*.csv"))[-1]
    d = {}
    with open(yol, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            x = d.setdefault(r["PTID"], {})
            if r["PTGENDER"] in ("1", "2"):
                x.setdefault("kadin", int(r["PTGENDER"] == "2"))
            if r["PTDOB"] and "/" in r["PTDOB"]:
                ay, yil = r["PTDOB"].split("/")
                x.setdefault("dogum", date(int(yil), int(ay), 15))
            if r["PTEDUCAT"].isdigit() and 0 < int(r["PTEDUCAT"]) < 30:
                x.setdefault("egitim", int(r["PTEDUCAT"]))
    return d


def komut_cikar():
    with open(ENVANTER, encoding="utf-8", newline="") as f:
        vizitler = [v for v in csv.DictReader(f) if v["veri_seti"] == "ADNI"
                    and (vizit_dizini(v["veri_seti"], v["hasta"], v["vizit_kodu"]) / "qc.json").exists()]
    print(f"{len(vizitler)} vizit, bölgesel ölçüler çıkarılıyor...", flush=True)
    with ProcessPoolExecutor(max_workers=3) as havuz:
        satirlar = list(havuz.map(_vizit_olcu, vizitler, chunksize=16))
    # Yıllık değişim: aynı hastanın aynı ölçüye sahip bir önceki vizidi
    hasta = defaultdict(list)
    for s in satirlar:
        hasta[s["hasta"]].append(s)
    for liste in hasta.values():
        liste.sort(key=lambda s: int(s["gun"]))
        for i, s in enumerate(liste):
            for onek in ("mri_", "fdg_"):
                onceki = next((p for p in reversed(liste[:i]) if any(k.startswith(onek) for k in p)), None)
                if onceki is None or not any(k.startswith(onek) for k in s):
                    continue
                yil = (int(s["gun"]) - int(onceki["gun"])) / 365.25
                if yil < 0.25:
                    continue
                for k in [k for k in s if k.startswith(onek)]:
                    if k in onceki:
                        s[f"d_{k}"] = round((s[k] - onceki[k]) / yil, 5)
    demog = demografi()
    for s in satirlar:
        x = demog.get(s["hasta"], {})
        if "dogum" in x:
            s["yas"] = round((date.fromisoformat(s["vizit_kodu"]) - x["dogum"]).days / 365.25, 2)
        for k in ("kadin", "egitim"):
            if k in x:
                s[k] = x[k]
    alanlar = sorted({k for s in satirlar for k in s})
    DIZIN.mkdir(parents=True, exist_ok=True)
    with open(DIZIN / "ADNI_bolgesel.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=alanlar)
        w.writeheader()
        w.writerows(satirlar)
    print(f"{len(satirlar)} vizit, {len(alanlar) - 3} öznitelik -> {DIZIN / 'ADNI_bolgesel.csv'}")


def blok_adi(sutun: str) -> str:
    if sutun.startswith("d_"):
        return "delta"
    if sutun.startswith("mri_"):
        return "mri_roi"
    if sutun.startswith("fdg_"):
        return "fdg_roi"
    return "demog"


def bolgesel_matris(veri, pet: str = "fdg"):
    """veri.meta sırasına hizalı (N, F) dizi, sütun adları ve blok adları. Yoksa (None, [], [])."""
    yol = DIZIN / "ADNI_bolgesel.csv"
    if not yol.exists():
        return None, [], []
    with open(yol, encoding="utf-8", newline="") as f:
        satir = {(r["hasta"], r["vizit_kodu"]): r for r in csv.DictReader(f)}
    ornek = next(iter(satir.values()))
    sutunlar = [k for k in ornek if k not in ("hasta", "vizit_kodu", "gun")
                and (pet == "fdg" or not k.startswith(("fdg_", "d_fdg_")))]
    X = np.full((len(veri.meta), len(sutunlar)), np.nan, dtype=np.float32)
    for i, m in enumerate(veri.meta):
        r = satir.get((m["hasta"], m["vizit_kodu"]))
        if r:
            X[i] = [float(r[k]) if r[k] not in ("", None) else np.nan for k in sutunlar]
    return X, sutunlar, [blok_adi(s) for s in sutunlar]


def alan_gucu(veri) -> np.ndarray:
    """Vizit başına alan gücü ('3T' / '1.5T'). ADNI'de EMCI yalnız 3T fazlarında kaydedildiği için önemli
    bir çekim karıştırıcısıdır (ek materyal)."""
    with open(ENVANTER, encoding="utf-8", newline="") as f:
        d = {(r["hasta"], r["vizit_kodu"]): ("3T" if str(r["alan_gucu"]).startswith("3") else "1.5T")
             for r in csv.DictReader(f)}
    return np.array([d.get((m["hasta"], m["vizit_kodu"]), "1.5T") for m in veri.meta])


def harmonize(X_tr, X_te, fs_tr, fs_te, ref_tr, sutunlar):
    """Kontrol (CN) referanslı konum-ölçek harmonizasyonu: her sütun ve alan gücü için ortalama/SS yalnız EĞİTİM
    katmanındaki CN vizitlerinden kestirilir; tüm vizitler havuzlanmış CN dağılımına taşınır (sızıntısız)."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    ref = X_tr[ref_tr][:, sutunlar]
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0) + 1e-6
    for g in np.unique(np.concatenate([fs_tr, fs_te])):
        m = ref_tr & (fs_tr == g)
        if m.sum() < 10:
            continue
        mu_g, sd_g = np.nanmean(X_tr[m][:, sutunlar], 0), np.nanstd(X_tr[m][:, sutunlar], 0) + 1e-6
        for X, fs in ((X_tr, fs_tr), (X_te, fs_te)):
            satir = np.where(fs == g)[0]
            X[np.ix_(satir, sutunlar)] = (X[np.ix_(satir, sutunlar)] - mu_g) / sd_g * sd + mu
    return X_tr, X_te


def karistirici_analizi(y, ol, fs, siniflar) -> dict:
    """Alan gücü katmanlarında ikili AUC'ler ve CN içinde p(EMCI)'nin 3T'yi ne kadar 'bildiği'."""
    from sklearn.metrics import roc_auc_score
    from .metrikler import ikili
    sonuc = {k: ikili(y[fs == k], ol[fs == k], siniflar) for k in ("1.5T", "3T")}
    cn = y == siniflar.index("CN")
    if "EMCI" in siniflar and len(np.unique(fs[cn])) == 2:
        sonuc["cn_icinde_pEMCI_3T_auc"] = float(roc_auc_score((fs[cn] == "3T").astype(int), ol[cn, siniflar.index("EMCI")]))
    return sonuc


# ----------------------------------------------------------------- deney: embedding + bölgesel füzyon
def komut_deney(a):
    from .deney import xgb_kalibreli, xgb_optimize
    from .metrikler import bootstrap, ikili, mcnemar, temel
    from .veri import YamaVerisi, bolmeler, capraz_uretim, egitim_oncelleri

    kok = SONUC / a.ad
    ayar = json.loads((kok / "ayarlar.json").read_text(encoding="utf-8"))
    cikti = kok / ("bolgesel_harmonize" if a.harmonize else "bolgesel")
    cikti.mkdir(exist_ok=True)
    veri = YamaVerisi("ADNI", ayar["gorev"], ayar["pet"], ayar["delta_gerekli"], ayar.get("etiket_tutarli", True))
    R, sutunlar, bloklar = bolgesel_matris(veri, ayar["pet"])
    if R is None:
        raise SystemExit("Önce: python -m neuroxai.bolgesel cikar")
    n = len(veri.siniflar)
    fs = alan_gucu(veri)
    ref_sinif = veri.siniflar.index("CN")
    print(f"=== bölgesel ({a.ad}): {veri.ozet()} | {len(sutunlar)} bölgesel öznitelik", flush=True)
    varyantlar = {"fuzyon_bolgesel": "gomme+bolge", "bolgesel": "bolge", "fuzyon_tekrar": "gomme"}
    tahmin = {v: [] for v in varyantlar}
    for k, (egitim, test) in enumerate(bolmeler(veri, ayar["kat"], ayar["tohum"])):
        g = np.load(kok / f"kat{k}" / "gommeler.npz")
        assert np.array_equal(g["y_egitim"], veri.y[egitim])
        parcalar = capraz_uretim(veri, egitim, ayar.get("ic_kat", 3), ayar["tohum"])
        konum = {v: i for i, v in enumerate(egitim.tolist())}
        cv = [(np.array([konum[v] for v in tr]), np.array([konum[v] for v in p])) for tr, p in parcalar]
        for varyant, icerik in varyantlar.items():
            hedef = cikti / f"kat{k}_{varyant}.npz"
            if hedef.exists():
                tahmin[varyant].append((test, np.load(hedef)["olasilik"]))
                continue
            parca_tr, parca_te, harm = [], [], []
            if "gomme" in icerik:
                parca_tr.append(g["egitim"])
                parca_te.append(g["test"])
                harm += [True] * g["egitim"].shape[1]
            if "bolge" in icerik:
                parca_tr.append(R[egitim])
                parca_te.append(R[test])
                harm += [b in ("mri_roi", "fdg_roi") for b in bloklar]  # yıllık değişim ve demografi hariç
            X_tr, X_te = np.concatenate(parca_tr, 1), np.concatenate(parca_te, 1)
            if a.harmonize:
                X_tr, X_te = harmonize(X_tr, X_te, fs[egitim], fs[test], veri.y[egitim] == ref_sinif,
                                       np.where(harm)[0])
            params = xgb_optimize(X_tr, veri.y[egitim], veri.hasta[egitim], n, a.optuna, ayar["tohum"])
            model, kalibre_et = xgb_kalibreli(X_tr, veri.y[egitim], cv, params, n, ayar["tohum"])
            ol = kalibre_et(model.predict_proba(X_te))
            np.savez_compressed(hedef, olasilik=ol, params=json.dumps(params))
            if varyant == "fuzyon_bolgesel":
                model.save_model(cikti / f"kat{k}_xgb_fuzyon_bolgesel.json")
            tahmin[varyant].append((test, ol))
            print(f"kat {k + 1} {varyant:16s}: test macro-F1 {temel(veri.y[test], ol)['macro_f1']:.3f}", flush=True)
    ozet = {"veri": veri.ozet(), "siniflar": veri.siniflar, "bolgesel_sutun": len(sutunlar), "modeller": {},
            "ikili": {}}
    oller = {}
    for varyant, parca in tahmin.items():
        idx = np.concatenate([t for t, _ in parca])
        ol = np.concatenate([o for _, o in parca])
        oller[varyant] = (idx, ol)
        oncel = egitim_oncelleri(veri, idx, ayar["kat"], ayar["tohum"])  # kalibre olasılık -> dengeli karar
        ozet["modeller"][varyant] = {**temel(veri.y[idx], ol, oncel),
                                     "ga95": bootstrap(veri.y[idx], ol, veri.hasta[idx], oncel=oncel)}
        ozet["ikili"][varyant] = ikili(veri.y[idx], ol, veri.siniflar)
        ozet.setdefault("karistirici", {})[varyant] = karistirici_analizi(veri.y[idx], ol, fs[idx], veri.siniflar)
        m = ozet["modeller"][varyant]
        print(f"{varyant:16s} ROC-AUC {m['roc_auc']:.3f}  macro-F1 {m['macro_f1']:.3f}  "
              f"dengeli-doğr {m['dengeli_dogruluk']:.3f}  ECE {m['ece']:.3f}")
    # Ana deneyin füzyon tahminleri (aynı katlar) ile eşleştirilmiş McNemar
    ana = [dict(np.load(p)) for p in sorted(kok.glob("kat*/tahminler.npz"))]
    idx_a = np.concatenate([s["indis"] for s in ana])
    ol_a = np.concatenate([s["fuzyon"] for s in ana])
    ozet["ikili"]["fuzyon_ana"] = ikili(veri.y[idx_a], ol_a, veri.siniflar)
    ozet["karistirici"]["fuzyon_ana"] = karistirici_analizi(veri.y[idx_a], ol_a, fs[idx_a], veri.siniflar)
    sira = np.argsort(idx_a)
    idx_b, ol_b = oller["fuzyon_bolgesel"]
    sira_b = np.argsort(idx_b)
    onc_a = egitim_oncelleri(veri, idx_a[sira], ayar["kat"], ayar["tohum"])
    ozet["mcnemar_bolgesel_vs_ana"] = mcnemar(veri.y[idx_a[sira]], (ol_b[sira_b] / onc_a).argmax(1),
                                              (ol_a[sira] / onc_a).argmax(1))
    (cikti / "ozet.json").write_text(json.dumps(ozet, indent=1, ensure_ascii=False), encoding="utf-8")
    for cift, d in ozet["ikili"]["fuzyon_bolgesel"].items():
        print(f"  {cift:10s} AUC {d['auc']:.3f}  dengeli-doğr {d['dengeli_dogruluk']:.3f}  (n={d['n']})")
    k_ = ozet["karistirici"]["fuzyon_bolgesel"]
    print("  3T içinde:", {c: round(d["auc"], 3) for c, d in k_["3T"].items()}, "| CN içinde p(EMCI)->3T AUC",
          round(k_.get("cn_icinde_pEMCI_3T_auc", float("nan")), 3))
    print("->", cikti / "ozet.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("komut", choices=["cikar", "deney"])
    p.add_argument("--ad", default="ana")
    p.add_argument("--optuna", type=int, default=30)
    p.add_argument("--harmonize", action="store_true", help="CN referanslı alan gücü harmonizasyonu (1.5T/3T)")
    a = p.parse_args()
    uyanik_tut()
    komut_cikar() if a.komut == "cikar" else komut_deney(a)


if __name__ == "__main__":
    main()
