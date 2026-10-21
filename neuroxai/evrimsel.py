"""Evrimsel çok amaçlı füzyon yapılandırması (EvoApps 2027 / EvoMED bildirisi).

deney.py'nin her dış katında kaydettiği sızıntısız (çapraz üretimli) özellikler kullanılır:
<sonuc>/<ad>/kat*/gommeler.npz -> egitim / test = [MRI PCA (128) ‖ PET PCA (128)]. CNN'ler yeniden eğitilmez.

Arama uzayı (genler): XGBoost hiperparametreleri + modalite başına kullanılacak PCA bileşen sayısı (k_mri, k_pet).
Amaçlar (iç 3 katlı hasta bazlı CV, katman dışı ham olasılıklar): macro-F1 (maks.), ECE (min.).
Algoritmalar (aynı bütçe, aynı iç CV, birden çok tohum):
  nsga2  : NSGA-II (evrimsel, çok amaçlı)          motpe : çok amaçlı TPE (Bayes)
  rastgele: rastgele arama (çok amaçlı)            tpe   : tek amaçlı TPE, yalnız macro-F1 (deney.py'deki yöntem)
Seçim: iç ECE <= --ece-esik olan baskın olmayan çözümlerden en yüksek macro-F1 (yoksa en düşük ECE);
       tpe için en yüksek macro-F1 (tek amaç). Seçilen yapılandırma tüm eğitim katmanında eğitilip test edilir.
Cephe kalitesi: (1 - F1, ECE) uzayında hipervolüm, referans (1, --hv-ref-ece).

Kullanım: .venv\\Scripts\\python -m neuroxai.evrimsel --ad ana [--butce 120 --tohumlar 0,1,2]
"""
import argparse
import csv
import json
import time

import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from .deney import xgb_model
from .metrikler import bootstrap, ece, temel
from .ortak import SONUC, uyanik_tut
from .bolgesel import BLOKLAR, bolgesel_matris
from .veri import YamaVerisi, bolmeler

ALGORITMALAR = ("nsga2", "motpe", "rastgele", "tpe")


def ornekleyici(ad: str, tohum: int, nufus: int):
    if ad == "nsga2":
        return optuna.samplers.NSGAIISampler(population_size=nufus, seed=tohum)
    if ad == "rastgele":
        return optuna.samplers.RandomSampler(seed=tohum)
    return optuna.samplers.TPESampler(seed=tohum)  # motpe (çok amaçlı) ya da tpe (tek amaçlı)


def baskin_olmayan(noktalar: np.ndarray) -> np.ndarray:
    """noktalar: (n, 2) en aza indirilecek (1-F1, ECE). Baskın olmayanların maskesi."""
    maske = np.ones(len(noktalar), dtype=bool)
    for i, p in enumerate(noktalar):
        if maske[i]:
            baskin = np.all(noktalar <= p, axis=1) & np.any(noktalar < p, axis=1)
            if baskin.any():
                maske[i] = False
    return maske


def hipervolum(noktalar: np.ndarray, ref: np.ndarray) -> float:
    p = noktalar[baskin_olmayan(noktalar)]
    p = p[np.all(p < ref, axis=1)]
    if not len(p):
        return 0.0
    p = p[np.argsort(p[:, 0])]
    hv, onceki_y = 0.0, ref[1]
    for x, yy in p:
        if yy < onceki_y:
            hv += (ref[0] - x) * (onceki_y - yy)
            onceki_y = yy
    return float(hv)


def sec(kayit: list[dict], esik: float, tek_amac: bool) -> dict:
    if tek_amac:
        return max(kayit, key=lambda r: r["f1"])
    nok = np.array([[1 - r["f1"], r["ece"]] for r in kayit])
    on = [r for r, m in zip(kayit, baskin_olmayan(nok)) if m]
    uygun = [r for r in on if r["ece"] <= esik]
    return max(uygun, key=lambda r: r["f1"]) if uygun else min(on, key=lambda r: r["ece"])


def sutun_sec(yari: int, k_mri: int, k_pet: int, blok_sutun: dict, secili: dict) -> list[int]:
    """[MRI PCA ‖ PET PCA ‖ bölgesel bloklar] içinden genotipin seçtiği sütunlar."""
    sutun = list(range(k_mri)) + list(range(yari, yari + k_pet))
    for b, idx in blok_sutun.items():
        if secili.get(b):
            sutun += idx
    return sutun


def calistir_kat(X, y, grup, n_sinif, algoritma, tohum, a, yari, blok_sutun):
    ic = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=a.tohum).split(X, y, grup))
    agirlik = {i: compute_sample_weight("balanced", y[tr]) for i, (tr, _) in enumerate(ic)}
    kayit = []

    def amac(trial):
        p = {"n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
             "max_depth": trial.suggest_int("max_depth", 2, 6),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
             "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
        k_mri = trial.suggest_int("k_mri", 4, yari, step=4)
        k_pet = trial.suggest_int("k_pet", 4, yari, step=4)
        # Bölgesel öznitelik blokları (atlas ölçüleri, yıllık değişim, demografi): evrimsel blok seçimi
        secili = {b: trial.suggest_categorical(f"blok_{b}", [0, 1]) for b in blok_sutun}
        sutun = sutun_sec(yari, k_mri, k_pet, blok_sutun, secili)
        oof = np.zeros((len(y), n_sinif))
        for i, (tr, va) in enumerate(ic):
            m = xgb_model(p, n_sinif, a.tohum)
            m.fit(X[tr][:, sutun], y[tr], sample_weight=agirlik[i])
            oof[va] = m.predict_proba(X[va][:, sutun])
        olcu = temel(y, oof)
        kayit.append({"deneme": trial.number, "f1": olcu["macro_f1"], "ece": olcu["ece"],
                      "auc": olcu["roc_auc"], "params": p, "k_mri": k_mri, "k_pet": k_pet, "bloklar": secili})
        return olcu["macro_f1"] if algoritma == "tpe" else (olcu["macro_f1"], olcu["ece"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yonler = ["maximize"] if algoritma == "tpe" else ["maximize", "minimize"]
    calisma = optuna.create_study(directions=yonler, sampler=ornekleyici(algoritma, tohum, a.nufus))
    t = time.time()
    calisma.optimize(amac, n_trials=a.butce)
    return kayit, time.time() - t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ad", default="ana", help="deney.py çıktısı (kat*/gommeler.npz)")
    p.add_argument("--butce", type=int, default=120, help="algoritma başına değerlendirme sayısı")
    p.add_argument("--nufus", type=int, default=20, help="NSGA-II nüfus büyüklüğü")
    p.add_argument("--tohumlar", default="0,1,2")
    p.add_argument("--algoritmalar", default=",".join(ALGORITMALAR))
    p.add_argument("--ece-esik", type=float, default=0.05, help="proje formu hedefi ECE <= %%5")
    p.add_argument("--hv-ref-ece", type=float, default=0.30)
    p.add_argument("--tohum", type=int, default=2026, help="iç CV ve XGBoost tohumu (algoritmalar arası sabit)")
    p.add_argument("--sadece-kat", type=int, default=None)
    p.add_argument("--bolgesel", action=argparse.BooleanOptionalAction, default=True,
                   help="bölgesel öznitelik bloklarını (neuroxai.bolgesel cikar) genotipe ekle")
    p.add_argument("--cikti", default="evrimsel", help="<sonuc>/<ad>/ altındaki çıktı klasörü")
    a = p.parse_args()
    uyanik_tut()
    kok = SONUC / a.ad
    ayar = json.loads((kok / "ayarlar.json").read_text(encoding="utf-8"))
    cikti = kok / a.cikti
    cikti.mkdir(exist_ok=True)
    veri = YamaVerisi("ADNI", ayar["gorev"], ayar["pet"], ayar["delta_gerekli"], ayar.get("etiket_tutarli", True))
    n = len(veri.siniflar)
    R, _, bloklar = bolgesel_matris(veri, ayar["pet"]) if a.bolgesel else (None, [], [])
    if a.bolgesel and R is None:
        raise SystemExit("Bölgesel öznitelikler yok: önce python -m neuroxai.bolgesel cikar (ya da --no-bolgesel)")
    bloklar = np.array(bloklar)
    tohumlar = [int(t) for t in a.tohumlar.split(",")]
    algoritmalar = a.algoritmalar.split(",")
    ref = np.array([1.0, a.hv_ref_ece])
    print(f"=== evrimsel ({a.ad}): {veri.ozet()} | bütçe {a.butce}, tohumlar {tohumlar}", flush=True)

    satirlar, tahmin = [], {}
    for k, (egitim, test) in enumerate(bolmeler(veri, ayar["kat"], ayar["tohum"])):
        if a.sadece_kat is not None and k != a.sadece_kat:
            continue
        g = np.load(kok / f"kat{k}" / "gommeler.npz")
        X_tr, X_te = g["egitim"], g["test"]
        assert np.array_equal(g["y_egitim"], veri.y[egitim]) and np.array_equal(g["y_test"], veri.y[test])
        yari = X_tr.shape[1] // 2
        blok_sutun = {}
        if R is not None:
            genislik = X_tr.shape[1]
            X_tr, X_te = np.concatenate([X_tr, R[egitim]], 1), np.concatenate([X_te, R[test]], 1)
            blok_sutun = {b: (genislik + np.where(bloklar == b)[0]).tolist() for b in BLOKLAR if (bloklar == b).any()}
        for alg in algoritmalar:
            for s in tohumlar:
                hedef = cikti / f"kat{k}_{alg}_t{s}.json"
                if hedef.exists():  # yarıda kalan çalışma kaldığı yerden sürer
                    r = json.loads(hedef.read_text(encoding="utf-8"))
                else:
                    kayit, sure = calistir_kat(X_tr, veri.y[egitim], veri.hasta[egitim], n, alg, s, a,
                                               yari, blok_sutun)
                    secilen = sec(kayit, a.ece_esik, alg == "tpe")
                    sutun = sutun_sec(yari, secilen["k_mri"], secilen["k_pet"], blok_sutun, secilen["bloklar"])
                    m = xgb_model(secilen["params"], n, a.tohum)
                    m.fit(X_tr[:, sutun], veri.y[egitim], sample_weight=compute_sample_weight("balanced", veri.y[egitim]))
                    ol = m.predict_proba(X_te[:, sutun])
                    nok = np.array([[1 - r_["f1"], r_["ece"]] for r_ in kayit])
                    r = {"kat": k, "algoritma": alg, "tohum": s, "sure_sn": round(sure, 1),
                         "hv": hipervolum(nok, ref), "on_boyutu": int(baskin_olmayan(nok).sum()),
                         "secilen": secilen, "test_olasilik": ol.tolist(), "test_indis": test.tolist(),
                         "kayit": [{k_: v for k_, v in r_.items() if k_ != "params"} for r_ in kayit]}
                    hedef.write_text(json.dumps(r), encoding="utf-8")
                ol = np.array(r["test_olasilik"])
                olcu = temel(veri.y[test], ol)
                print(f"kat {k + 1} {alg:9s} t{s}: HV {r['hv']:.4f}  seçilen iç F1 {r['secilen']['f1']:.3f} "
                      f"ECE {r['secilen']['ece']:.3f} (k {r['secilen']['k_mri']}/{r['secilen']['k_pet']}) | "
                      f"test F1 {olcu['macro_f1']:.3f} ECE {olcu['ece']:.3f}  ({r['sure_sn'] / 60:.1f} dk)", flush=True)
                satirlar.append({"kat": k, "algoritma": alg, "tohum": s, "hv": r["hv"], "on_boyutu": r["on_boyutu"],
                                 "ic_f1": r["secilen"]["f1"], "ic_ece": r["secilen"]["ece"],
                                 "k_mri": r["secilen"]["k_mri"], "k_pet": r["secilen"]["k_pet"],
                                 "test_f1": olcu["macro_f1"], "test_ece": olcu["ece"], "test_auc": olcu["roc_auc"],
                                 "test_ba": olcu["dengeli_dogruluk"], "sure_sn": r["sure_sn"],
                                 **{f"blok_{b}": int(r["secilen"].get("bloklar", {}).get(b, 0)) for b in BLOKLAR}})
                tahmin.setdefault((alg, s), []).append((test, ol))

    with open(cikti / "ozet_satirlar.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    # Algoritma özeti: katlar birleştirilmiş test tahminleri (tohum başına), tohumlar arası ortalama ± SS
    ozet = {"ayarlar": vars(a), "algoritmalar": {}}
    for alg in algoritmalar:
        tohum_olcu = []
        for s in tohumlar:
            parca = tahmin.get((alg, s), [])
            if not parca:
                continue
            idx = np.concatenate([t for t, _ in parca])
            ol = np.concatenate([o for _, o in parca])
            tohum_olcu.append({**temel(veri.y[idx], ol), "ga95": bootstrap(veri.y[idx], ol, veri.hasta[idx], n=300)})
        sat = [r for r in satirlar if r["algoritma"] == alg]
        ozet["algoritmalar"][alg] = {
            "hv_ort": float(np.mean([r["hv"] for r in sat])), "hv_ss": float(np.std([r["hv"] for r in sat])),
            "on_boyutu_ort": float(np.mean([r["on_boyutu"] for r in sat])),
            "k_mri_ort": float(np.mean([r["k_mri"] for r in sat])), "k_pet_ort": float(np.mean([r["k_pet"] for r in sat])),
            "sure_dk_ort": float(np.mean([r["sure_sn"] for r in sat]) / 60),
            "blok_orani": {b: float(np.mean([r[f"blok_{b}"] for r in sat])) for b in BLOKLAR},
            **{f"test_{m}_ort": float(np.mean([o[k_] for o in tohum_olcu]))
               for m, k_ in (("auc", "roc_auc"), ("f1", "macro_f1"), ("ba", "dengeli_dogruluk"), ("ece", "ece"))},
            **{f"test_{m}_ss": float(np.std([o[k_] for o in tohum_olcu]))
               for m, k_ in (("auc", "roc_auc"), ("f1", "macro_f1"), ("ba", "dengeli_dogruluk"), ("ece", "ece"))},
            "tohumlar": tohum_olcu}
    # Eşleştirilmiş (kat × tohum) Wilcoxon: NSGA-II'ye karşı diğerleri, hipervolüm ve test ölçüleri
    from scipy.stats import wilcoxon
    karsi = {}
    for alg in algoritmalar:
        if alg == "nsga2" or "nsga2" not in algoritmalar:
            continue
        for olcut in ("hv", "test_f1", "test_ece"):
            a_ = {(r["kat"], r["tohum"]): r[olcut] for r in satirlar if r["algoritma"] == "nsga2"}
            b_ = {(r["kat"], r["tohum"]): r[olcut] for r in satirlar if r["algoritma"] == alg}
            ortak = sorted(set(a_) & set(b_))
            fark = [a_[i] - b_[i] for i in ortak]
            if len(ortak) >= 5 and any(fark):
                karsi[f"nsga2_vs_{alg}_{olcut}"] = {"n": len(ortak), "ort_fark": float(np.mean(fark)),
                                                    "p": float(wilcoxon(fark).pvalue)}
    ozet["wilcoxon"] = karsi
    (cikti / "ozet_evrimsel.json").write_text(json.dumps(ozet, indent=1, ensure_ascii=False), encoding="utf-8")
    for alg, o in ozet["algoritmalar"].items():
        print(f"{alg:9s} HV {o['hv_ort']:.4f}±{o['hv_ss']:.4f}  test F1 {o['test_f1_ort']:.3f}±{o['test_f1_ss']:.3f}  "
              f"ECE {o['test_ece_ort']:.3f}  AUC {o['test_auc_ort']:.3f}  k {o['k_mri_ort']:.0f}/{o['k_pet_ort']:.0f}")
    for k_, v in karsi.items():
        print(f"  {k_}: fark {v['ort_fark']:+.4f}, p={v['p']:.4f} (n={v['n']})")
    print("->", cikti)


if __name__ == "__main__":
    main()
