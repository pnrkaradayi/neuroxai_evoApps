"""İkili görevler (literatürle doğrudan karşılaştırma için): CN–AD, CN–MCI, MCI–AD.

uc_sinif deneyinin aynı dış katları kullanılır; her katta yalnız iki sınıfın vizitleri alınır. Özellikler:
çapraz üretimli CNN embedding'leri (PCA) + bölgesel ölçüler. XGBoost, Optuna TPE (30 deneme, iç hasta bazlı CV,
ölçüt dengeli doğruluk) + çapraz üretilmiş izotonik kalibrasyon. Her şey hasta bazlı; CNN yeniden eğitilmez.
Ölçüler: ROC-AUC, doğruluk, dengeli doğruluk, duyarlılık, özgüllük (hasta kümeli bootstrap %95 GA) ve alan gücü
(1.5T / 3T) katmanlarında AUC.

Kullanım: .venv\\Scripts\\python -m neuroxai.ikili_deney --ad uc_sinif
"""
import argparse
import json

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .bolgesel import alan_gucu, bolgesel_matris
from .deney import xgb_kalibreli, xgb_optimize
from .ortak import SONUC, uyanik_tut
from .veri import YamaVerisi, bolmeler, capraz_uretim

CIFTLER = (("CN", "AD"), ("CN", "MCI"), ("MCI", "AD"))


def olcu(y, p, esik=0.5):
    """esik: karar eşiği (dizi olabilir). Kalibre olasılıkta eğitim katmanındaki pozitif sınıf oranı, dengeli
    hata için Bayes kararına karşılık gelir; yalnız eğitim verisinden belirlenir."""
    t = (p >= esik).astype(int)
    duy = float(((t == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    ozg = float(((t == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return {"auc": float(roc_auc_score(y, p)), "dogruluk": float((t == y).mean()),
            "dengeli_dogruluk": float(balanced_accuracy_score(y, t)), "duyarlilik": duy, "ozgulluk": ozg}


def bootstrap(y, p, grup, esik, n=1000, tohum=2026):
    rng = np.random.default_rng(tohum)
    hastalar = np.unique(grup)
    indis = {h: np.where(grup == h)[0] for h in hastalar}
    orn = {k: [] for k in ("auc", "dogruluk", "dengeli_dogruluk", "duyarlilik", "ozgulluk")}
    for _ in range(n):
        s = np.concatenate([indis[h] for h in rng.choice(hastalar, len(hastalar), replace=True)])
        if len(np.unique(y[s])) < 2:
            continue
        m = olcu(y[s], p[s], esik[s])
        for k in orn:
            orn[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in orn.items()}


def esikler(veri, ayar, ia, ib, idx):
    """Her test vizidi için kendi katının eğitim kısmındaki pozitif sınıf oranı (yalnız eğitim verisi)."""
    e = {}
    for egitim, test in bolmeler(veri, ayar["kat"], ayar["tohum"]):
        y_tr = veri.y[egitim][np.isin(veri.y[egitim], [ia, ib])]
        for v in test:
            e[int(v)] = float((y_tr == ib).mean())
    return np.array([e[int(v)] for v in idx])


def ozetle(kok, veri, ayar):
    fs = alan_gucu(veri)
    ozet = {"veri": veri.ozet(), "esik": "eğitim katmanındaki pozitif sınıf oranı (kalibre olasılık)", "ciftler": {}}
    for a_ad, b_ad in CIFTLER:
        dosya = kok / "ikili" / f"{a_ad}-{b_ad}.npz"
        if not dosya.exists():
            continue
        d = np.load(dosya)
        idx, p, y = d["indis"], d["olasilik"], d["y"]
        esik = esikler(veri, ayar, veri.siniflar.index(a_ad), veri.siniflar.index(b_ad), idx)
        sonuc = {"n": int(len(idx)), "n_hasta": int(len(set(veri.hasta[idx]))), **olcu(y, p, esik),
                 "ga95": bootstrap(y, p, veri.hasta[idx], esik),
                 "alan_gucu": {g_: olcu(y[fs[idx] == g_], p[fs[idx] == g_], esik[fs[idx] == g_])
                               for g_ in ("1.5T", "3T") if len(np.unique(y[fs[idx] == g_])) == 2}}
        ozet["ciftler"][f"{a_ad}-{b_ad}"] = sonuc
        print(f"== {a_ad}-{b_ad}: AUC {sonuc['auc']:.3f} {[round(x, 3) for x in sonuc['ga95']['auc']]}  doğruluk "
              f"{sonuc['dogruluk']:.3f}  dengeli {sonuc['dengeli_dogruluk']:.3f}  duy {sonuc['duyarlilik']:.3f}  "
              f"özg {sonuc['ozgulluk']:.3f} | 1.5T/3T AUC "
              + " / ".join(f"{v['auc']:.3f}" for v in sonuc["alan_gucu"].values()), flush=True)
    (kok / "ikili" / "ozet.json").write_text(json.dumps(ozet, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", kok / "ikili" / "ozet.json")


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--ad", default="uc_sinif")
    pa.add_argument("--optuna", type=int, default=30)
    pa.add_argument("--sadece-ozet", action="store_true", help="kaydedilmiş tahminlerden özeti yeniden hesapla")
    a = pa.parse_args()
    uyanik_tut()
    kok = SONUC / a.ad
    ayar = json.loads((kok / "ayarlar.json").read_text(encoding="utf-8"))
    cikti = kok / "ikili"
    cikti.mkdir(exist_ok=True)
    veri = YamaVerisi("ADNI", ayar["gorev"], ayar["pet"], ayar["delta_gerekli"], ayar.get("etiket_tutarli", True))
    if a.sadece_ozet:
        return ozetle(kok, veri, ayar)
    R, _, _ = bolgesel_matris(veri, ayar["pet"])
    for a_ad, b_ad in CIFTLER:
        if (cikti / f"{a_ad}-{b_ad}.npz").exists():
            continue
        ia, ib = veri.siniflar.index(a_ad), veri.siniflar.index(b_ad)
        idx_all, p_all = [], []
        for k, (egitim, test) in enumerate(bolmeler(veri, ayar["kat"], ayar["tohum"])):
            g = np.load(kok / f"kat{k}" / "gommeler.npz")
            Xg_tr = np.concatenate([g["egitim"], R[egitim]], 1)
            Xg_te = np.concatenate([g["test"], R[test]], 1)
            m_tr = np.isin(veri.y[egitim], [ia, ib])
            m_te = np.isin(veri.y[test], [ia, ib])
            tr, te = egitim[m_tr], test[m_te]
            y_tr = (veri.y[tr] == ib).astype(int)
            parcalar = capraz_uretim(veri, tr, 3, ayar["tohum"])
            konum = {v: i for i, v in enumerate(tr.tolist())}
            cv = [(np.array([konum[v] for v in t_]), np.array([konum[v] for v in p_])) for t_, p_ in parcalar]
            params = xgb_optimize(Xg_tr[m_tr], y_tr, veri.hasta[tr], 2, a.optuna, ayar["tohum"])
            model, kalibre_et = xgb_kalibreli(Xg_tr[m_tr], y_tr, cv, params, 2, ayar["tohum"])
            p = kalibre_et(model.predict_proba(Xg_te[m_te]))[:, 1]
            idx_all.append(te)
            p_all.append(p)
            print(f"{a_ad}-{b_ad} kat {k + 1}: AUC {roc_auc_score((veri.y[te] == ib).astype(int), p):.3f}", flush=True)
        idx, p = np.concatenate(idx_all), np.concatenate(p_all)
        np.savez_compressed(cikti / f"{a_ad}-{b_ad}.npz", indis=idx, olasilik=p, y=(veri.y[idx] == ib).astype(int))
    ozetle(kok, veri, ayar)


if __name__ == "__main__":
    main()
