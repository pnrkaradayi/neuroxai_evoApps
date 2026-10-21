"""NeuroXAI-Fusion deneyi: hasta bazlı 5 katlı çapraz doğrulama (ADNI).

Her dış katta (proje formu 4.6-4.7):
  1. Çapraz üretim: eğitim katmanı hasta bazlı 3 parçaya bölünür; her parça için MRI ve PET EfficientNet-B0'ları
     o parçayı görmeden ince ayarlanır (%15 hasta bazlı erken durdurma). Parçanın vizit embedding'leri (3 koronal
     dilimin GAP 1280 ortalaması) bu modellerden alınır -> XGBoost eğitim özellikleri sızıntısız (katman dışı) olur.
     Test vizitlerinin embedding'i / logit'i üç modelin ortalamasıdır.
  2. Embedding'ler eğitim katmanında PCA ile modalite başına --pca bileşene indirgenir; füzyon = [MRI ‖ PET].
  3. XGBoost; hiperparametreler Optuna (TPE, Bayes) ile iç 3 katlı hasta bazlı CV'de; sınıf ağırlıklı.
  4. İzotonik kalibrasyon: aynı parçalarda çapraz üretilmiş (katman dışı) XGBoost olasılıklarına sınıf başına.
  Karşılaştırma (aynı CNN'lerle): yalnız-MRI / yalnız-PET XGBoost, CNN-MRI, CNN-PET, CNN olasılık ortalaması.
Kanal ablasyonu (Δ / kenar katkısı) için --kanallar ile ayrı deney çalıştırılır (CNN'ler yeniden eğitilir).

Etiketler: 4 sınıf kayıt grubundan; vizit tanısıyla (dx3) çelişen takip vizitleri alınmaz (veri.YamaVerisi).

Kullanım:
  .venv\\Scripts\\python -m neuroxai.deney --ad ana --pet fdg              (4 sınıf, T1 + FDG-PET)
  .venv\\Scripts\\python -m neuroxai.deney --ad delta_yok --pet fdg --kanallar orj,kenar
"""
import argparse
import json
import time

import numpy as np
import optuna
import torch
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from . import cnn
from .metrikler import bootstrap, mcnemar, temel
from .ortak import SONUC, uyanik_tut
from .veri import YamaVerisi, bolmeler, capraz_uretim, egitim_oncelleri, ic_dogrulama

KANAL = {"orj": 0, "delta": 1, "kenar": 2}


def kanal_sec(X: np.ndarray, kanallar: list[str]) -> np.ndarray:
    """Kullanılmayan kanalları sıfırlar (girdi boyutu sabit kalır; ablasyon için)."""
    X = X.copy()
    for ad, i in KANAL.items():
        if ad not in kanallar:
            X[:, :, i] = 0
    return X


def softmax(lg: np.ndarray) -> np.ndarray:
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def xgb_model(params: dict, n_sinif: int, tohum: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**params, objective="multi:softprob", num_class=n_sinif, tree_method="hist",
                             n_jobs=-1, random_state=tohum, eval_metric="mlogloss")


def xgb_optimize(X: np.ndarray, y: np.ndarray, grup: np.ndarray, n_sinif: int, deneme: int, tohum: int) -> dict:
    ic = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=tohum).split(X, y, grup))

    def amac(trial):
        p = {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
             "max_depth": trial.suggest_int("max_depth", 2, 6),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
             "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
        puan = []
        for tr, va in ic:
            m = xgb_model(p, n_sinif, tohum)
            m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            puan.append(temel(y[va], m.predict_proba(X[va]))["macro_f1"])
        return float(np.mean(puan))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    calisma = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tohum))
    calisma.optimize(amac, n_trials=deneme)
    return calisma.best_params


def xgb_kalibreli(X: np.ndarray, y: np.ndarray, cv: list, params: dict, n_sinif: int, tohum: int):
    """Tüm eğitim katmanında XGBoost + çapraz üretilmiş (katman dışı) olasılıklarla sınıf başına izotonik kalibrasyon.
    Döner: (model, kalibre_et(olasılık) fonksiyonu)."""
    oof = np.zeros((len(y), n_sinif))
    for tr, va in cv:
        m = xgb_model(params, n_sinif, tohum)
        m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        oof[va] = m.predict_proba(X[va])
    izo = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof[:, c], (y == c).astype(float))
           for c in range(n_sinif)]
    model = xgb_model(params, n_sinif, tohum)
    model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))

    def kalibre_et(olasilik):
        q = np.stack([izo[c].predict(olasilik[:, c]) for c in range(n_sinif)], 1) + 1e-6
        return q / q.sum(1, keepdims=True)

    return model, kalibre_et


def xgb_egit(X_tr, y_tr, X_ka, y_ka, params, n_sinif, tohum):
    """(dis_dogrulama.py için) Eğitim kümesinde XGBoost, ayrı kümede izotonik kalibrasyon."""
    m = xgb_model(params, n_sinif, tohum)
    m.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    kal = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic")
    kal.fit(X_ka, y_ka)
    return m, kal


def kat_calistir(veri, X, egitim, test, a, gunluk, kat_dizin):
    n = len(veri.siniflar)
    parcalar = capraz_uretim(veri, egitim, a.ic_kat, a.tohum)
    gunluk(f"  eğitim {len(egitim)} / test {len(test)} vizit; CNN'ler {a.ic_kat} parçalı çapraz üretimle")
    gomme, test_logit = {}, {}
    for m, Xm in X.items():
        t = time.time()
        g_egitim = np.zeros((len(veri), 1280), dtype=np.float32)
        g_test, l_test = [], []
        for j, (tr, parca) in enumerate(parcalar):
            tr2, va = ic_dogrulama(veri, tr, tohum=a.tohum)
            gunluk(f"  CNN-{m} parça {j + 1}/{len(parcalar)} (eğitim {len(tr2)} / erken durdurma {len(va)})")
            model = cnn.egit(Xm, veri.y, tr2, va, n, epok=a.epok, tohum=a.tohum + j, gunluk=gunluk)
            g_egitim[parca] = cnn.vizit_ciktilari(model, Xm, parca)[0]
            g, lg = cnn.vizit_ciktilari(model, Xm, test)
            g_test.append(g)
            l_test.append(lg)
            torch.save(model.state_dict(), kat_dizin / f"cnn_{m}_{j}.pt")
        g_egitim[test] = np.mean(g_test, 0)
        gomme[m], test_logit[m] = g_egitim, np.mean(l_test, 0)
        gunluk(f"  CNN-{m} tamam ({(time.time() - t) / 60:.1f} dk)")

    # PCA (yalnız eğitim katmanında öğrenilir), modalite başına
    ozel = {}
    for m in X:
        if a.pca:
            pca = PCA(n_components=a.pca, random_state=a.tohum).fit(gomme[m][egitim])
            ozel[m] = pca.transform(gomme[m]).astype(np.float32)
            np.savez_compressed(kat_dizin / f"pca_{m}.npz", bilesen=pca.components_, ortalama=pca.mean_,
                                aciklanan=pca.explained_variance_ratio_)
        else:
            ozel[m] = gomme[m]

    def ozellik(indisler, hangi):
        return np.concatenate([ozel[m][indisler] for m in hangi], 1)

    konum = {v: i for i, v in enumerate(egitim.tolist())}
    cv = [(np.array([konum[v] for v in tr]), np.array([konum[v] for v in parca])) for tr, parca in parcalar]
    sonuc = {"indis": test, "y": veri.y[test]}
    for varyant, hangi in (("fuzyon", ["mri", "pet"]), ("mri", ["mri"]), ("pet", ["pet"])):
        t = time.time()
        X_tr = ozellik(egitim, hangi)
        params = xgb_optimize(X_tr, veri.y[egitim], veri.hasta[egitim], n, a.optuna, a.tohum)
        model, kalibre_et = xgb_kalibreli(X_tr, veri.y[egitim], cv, params, n, a.tohum)
        ham = model.predict_proba(ozellik(test, hangi))
        sonuc[varyant], sonuc[varyant + "_kalibresiz"] = kalibre_et(ham), ham
        if varyant == "fuzyon":
            model.save_model(kat_dizin / "xgb_fuzyon.json")
            (kat_dizin / "xgb_params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
            np.savez_compressed(kat_dizin / "gommeler.npz", egitim=X_tr, y_egitim=veri.y[egitim],
                                test=ozellik(test, hangi), y_test=veri.y[test])
        gunluk(f"  XGB-{varyant}: test macro-F1 {temel(veri.y[test], sonuc[varyant])['macro_f1']:.3f} "
               f"({(time.time() - t) / 60:.1f} dk)")
    sonuc["cnn_mri"], sonuc["cnn_pet"] = softmax(test_logit["mri"]), softmax(test_logit["pet"])
    sonuc["cnn_ortalama"] = (sonuc["cnn_mri"] + sonuc["cnn_pet"]) / 2
    gunluk(f"  CNN ortalama: test macro-F1 {temel(veri.y[test], sonuc['cnn_ortalama'])['macro_f1']:.3f}")
    return sonuc


VARYANTLAR = ("fuzyon", "mri", "pet", "cnn_ortalama", "cnn_mri", "cnn_pet", "fuzyon_kalibresiz")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ad", required=True)
    p.add_argument("--gorev", choices=["dort", "uc"], default="dort")
    p.add_argument("--pet", choices=["amiloid", "fdg"], default="fdg")
    p.add_argument("--kanallar", default="orj,delta,kenar")
    p.add_argument("--delta-gerekli", action=argparse.BooleanOptionalAction, default=False,
                   help="yalnız önceki vizidi olan (Δ'lı) vizitler; varsayılan: tümü (Δ yoksa kanal 0)")
    p.add_argument("--etiket-tutarli", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kat", type=int, default=5)
    p.add_argument("--ic-kat", type=int, default=3, help="CNN çapraz üretim parça sayısı")
    p.add_argument("--pca", type=int, default=128, help="modalite başına PCA bileşeni (0: kapalı)")
    p.add_argument("--epok", type=int, default=12)
    p.add_argument("--optuna", type=int, default=30)
    p.add_argument("--tohum", type=int, default=2026)
    p.add_argument("--sadece-kat", type=int, default=None, help="yalnız bu katı çalıştır (deneme)")
    p.add_argument("--sizinti", action="store_true",
                   help="YALNIZ GÖSTERİM: literatürdeki yaygın hatayı bilerek uygular — vizit bazlı bölme (aynı "
                        "hastanın vizitleri eğitim ve teste dağılır) + çapraz üretimsiz yığma. Asıl sonuç değildir.")
    a = p.parse_args()
    if a.sizinti:
        a.ic_kat = 1
    uyanik_tut()  # ekran kapansa da uzun eğitim sürsün
    dizin = SONUC / a.ad
    dizin.mkdir(parents=True, exist_ok=True)
    gunluk_dosya = open(dizin / "gunluk.txt", "a", encoding="utf-8")

    def gunluk(s):
        print(s, flush=True)
        gunluk_dosya.write(s + "\n")
        gunluk_dosya.flush()

    (dizin / "ayarlar.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    veri = YamaVerisi("ADNI", a.gorev, a.pet, a.delta_gerekli, a.etiket_tutarli)
    gunluk(f"=== {a.ad}: {veri.ozet()}")
    if a.sizinti:
        # Her vizit kendi "grubu": tüm bölmeler (dış, iç, erken durdurma, XGBoost araması) vizit bazlı olur
        veri.hasta = np.array([f"vizit{i}" for i in range(len(veri))])
        gunluk("!!! SIZINTI GÖSTERİMİ: vizit bazlı bölme + çapraz üretimsiz yığma (yalnız karşılaştırma için)")
    kanallar = a.kanallar.split(",")
    X = {"mri": kanal_sec(cnn.girdi_olcekle(veri.mri), kanallar),
         "pet": kanal_sec(cnn.girdi_olcekle(veri.pet), kanallar)}
    sonuclar = []
    for k, (egitim, test) in enumerate(bolmeler(veri, a.kat, a.tohum)):
        if a.sadece_kat is not None and k != a.sadece_kat:
            continue
        kat_dizin = dizin / f"kat{k}"
        kat_dizin.mkdir(exist_ok=True)
        if (kat_dizin / "tahminler.npz").exists():  # yarıda kalan deney kaldığı kattan sürer
            gunluk(f"--- kat {k + 1}/{a.kat}: önceden bitmiş, yükleniyor")
            sonuclar.append(dict(np.load(kat_dizin / "tahminler.npz")))
            continue
        gunluk(f"--- kat {k + 1}/{a.kat}")
        s = kat_calistir(veri, X, egitim, test, a, gunluk, kat_dizin)
        np.savez_compressed(kat_dizin / "tahminler.npz", **s)
        sonuclar.append(s)

    # Katların tahminlerini birleştir ve raporla
    indis = np.concatenate([s["indis"] for s in sonuclar])
    y, hasta, delta = veri.y[indis], veri.hasta[indis], veri.delta_var[indis]
    ozet = {"veri": veri.ozet(), "siniflar": veri.siniflar, "ayarlar": vars(a), "modeller": {},
            "delta_var_altkume": {}}
    # İzotonik kalibre varyantlarda sınıf kararı eğitim katmanı önceline göre (argmax p/oncel; dengeli Bayes kararı)
    oncel_tum = egitim_oncelleri(veri, indis, a.kat, a.tohum)
    kalibreli = {"fuzyon", "mri", "pet"}
    for varyant in VARYANTLAR:
        ol = np.concatenate([s[varyant] for s in sonuclar])
        oncel = oncel_tum if varyant in kalibreli else None
        ozet["modeller"][varyant] = {**temel(y, ol, oncel), "ga95": bootstrap(y, ol, hasta, oncel=oncel)}
        if delta.sum() >= 30 and len(np.unique(y[delta])) == len(veri.siniflar):
            ozet["delta_var_altkume"][varyant] = {"n": int(delta.sum()), **temel(y[delta], ol[delta],
                                                                                None if oncel is None else oncel[delta])}
        m = ozet["modeller"][varyant]
        gunluk(f"{varyant:18s} ROC-AUC {m['roc_auc']:.3f}  macro-F1 {m['macro_f1']:.3f}  "
               f"dengeli-doğr {m['dengeli_dogruluk']:.3f}  ECE {m['ece']:.3f}")
    def karar(v):
        ol_ = np.concatenate([s[v] for s in sonuclar])
        return (ol_ / oncel_tum).argmax(1) if v in kalibreli else ol_.argmax(1)

    fz = karar("fuzyon")
    for diger in ("mri", "pet", "cnn_ortalama"):
        ozet[f"mcnemar_fuzyon_vs_{diger}"] = mcnemar(y, fz, karar(diger))
    (dizin / "ozet.json").write_text(json.dumps(ozet, indent=1, ensure_ascii=False), encoding="utf-8")
    gunluk(f"-> {dizin / 'ozet.json'}")


if __name__ == "__main__":
    main()
