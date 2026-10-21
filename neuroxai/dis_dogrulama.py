"""Dış doğrulama (proje formu 4.2, 4.7): tüm ADNI ile son modeli eğit, OASIS-3'te (hiç görülmemiş) test et.

ADNI hastalarının %15'i (hasta bazlı) CNN erken durdurma ve XGBoost kalibrasyonu için ayrılır.
Son modeller <sonuc>/<ad>/son_model altına kaydedilir (xai.py bunları kullanır).

Kullanım: .venv\\Scripts\\python -m neuroxai.dis_dogrulama --ad ana
"""
import argparse
import json

import numpy as np
import torch

from . import cnn
from .deney import KANAL, kanal_sec, xgb_egit, xgb_optimize
from .metrikler import bootstrap, temel
from .ortak import SONUC, uyanik_tut
from .veri import YamaVerisi, ic_dogrulama


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ad", required=True)
    p.add_argument("--gorev", choices=["dort", "uc"], default="dort")
    p.add_argument("--kanallar", default="orj,delta,kenar")
    p.add_argument("--epok", type=int, default=12)
    p.add_argument("--optuna", type=int, default=30)
    p.add_argument("--tohum", type=int, default=2026)
    a = p.parse_args()
    uyanik_tut()  # ekran kapansa da uzun eğitim sürsün
    dizin = SONUC / a.ad / "son_model"
    dizin.mkdir(parents=True, exist_ok=True)
    kanallar = a.kanallar.split(",")
    assert set(kanallar) <= set(KANAL)

    adni = YamaVerisi("ADNI", a.gorev, "amiloid", delta_gerekli=True)
    # OASIS'te tek vizitli hastaların Δ-MRI'ı ek (önceki) T1'den gelir; Δ yoksa vizit yine test edilir (Δ=0)
    oasis = YamaVerisi("OASIS3", a.gorev, "amiloid", delta_gerekli=False)
    print(f"ADNI  : {adni.ozet()}\nOASIS : {oasis.ozet()}", flush=True)
    n = len(adni.siniflar)
    X = {m: kanal_sec(cnn.girdi_olcekle(getattr(adni, "mri" if m == "mri" else "pet")), kanallar)
         for m in ("mri", "pet")}
    Xo = {m: kanal_sec(cnn.girdi_olcekle(getattr(oasis, "mri" if m == "mri" else "pet")), kanallar)
          for m in ("mri", "pet")}
    tr, va = ic_dogrulama(adni, np.arange(len(adni)), tohum=a.tohum)

    gomme_tr, gomme_va, gomme_o, logit_o = [], [], [], []
    for m in ("mri", "pet"):
        print(f"CNN-{m} (tüm ADNI) eğitiliyor...", flush=True)
        model = cnn.egit(X[m], adni.y, tr, va, n, epok=a.epok, tohum=a.tohum)
        torch.save(model.state_dict(), dizin / f"cnn_{m}.pt")
        gomme_tr.append(cnn.vizit_ciktilari(model, X[m], tr)[0])
        gomme_va.append(cnn.vizit_ciktilari(model, X[m], va)[0])
        g, l = cnn.vizit_ciktilari(model, Xo[m], np.arange(len(oasis)))
        gomme_o.append(g)
        logit_o.append(l)
    G_tr, G_va, G_o = (np.concatenate(g, 1) for g in (gomme_tr, gomme_va, gomme_o))
    params = xgb_optimize(G_tr, adni.y[tr], adni.hasta[tr], n, a.optuna, a.tohum)
    xg, kal = xgb_egit(G_tr, adni.y[tr], G_va, adni.y[va], params, n, a.tohum)
    xg.save_model(dizin / "xgb_fuzyon.json")
    np.savez_compressed(dizin / "gommeler.npz", egitim=G_tr, y_egitim=adni.y[tr], oasis=G_o, y_oasis=oasis.y)

    ol = kal.predict_proba(G_o)
    sonuc = {"adni": adni.ozet(), "oasis": oasis.ozet(), "siniflar": adni.siniflar, "params": params,
             "fuzyon": {**temel(oasis.y, ol), "ga95": bootstrap(oasis.y, ol, oasis.hasta)}}
    # Alt kümeler: Δ-MRI'ı olan / olmayan OASIS vizitleri, izleyici (AV45 / PIB)
    for ad, maske in (("delta_var", np.array([m.get("mri_delta_var") == "1" for m in oasis.meta])),
                      ("AV45", np.array([m["amiloid_izleyici"] == "AV45" for m in oasis.meta])),
                      ("PIB", np.array([m["amiloid_izleyici"] == "PIB" for m in oasis.meta]))):
        if maske.sum() >= 20 and len(np.unique(oasis.y[maske])) == n:
            sonuc[f"fuzyon_{ad}"] = {"n": int(maske.sum()), **temel(oasis.y[maske], ol[maske])}
    np.savez_compressed(dizin / "oasis_tahmin.npz", olasilik=ol, y=oasis.y, hasta=oasis.hasta)
    (SONUC / a.ad / "dis_dogrulama.json").write_text(json.dumps(sonuc, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    m = sonuc["fuzyon"]
    print(f"OASIS-3 dış doğrulama: ROC-AUC {m['roc_auc']:.3f} {m['ga95'].get('roc_auc')}  "
          f"macro-F1 {m['macro_f1']:.3f}  dengeli-doğr {m['dengeli_dogruluk']:.3f}  ECE {m['ece']:.3f}")


if __name__ == "__main__":
    main()
