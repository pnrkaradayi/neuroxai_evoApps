"""Açıklanabilirlik (proje formu 4.8) — deney.py ve dis_dogrulama.py çıktıları üzerinde:

1. Grad-CAM (görüntü düzeyi): her CV katının CNN'i yalnız kendi test vizitlerinde (örneklem dışı) çalıştırılır;
   orta koronal dilimde sınıf bazında ortalama ısı haritası ve AAL bölgelerine yüzdesel dağılım.
2. TreeSHAP (karar düzeyi): son füzyon XGBoost'unda |SHAP| -> MRI ve PET embedding'lerinin sınıf bazında payı.
3. Kanal katkısı (pertürbasyon): orijinal / Δ / kenar kanalı sıfırlandığında doğru sınıf olasılığındaki
   ortalama düşüş, modalite içinde yüzdeye çevrilir.
Çıktılar: <sonuc>/<ad>/xai/ (json + png)

Kullanım: .venv\\Scripts\\python -m neuroxai.xai --ad ana
"""
import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import xgboost as xgb

from . import cnn
from .deney import KANAL, kanal_sec
from .ortak import ISLENMIS, SONUC, sablon, uyanik_tut
from .veri import YamaVerisi

BOLGE_GRUBU = {"Hippocampus": "Hipokampus", "ParaHippocampal": "Parahipokampal", "Amygdala": "Amigdala",
               "Fusiform": "Fusiform", "Temporal": "Temporal korteks", "Thalamus": "Talamus",
               "Insula": "İnsula", "Putamen": "Bazal gangliyon", "Pallidum": "Bazal gangliyon",
               "Caudate": "Bazal gangliyon", "Lingual": "Lingual", "Cingul": "Singulat"}


def bolge_adi(aal: str) -> str:
    for anahtar, ad in BOLGE_GRUBU.items():
        if aal.startswith(anahtar) or anahtar in aal:
            return ad
    return "Diğer"


def softmax(lg):
    e = np.exp(lg - lg.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ad", required=True)
    p.add_argument("--en-fazla", type=int, default=400, help="Grad-CAM/pertürbasyon için kat başına vizit sınırı")
    a = p.parse_args()
    uyanik_tut()  # ekran kapansa da uzun eğitim sürsün
    kok = SONUC / a.ad
    ayar = json.loads((kok / "ayarlar.json").read_text(encoding="utf-8"))
    cikti = kok / "xai"
    cikti.mkdir(exist_ok=True)
    veri = YamaVerisi("ADNI", ayar["gorev"], ayar["pet"], ayar["delta_gerekli"], ayar.get("etiket_tutarli", True))
    kanallar = ayar["kanallar"].split(",")
    X = {"mri": kanal_sec(cnn.girdi_olcekle(veri.mri), kanallar),
         "pet": kanal_sec(cnn.girdi_olcekle(veri.pet), kanallar)}
    aal = np.load(ISLENMIS / "yamalar" / "aal_yamalar.npy")[1]  # orta dilim
    etiket = {int(k): v for k, v in sablon()["aal_etiketler"].items()}
    n = len(veri.siniflar)

    cam_toplam = {m: np.zeros((n, *aal.shape)) for m in X}
    cam_sayi = {m: np.zeros(n) for m in X}  # her modalitenin kendi CNN'inin doğru sınıfladığı vizitler
    bolge_kutle = {m: defaultdict(lambda: np.zeros(n)) for m in X}
    kanal_dusus = {m: defaultdict(list) for m in X}
    for kat_dizin in sorted(kok.glob("kat*")):
        test = np.load(kat_dizin / "tahminler.npz")["indis"][:a.en_fazla]
        for m in X:
            # Katın çapraz üretim modelleri (cnn_<m>_<j>.pt; eski deneylerde tek cnn_<m>.pt): hepsi test vizitlerini
            # görmemiştir; olasılık, pertürbasyon ve Grad-CAM üç modelin ortalaması
            modeller = []
            for yol in sorted(kat_dizin.glob(f"cnn_{m}_*.pt")) or [kat_dizin / f"cnn_{m}.pt"]:
                model = cnn.model_olustur(n, onegitimli=False)
                model.load_state_dict(torch.load(yol))
                model.eval()
                modeller.append(model)

            def olasilik(Xa, idx):
                return np.mean([softmax(cnn.vizit_ciktilari(md, Xa, idx)[1]) for md in modeller], 0)

            ol = olasilik(X[m], test)
            temel_ol = ol[np.arange(len(test)), veri.y[test]]
            for kanal, i in KANAL.items():  # pertürbasyon: kanalı sıfırla
                if kanal not in kanallar:
                    continue
                Xp = X[m][test].copy()
                Xp[:, :, i] = 0
                ol2 = olasilik(Xp, np.arange(len(test)))
                kanal_dusus[m][kanal].extend((temel_ol - ol2[np.arange(len(test)), veri.y[test]]).tolist())
            for sira, v in enumerate(test):
                if ol[sira].argmax() != veri.y[v]:
                    continue  # yalnız doğru sınıflanan vizitlerin ısı haritaları
                cam = np.mean([cnn.gradcam(md, X[m][v, 1], int(veri.y[v])) for md in modeller], 0)
                cam = cam / (cam.max() + 1e-8)
                cam_toplam[m][veri.y[v]] += cam
                cam_sayi[m][veri.y[v]] += 1
                toplam = cam[aal > 0].sum() + 1e-8
                for lbl in np.unique(aal[aal > 0]):
                    bolge_kutle[m][bolge_adi(etiket.get(int(lbl), "?"))][veri.y[v]] += cam[aal == lbl].sum() / toplam

    rapor = {"siniflar": veri.siniflar, "gradcam_bolge_yuzde": {}, "kanal_katki_yuzde": {}}
    for m in X:
        yuzde = {}
        for s, sinif in enumerate(veri.siniflar):
            sayi = max(cam_sayi[m][s], 1)
            yuzde[sinif] = dict(sorted(((b, round(100 * k[s] / sayi, 1)) for b, k in bolge_kutle[m].items()),
                                       key=lambda x: -x[1]))
        rapor["gradcam_bolge_yuzde"][m] = yuzde
        dusus = {k: max(float(np.mean(v)), 0.0) for k, v in kanal_dusus[m].items()}
        toplam = sum(dusus.values()) + 1e-8
        rapor["kanal_katki_yuzde"][m] = {k: round(100 * v / toplam, 1) for k, v in dusus.items()}
        rapor.setdefault("kanal_olasilik_dususu", {})[m] = {k: round(v, 4) for k, v in dusus.items()}

    # TreeSHAP: son füzyon modeli (dis_dogrulama.py) varsa onu, yoksa ilk katın modelini kullan
    son = kok / "son_model"
    kaynak = son if (son / "xgb_fuzyon.json").exists() else sorted(kok.glob("kat*"))[0]
    model = xgb.XGBClassifier()
    model.load_model(kaynak / "xgb_fuzyon.json")
    G = np.load(kaynak / "gommeler.npz")["egitim"]
    deger = np.array(shap.TreeExplainer(model).shap_values(G))  # (örnek, özellik, sınıf) ya da (sınıf, ...)
    if deger.shape[0] == n and deger.shape[1] == len(G):
        deger = np.moveaxis(deger, 0, -1)
    mutlak = np.abs(deger).mean(0).astype(float)  # (özellik, sınıf)
    yari = mutlak.shape[0] // 2  # [MRI ‖ PET] özellikleri eşit boyutlu (1280 ya da PCA bileşeni)
    rapor["shap_modalite_yuzde"] = {
        sinif: {"MRI": round(100 * float(mutlak[:yari, s].sum() / mutlak[:, s].sum()), 1),
                "PET": round(100 * float(mutlak[yari:, s].sum() / mutlak[:, s].sum()), 1)}
        for s, sinif in enumerate(veri.siniflar)}
    rapor["shap_kaynak"] = str(kaynak)
    (cikti / "xai_ozet.json").write_text(json.dumps(rapor, indent=1, ensure_ascii=False, default=float),
                                         encoding="utf-8")

    # Şekil: sınıf bazında ortalama Grad-CAM (MRI ve PET), ortalama yama üstünde
    fig, eksen = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    for r, m in enumerate(X):
        for s, sinif in enumerate(veri.siniflar):
            ax = eksen[r, s]
            secili = veri.y == s
            ax.imshow(X[m][secili, 1, 0].mean(0), cmap="gray")
            ax.imshow(cam_toplam[m][s] / max(cam_sayi[m][s], 1), cmap="jet", alpha=0.4)
            ax.set_title(f"{'MRI' if m == 'mri' else ayar['pet'].upper()} – {sinif} (n={int(cam_sayi[m][s])})",
                         fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(cikti / "gradcam_sinif_ortalama.png", dpi=150)
    print(json.dumps({k: rapor[k] for k in ("kanal_katki_yuzde", "shap_modalite_yuzde")}, indent=1,
                     ensure_ascii=False, default=float))
    print("->", cikti)


if __name__ == "__main__":
    main()
