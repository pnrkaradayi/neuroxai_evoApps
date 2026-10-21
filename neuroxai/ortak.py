"""Ortak ayarlar, yollar ve MNI şablon/atlas hazırlığı.

Şablon: ICBM152 2009c (kafatası dahil T1 + beyin maskesi; nilearn ile bir kez indirilir).
Atlas:  AAL (SPM12) — hipokampus (yama merkezi, Grad-CAM ROI yüzdeleri) ve serebellum (PET SUVR referansı).
Hepsi <islenmis_kok>/sablon altına şablon ızgarasında (1 mm) kaydedilir.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

AYARLAR = json.loads((Path(__file__).parent / "ayarlar.json").read_text(encoding="utf-8"))
VERI_KOK = Path(AYARLAR["veri_kok"])
ISLENMIS = Path(AYARLAR["islenmis_kok"])
SONUC = Path(AYARLAR["sonuc_kok"])
SABLON_DIZIN = ISLENMIS / "sablon"
ENVANTER = ISLENMIS / "envanter.csv"


def sablon_hazirla() -> dict:
    """Şablon ve atlas maskelerini (yoksa) üretir; yolları ve hipokampus merkezini döndürür."""
    hedef = SABLON_DIZIN / "sablon.json"
    if hedef.exists():
        return json.loads(hedef.read_text(encoding="utf-8"))
    import nibabel as nib
    from nilearn import datasets, image

    SABLON_DIZIN.mkdir(parents=True, exist_ok=True)
    icbm = datasets.fetch_icbm152_2009(data_dir=str(SABLON_DIZIN / "nilearn"))
    t1 = image.load_img(icbm["t1"])
    maske = image.resample_to_img(icbm["mask"], t1, interpolation="nearest")
    aal = datasets.fetch_atlas_aal(version="SPM12", data_dir=str(SABLON_DIZIN / "nilearn"))
    aal_img = image.resample_to_img(aal["maps"], t1, interpolation="nearest")
    aal_veri = np.asarray(aal_img.dataobj).astype(np.int32)
    etiket = dict(zip((int(i) for i in aal["indices"]), aal["labels"]))

    def bolge(kosul) -> np.ndarray:
        return np.isin(aal_veri, [i for i, ad in etiket.items() if kosul(ad)]).astype(np.uint8)

    hipokampus = bolge(lambda ad: ad.startswith("Hippocampus"))
    mtl = bolge(lambda ad: ad.startswith(("Hippocampus", "ParaHippocampal", "Amygdala")))
    serebellum = bolge(lambda ad: ad.startswith(("Cerebelum", "Vermis")))

    yollar = {}
    for ad, veri in (("t1", np.asarray(t1.dataobj, dtype=np.float32)),
                     ("maske", (np.asarray(maske.dataobj) > 0.5).astype(np.uint8)),
                     ("aal", aal_veri), ("hipokampus", hipokampus), ("mtl", mtl), ("serebellum", serebellum)):
        yol = SABLON_DIZIN / f"{ad}.nii.gz"
        nib.save(nib.Nifti1Image(veri, t1.affine), yol)
        yollar[ad] = str(yol)
    # Hipokampus merkezi (mm, MNI) — 2.5D koronal yamaların merkezi
    ijk = np.argwhere(hipokampus > 0).mean(axis=0)
    merkez_mm = (t1.affine @ np.append(ijk, 1))[:3]
    bilgi = {"yollar": yollar, "hipokampus_merkez_mm": [round(float(x), 2) for x in merkez_mm],
             # (i, j, k) voksel indisi; SimpleITK dizisinde [k, j, i] olarak erişilir
             "hipokampus_merkez_ijk": [int(round(float(x))) for x in ijk],
             "aal_etiketler": {str(k): v for k, v in etiket.items()}}
    hedef.write_text(json.dumps(bilgi, ensure_ascii=False, indent=2), encoding="utf-8")
    return bilgi


@lru_cache(maxsize=None)
def sablon() -> dict:
    return sablon_hazirla()


def vizit_dizini(veri_seti: str, hasta: str, vizit_kodu: str) -> Path:
    return ISLENMIS / veri_seti / hasta / vizit_kodu


def uyanik_tut():
    """Uzun işler sürerken Windows'un ekranı kapatıp Modern Bekleme'ye (arka plan neredeyse durur) ya da
    uykuya geçmesini engeller. Sistem ayarı değiştirmez; istek bu işlem bitince kendiliğinden kalkar."""
    import ctypes
    import sys
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
