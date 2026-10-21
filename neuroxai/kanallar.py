"""Kanal oluşturma ve 2.5D yamalar (proje formu 4.5).

Her vizit ve modalite (MRI, FDG, amiloid) için hipokampus merkezinden geçen koronal dilim k ve komşuları
k±komsu_dilim_mm alınır; her dilimde 3 kanal üretilir:
  0 orijinal  : MRI robust z / PET SUVR'ın beyin içi z-skoru
  1 Δ (delta) : aynı hastanın bir önceki vizidinden fark (aynı modalite/izleyici), ±3σ kırpılıp σ'ya bölünür;
                önceki vizit yoksa 0 (delta_var=0)
  2 kenar     : Sobel gradyan büyüklüğü + |Laplacian| (her biri %99'luk dilime göre [0,1]), ağırlıklı toplam
Çıktı: <islenmis>/yamalar/<veri_seti>.npz
  mri, fdg, amiloid : (N, 3 dilim, 3 kanal, 128, 128) float16  (yoksa sıfır, *_var=0)
  meta.csv          : satır başına vizit bilgisi ve maskeler

Kullanım: .venv\\Scripts\\python -m neuroxai.kanallar [--veri-seti ADNI|OASIS3]
"""
import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .ortak import AYARLAR, ENVANTER, ISLENMIS, sablon, vizit_dizini

YA = AYARLAR["yama"]
MODALITELER = ("mri", "fdg", "amiloid")
DOSYA = {"mri": "t1_mni.nii.gz", "fdg": "fdg_mni.nii.gz", "amiloid": "amiloid_mni.nii.gz"}


def dilim_indisleri() -> list[int]:
    j = sablon()["hipokampus_merkez_ijk"][1]
    adim = YA["komsu_dilim_mm"]  # şablon 1 mm
    return [j - adim, j, j + adim]


def yama_kes(hacim: np.ndarray, j: int) -> np.ndarray:
    """SimpleITK dizisi [k, j, i] -> koronal dilim j'den hipokampus merkezli boyut×boyut yama (üst = superior)."""
    b = YA["boyut"] // 2
    i0, _, k0 = sablon()["hipokampus_merkez_ijk"]
    dilim = hacim[:, j, :]  # (k, i)
    dilim = np.pad(dilim, b, mode="constant")
    return dilim[k0:k0 + 2 * b, i0:i0 + 2 * b][::-1].copy()


def kenar(dilim: np.ndarray, maske: np.ndarray) -> np.ndarray:
    def olcekle(x):
        ust = np.percentile(x[maske], 99) if maske.any() else x.max()
        return np.clip(x / (ust + 1e-6), 0, 1)
    sobel = np.hypot(ndimage.sobel(dilim, 0), ndimage.sobel(dilim, 1))
    lap = np.abs(ndimage.laplace(dilim))
    w = YA["kenar_agirlik"]
    return (w["sobel"] * olcekle(sobel) + w["laplacian"] * olcekle(lap)) * maske


def hacim_oku(yol) -> np.ndarray | None:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(yol))) if yol.exists() else None


def hasta_yamalari(vizitler: list[dict]) -> list[dict]:
    """Bir hastanın tüm vizitleri (gün sırasına göre); Δ için önceki vizit gerektiğinden hasta bazlı işlenir."""
    maske3d = sitk.GetArrayFromImage(sitk.ReadImage(sablon()["yollar"]["maske"])) > 0
    js = dilim_indisleri()
    maske_y = [yama_kes(maske3d.astype(np.float32), j) > 0.5 for j in js]
    onceki = {}
    cikti = []
    for v in sorted(vizitler, key=lambda v: int(v["gun"])):
        d = vizit_dizini(v["veri_seti"], v["hasta"], v["vizit_kodu"])
        kayit = {"meta": {k: v[k] for k in ("veri_seti", "hasta", "vizit_kodu", "gun", "vizit_no", "etiket4",
                                             "dx3", "ornek", "amiloid_izleyici")}}
        for m in MODALITELER:
            hacim = hacim_oku(d / DOSYA[m])
            anahtar = m if m != "amiloid" else f"amiloid_{v['amiloid_izleyici']}"
            if hacim is None:
                kayit[m] = None
                continue
            if m != "mri":  # PET: SUVR'ı beyin içinde z-skora çevir
                ic = hacim[maske3d]
                hacim = np.where(maske3d, (hacim - ic.mean()) / (ic.std() + 1e-6), 0)
            yamalar = np.zeros((3, 3, YA["boyut"], YA["boyut"]), dtype=np.float32)
            onc = onceki.get(anahtar)
            if onc is not None:
                fark = (hacim - onc)[maske3d]
                sigma = fark.std() + 1e-6
                delta3d = np.where(maske3d, np.clip(hacim - onc, -YA["delta_kirpma_sigma"] * sigma,
                                                    YA["delta_kirpma_sigma"] * sigma) / sigma, 0)
            for s, j in enumerate(js):
                orj = yama_kes(hacim, j)
                yamalar[s, 0] = orj
                if onc is not None:
                    yamalar[s, 1] = yama_kes(delta3d, j)
                yamalar[s, 2] = kenar(orj, maske_y[s])
            kayit[m] = yamalar.astype(np.float16)
            kayit["meta"][f"{m}_var"] = 1
            kayit["meta"][f"{m}_delta_var"] = int(onc is not None)
            if onc is not None:
                kayit["meta"][f"{m}_delta_gun"] = int(v["gun"]) - onceki[anahtar + "_gun"]
            onceki[anahtar], onceki[anahtar + "_gun"] = hacim, int(v["gun"])
        cikti.append(kayit)
    return cikti


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--veri-seti", choices=["ADNI", "OASIS3"], default=None)
    a = p.parse_args()
    ncc_min = AYARLAR["qc"]["t1_ncc_min"]

    def qc_gecti(v):  # ön-işlemesi bitmiş ve T1 kaydı başarılı vizit
        yol = vizit_dizini(v["veri_seti"], v["hasta"], v["vizit_kodu"]) / "qc.json"
        return yol.exists() and json.loads(yol.read_text(encoding="utf-8")).get("t1_ncc", 0) >= ncc_min

    with open(ENVANTER, encoding="utf-8", newline="") as f:
        envanter = [v for v in csv.DictReader(f) if (a.veri_seti is None or v["veri_seti"] == a.veri_seti)]
    elenen = [v for v in envanter if not qc_gecti(v)]
    envanter = [v for v in envanter if qc_gecti(v)]
    if elenen:
        print(f"QC/ön-işleme eksik nedeniyle alınmayan {len(elenen)} vizit: "
              + ", ".join(f"{v['hasta']} {v['vizit_kodu']}" for v in elenen[:10]))
    hastalar = defaultdict(list)
    for v in envanter:
        hastalar[(v["veri_seti"], v["hasta"])].append(v)
    hedef = ISLENMIS / "yamalar"
    hedef.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as havuz:
        sonuclar = list(havuz.map(hasta_yamalari, hastalar.values(), chunksize=4))
    kayitlar = [k for liste in sonuclar for k in liste]
    for vs in ("ADNI", "OASIS3"):
        alt = [k for k in kayitlar if k["meta"]["veri_seti"] == vs]
        if not alt:
            continue
        bos = np.zeros((3, 3, YA["boyut"], YA["boyut"]), dtype=np.float16)
        diziler = {m: np.stack([k[m] if k[m] is not None else bos for k in alt]) for m in MODALITELER}
        np.savez(hedef / f"{vs}.npz", **diziler)
        alanlar = sorted({a for k in alt for a in k["meta"]})
        with open(hedef / f"{vs}_meta.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=alanlar)
            w.writeheader()
            for k in alt:
                w.writerow({a: k["meta"].get(a, 0 if a.endswith("_var") else "") for a in alanlar})
        print(f"{vs}: {len(alt)} vizit -> {hedef / (vs + '.npz')}  "
              + "  ".join(f"{m}={sum(k[m] is not None for k in alt)}" for m in MODALITELER))
    # Grad-CAM ROI yüzdeleri için AAL etiket yamaları (şablon uzayında sabit)
    aal = sitk.GetArrayFromImage(sitk.ReadImage(sablon()["yollar"]["aal"]))
    np.save(hedef / "aal_yamalar.npy", np.stack([yama_kes(aal, j) for j in dilim_indisleri()]).astype(np.int16))


if __name__ == "__main__":
    main()
