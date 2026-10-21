"""Görüntü ön-işleme (proje formu 4.3) — envanterdeki her vizit için:

T1 : DICOM/NIfTI -> N4 bias düzeltme -> MNI (ICBM152 2009) şablonuna rijit + afin kayıt ->
     1 mm şablon ızgarasına yeniden örnekleme -> beyin maskesi -> robust z (medyan/MAD)
PET: (OASIS 4D ise geç çerçeve ortalaması) -> aynı vizitin T1'ine rijit kayıt (karşılıklı bilgi) ->
     T1->MNI dönüşümüyle MNI'ye -> serebellum referanslı SUVR
Çıktı: <islenmis>/<veri_seti>/<hasta>/<vizit_kodu>/{t1,fdg,amiloid}_mni.nii.gz, t1_mni.tfm, qc.json

Kalite kontrolü (qc.json): t1_ncc = kayıtlı T1 ile şablonun beyin içi korelasyonu (düşük = kayıt şüpheli);
{fdg,amiloid}_ncc_t1 = MNI'deki PET ile T1'in beyin içi korelasyonu (kaba PET-T1 hizalama ölçüsü).

Kullanım: .venv\\Scripts\\python -m neuroxai.onisleme [--sinir N] [--paralel K]
"""
import argparse
import csv
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .ortak import AYARLAR, ENVANTER, ISLENMIS, sablon, uyanik_tut, vizit_dizini

OA = AYARLAR["onisleme"]


# ----------------------------------------------------------------- okuma
def goruntu_oku(kaynak: str) -> sitk.Image:
    yol = Path(kaynak)
    if yol.is_dir():  # ADNI DICOM serisi
        okuyucu = sitk.ImageSeriesReader()
        seriler = okuyucu.GetGDCMSeriesIDs(str(yol))
        okuyucu.SetFileNames(max((okuyucu.GetGDCMSeriesFileNames(str(yol), s) for s in seriler), key=len))
        goruntu = okuyucu.Execute()
    else:
        goruntu = sitk.ReadImage(str(yol))
    # Tek dosyalık çok çerçeveli (enhanced) DICOM'lar (X, Y, Z, 1) olarak gelir: tekil 4. boyutu at
    if goruntu.GetDimension() == 4 and goruntu.GetSize()[3] == 1:
        goruntu = sitk.Extract(goruntu, list(goruntu.GetSize()[:3]) + [0], [0, 0, 0, 0])
    if goruntu.GetDimension() == 4:  # dinamik PET: Cast 4D desteklemiyor, çerçeveler pet_statik'te çevrilir
        return goruntu
    return sitk.Cast(goruntu, sitk.sitkFloat32)


def pet_statik(goruntu: sitk.Image, kaynak: str, izleyici: str) -> sitk.Image:
    """4D (dinamik) PET'i, izleyiciye göre geç çerçevelerin süre ağırlıklı ortalamasına indirger."""
    if goruntu.GetDimension() == 3:
        return goruntu
    n = goruntu.GetSize()[3]
    tsv = Path(kaynak.replace(".nii.gz", ".tsv"))
    if tsv.exists():
        with open(tsv, encoding="utf-8") as f:
            satirlar = list(csv.DictReader(f, delimiter="\t"))
        bas = np.array([float(s["Frame_Start"]) for s in satirlar])[:n]
        sure = np.array([float(s["Frame_duration"]) for s in satirlar])[:n]
    else:
        bas, sure = np.arange(n, dtype=float), np.ones(n)
    a, b = (60 * x for x in OA["pet_gec_cerceve_dk"].get(izleyici, [30, 60]))
    orta = bas + sure / 2
    sec = np.where((orta >= a) & (orta <= b))[0]
    if len(sec) == 0:  # çekim geç başlamış / kısa: son 20 dakika
        sec = np.where(orta >= orta.max() - 20 * 60)[0]
    boyut = list(goruntu.GetSize()[:3]) + [0]
    toplam = None
    for i in sec:
        cerceve = sitk.Cast(sitk.Extract(goruntu, boyut, [0, 0, 0, int(i)]), sitk.sitkFloat32) * float(sure[i])
        toplam = cerceve if toplam is None else toplam + cerceve
    return sitk.Cast(toplam / float(sure[sec].sum()), sitk.sitkFloat32)


# ----------------------------------------------------------------- işlem adımları
def n4(goruntu: sitk.Image) -> sitk.Image:
    k = OA["n4_kucultme"]
    maske = sitk.OtsuThreshold(goruntu, 0, 1, 200)
    duzeltici = sitk.N4BiasFieldCorrectionImageFilter()
    duzeltici.SetMaximumNumberOfIterations([50, 50, 30, 20])
    duzeltici.Execute(sitk.Shrink(goruntu, [k] * 3), sitk.Shrink(maske, [k] * 3))
    # SimpleITK'te "/" (DivideReal) her zaman 64-bit üretir; şablonla aynı tip için float32'ye çevir
    return sitk.Cast(goruntu / sitk.Exp(duzeltici.GetLogBiasFieldAsImage(goruntu)), sitk.sitkFloat32)


def kayit(sabit, hareketli, baslangic, kucultme, sigma, iterasyon=200, ornekleme=0.1):
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(ornekleme, seed=42)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=iterasyon,
                                               relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel(kucultme)
    r.SetSmoothingSigmasPerLevel(sigma)
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(baslangic, inPlace=True)  # inPlace=False bileşik dönüşüm döndürür; tipi koru
    r.Execute(sabit, hareketli)
    return baslangic


def t1_mni_kayit(sablon_t1: sitk.Image, t1: sitk.Image) -> sitk.Transform:
    """Önce rijit (moment başlangıcı), sonra afin. Dönüşüm MNI noktasını T1 noktasına götürür."""
    ilk = sitk.CenteredTransformInitializer(sablon_t1, t1, sitk.Euler3DTransform(),
                                            sitk.CenteredTransformInitializerFilter.MOMENTS)
    rijit = kayit(sablon_t1, t1, ilk, [4, 2], [2, 1])
    afin = sitk.AffineTransform(3)
    afin.SetCenter(rijit.GetCenter())
    afin.SetMatrix(rijit.GetMatrix())
    afin.SetTranslation(rijit.GetTranslation())
    if OA["kayit"] == "rijit":
        return rijit
    seviye = OA["afin_kucultme"]  # [4,2,1] tam; [4,2] ~2 mm'de durur (CPU'da çok daha hızlı)
    return kayit(sablon_t1, t1, afin, seviye, [s / 2 if s > 1 else 0 for s in seviye])


def pet_t1_kayit(t1: sitk.Image, pet: sitk.Image) -> sitk.Transform:
    ilk = sitk.CenteredTransformInitializer(t1, pet, sitk.Euler3DTransform(),
                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    seviye = OA["pet_kucultme"]
    return kayit(t1, pet, ilk, seviye, [float(s) for s in seviye], iterasyon=150, ornekleme=0.2)


def maskeli_ncc(a: np.ndarray, b: np.ndarray, maske: np.ndarray) -> float:
    x, y = a[maske] - a[maske].mean(), b[maske] - b[maske].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8))


def robust_z(veri: np.ndarray, maske: np.ndarray) -> np.ndarray:
    ic = veri[maske]
    medyan = np.median(ic)
    mad = np.median(np.abs(ic - medyan)) * 1.4826 + 1e-6
    cikti = np.zeros_like(veri, dtype=np.float32)
    cikti[maske] = (ic - medyan) / mad
    return cikti


# ----------------------------------------------------------------- vizit
_SABLON = {}


def _sablon_yukle():
    if not _SABLON:
        s = sablon()["yollar"]
        _SABLON["t1"] = sitk.ReadImage(s["t1"], sitk.sitkFloat32)
        _SABLON["maske"] = sitk.GetArrayFromImage(sitk.ReadImage(s["maske"])) > 0
        _SABLON["serebellum"] = sitk.GetArrayFromImage(sitk.ReadImage(s["serebellum"])) > 0
        _SABLON["t1_dizi"] = sitk.GetArrayFromImage(_SABLON["t1"])
    return _SABLON


def kaydet(dizi: np.ndarray, referans: sitk.Image, yol: Path):
    g = sitk.GetImageFromArray(dizi.astype(np.float32))
    g.CopyInformation(referans)
    sitk.WriteImage(g, str(yol), useCompression=True)


def vizit_isle(v: dict) -> dict:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(OA["is_basina_thread"])
    sitk.ProcessObject_SetGlobalWarningDisplay(False)  # "non uniform sampling 0.0003" gibi önemsiz uyarılar
    hedef = vizit_dizini(v["veri_seti"], v["hasta"], v["vizit_kodu"])
    qc_yol = hedef / "qc.json"
    if qc_yol.exists():
        return {**json.loads(qc_yol.read_text(encoding="utf-8")), "atlandi": True}
    S = _sablon_yukle()
    hedef.mkdir(parents=True, exist_ok=True)
    qc = {"veri_seti": v["veri_seti"], "hasta": v["hasta"], "vizit_kodu": v["vizit_kodu"]}
    t0 = time.time()
    try:
        t1 = n4(goruntu_oku(v["t1_kaynak"]))
        donusum = t1_mni_kayit(S["t1"], t1)
        t1_mni = sitk.Resample(t1, S["t1"], donusum, sitk.sitkLinear, 0.0)
        t1_dizi = sitk.GetArrayFromImage(t1_mni)
        qc["t1_ncc"] = round(maskeli_ncc(t1_dizi, S["t1_dizi"], S["maske"]), 4)
        kaydet(robust_z(t1_dizi, S["maske"]), S["t1"], hedef / "t1_mni.nii.gz")
        sitk.WriteTransform(donusum, str(hedef / "t1_mni.tfm"))
        for ad, kaynak, izleyici in (("fdg", v["fdg_kaynak"], "FDG"),
                                     ("amiloid", v["amiloid_kaynak"], v["amiloid_izleyici"])):
            if not kaynak:
                continue
            pet = pet_statik(goruntu_oku(kaynak), kaynak, izleyici)
            if v["veri_seti"] != "ADNI":
                # ADNI PET'leri "Uniform 6mm Res" ile gelir; diğerlerini aynı çözünürlüğe getir (form 4.3: 6-8 mm FWHM)
                sigma_mm = OA["pet_yumusatma_fwhm_mm"] / 2.3548
                pet = sitk.SmoothingRecursiveGaussian(pet, sigma_mm)
            rijit = pet_t1_kayit(t1, pet)
            # MNI noktası -> T1 (afin) -> PET (rijit): CompositeTransform son ekleneni önce uygular
            bilesik = sitk.CompositeTransform(3)
            bilesik.AddTransform(rijit)
            bilesik.AddTransform(donusum)
            pet_mni = sitk.GetArrayFromImage(sitk.Resample(pet, S["t1"], bilesik, sitk.sitkLinear, 0.0))
            ref = pet_mni[S["serebellum"] & (pet_mni > 0)]
            suvr = np.where(S["maske"], pet_mni / (ref.mean() + 1e-6), 0).astype(np.float32)
            kaydet(suvr, S["t1"], hedef / f"{ad}_mni.nii.gz")
            qc[f"{ad}_ncc_t1"] = round(maskeli_ncc(pet_mni, t1_dizi, S["maske"]), 4)
            qc[f"{ad}_izleyici"] = izleyici
        qc["durum"] = "tamam"
    except Exception as e:  # tek vizitin hatası tüm işi durdurmasın; qc yazılmaz -> sonraki çalıştırmada denenir
        qc["durum"] = "hata"
        qc["hata"] = f"{type(e).__name__}: {e}"
        qc["iz"] = traceback.format_exc(limit=3)
        return qc
    qc["sure_sn"] = round(time.time() - t0, 1)
    qc_yol.write_text(json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")
    return qc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sinir", type=int, default=0, help="yalnız ilk N vizit (deneme için)")
    p.add_argument("--paralel", type=int, default=OA["paralel_is"])
    p.add_argument("--veri-seti", choices=["ADNI", "OASIS3"], default=None)
    a = p.parse_args()
    uyanik_tut()
    sablon()  # şablon/atlası bir kez, ana süreçte hazırla
    with open(ENVANTER, encoding="utf-8", newline="") as f:
        vizitler = [v for v in csv.DictReader(f) if v["t1_kaynak"]]
    if a.veri_seti:
        vizitler = [v for v in vizitler if v["veri_seti"] == a.veri_seti]
    if a.sinir:
        vizitler = vizitler[:a.sinir]
    gunluk = ISLENMIS / "onisleme_hatalar.csv"
    sayac, hatalar, t0 = 0, [], time.time()
    print(f"{len(vizitler)} vizit, {a.paralel} paralel iş", flush=True)
    with ProcessPoolExecutor(max_workers=a.paralel) as havuz:
        isler = {havuz.submit(vizit_isle, v): v for v in vizitler}
        for is_ in as_completed(isler):
            sayac += 1
            qc = is_.result()
            if qc.get("durum") == "hata":
                satir = {k: qc.get(k, "") for k in ("veri_seti", "hasta", "vizit_kodu", "hata")}
                hatalar.append(satir)
                print(f"  ! HATA {satir['hasta']} {satir['vizit_kodu']}: {satir['hata'][:200]}", flush=True)
            if sayac % 20 == 0 or sayac == len(vizitler):
                gecen = time.time() - t0
                print(f"[{sayac}/{len(vizitler)}] {gecen / 60:.0f} dk, hata {len(hatalar)}, "
                      f"kalan ~{gecen / sayac * (len(vizitler) - sayac) / 60:.0f} dk", flush=True)
    if hatalar:
        with open(gunluk, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hatalar[0]))
            w.writeheader()
            w.writerows(hatalar)
        print(f"{len(hatalar)} hata -> {gunluk}")


if __name__ == "__main__":
    main()
