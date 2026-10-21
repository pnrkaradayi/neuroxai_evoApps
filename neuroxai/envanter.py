"""ADNI (DICOM) ve OASIS-3 (NIfTI) vizitlerini tek tabloda toplar -> <islenmis>/envanter.csv

Her satır bir MRI vizidi: T1 kaynağı ve (varsa) aynı vizitin FDG / amiloid PET kaynağı.
  etiket4: CN/EMCI/LMCI/AD (ADNI: kayıt grubu; OASIS: CDR + Logical Memory eşlemesi). "MCI" (alt tipi
           bilinmeyen) ve etiketsiz satırlar 4 sınıflı analizde kullanılmaz (boş).
  dx3:     vizit tanısı CN/MCI/AD.
  ornek:   1 = sınıflandırma örneği; 0 = yalnız Δ-MRI için önceki vizit (OASIS ek_mr_delta).

Kullanım: .venv\\Scripts\\python -m neuroxai.envanter
"""
import csv
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

from .ortak import AYARLAR, ENVANTER, VERI_KOK

DORT = {"CN", "EMCI", "LMCI", "AD"}
UC = {"CN": "CN", "EMCI": "MCI", "LMCI": "MCI", "MCI": "MCI", "AD": "AD"}


def oku(yol) -> list[dict]:
    with open(yol, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def adni_dicom_dizinleri() -> dict[str, str]:
    """image_id -> DICOM klasörü (IDA yapısı ADNI/<PTID>/<tanım>/<tarih>/I<image_id>)."""
    kok = VERI_KOK / "ADNI" / "raw" / "ADNI"
    dizinler = {}
    for yol, alt, _ in os.walk(kok):
        for d in alt:
            if d[:1] == "I" and d[1:].isdigit():
                dizinler[d[1:]] = os.path.join(yol, d)
    return dizinler


def adni_satirlari() -> list[dict]:
    dizin = adni_dicom_dizinleri()
    vizitler = oku(AYARLAR["adni_vizitler"])
    ilk = {}
    for v in vizitler:
        ilk[v["hasta"]] = min(ilk.get(v["hasta"], v["mr_tarih"]), v["mr_tarih"])
    satirlar = []
    for v in vizitler:
        gun = (date.fromisoformat(v["mr_tarih"]) - date.fromisoformat(ilk[v["hasta"]])).days
        satirlar.append({
            "veri_seti": "ADNI", "hasta": v["hasta"], "vizit_kodu": v["mr_tarih"], "gun": gun,
            "etiket4": v["grup_bl"] if v["grup_bl"] in DORT else "", "etiket_kayit": v["grup_bl"],
            "dx3": v["dx_vizit"], "alan_gucu": v["alan_gucu"],
            "t1_kaynak": dizin.get(v["t1_image_id"], ""),
            "fdg_kaynak": dizin.get(v["fdg_image_id"], "") if v["fdg_image_id"] else "",
            "amiloid_kaynak": dizin.get(v["av45_image_id"], "") if v["av45_image_id"] else "",
            "amiloid_izleyici": "AV45" if v["av45_image_id"] else "", "ornek": 1})
    return satirlar


def oasis_dosya(hasta: str, seans: str, desen: str) -> str:
    bulunan = sorted((VERI_KOK / "OASIS3" / "raw" / hasta / seans).glob(desen))
    return str(bulunan[0]) if bulunan else ""


def oasis_satirlari() -> list[dict]:
    vizitler = oku(AYARLAR["oasis_vizitler"])
    satirlar, mr_gunleri = [], defaultdict(set)
    for v in vizitler:
        satirlar.append({
            "veri_seti": "OASIS3", "hasta": v["hasta"], "vizit_kodu": v["mr_seans"], "gun": int(v["mr_gun"]),
            "etiket4": v["etiket"] if v["etiket"] in DORT else "", "etiket_kayit": v["etiket"],
            "dx3": UC.get(v["etiket"], ""), "alan_gucu": "",
            "t1_kaynak": oasis_dosya(v["hasta"], v["mr_seans"], "*/*T1w.nii.gz"), "fdg_kaynak": "",
            "amiloid_kaynak": oasis_dosya(v["hasta"], v["pet_seans"], "pet1/*_pet.nii.gz"),
            "amiloid_izleyici": v["izleyici"], "ornek": 1})
        mr_gunleri[v["hasta"]].add(v["mr_seans"])
    # Tek vizitli hastalar için Δ-MRI'a yetecek önceki T1 (yalnız MRI, sınıflandırma örneği değil)
    for s in oku(AYARLAR["oasis_indirme_listesi"]):
        if s["rol"] == "ek_mr_delta" and s["seans"] not in mr_gunleri[s["hasta"]]:
            satirlar.append({
                "veri_seti": "OASIS3", "hasta": s["hasta"], "vizit_kodu": s["seans"],
                "gun": int(s["seans"].rsplit("_d", 1)[1]), "etiket4": "", "etiket_kayit": "", "dx3": "",
                "alan_gucu": "", "t1_kaynak": oasis_dosya(s["hasta"], s["seans"], "*/*T1w.nii.gz"),
                "fdg_kaynak": "", "amiloid_kaynak": "", "amiloid_izleyici": "", "ornek": 0})
    return satirlar


def main():
    satirlar = adni_satirlari() + oasis_satirlari()
    satirlar.sort(key=lambda s: (s["veri_seti"], s["hasta"], s["gun"]))
    onceki = {}
    for s in satirlar:  # vizit_no hastanın tüm (örnek + ek) MRI vizitleri arasında sıra
        anahtar = (s["veri_seti"], s["hasta"])
        onceki[anahtar] = onceki.get(anahtar, 0) + 1
        s["vizit_no"] = onceki[anahtar]
    ENVANTER.parent.mkdir(parents=True, exist_ok=True)
    with open(ENVANTER, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    for vs in ("ADNI", "OASIS3"):
        alt = [s for s in satirlar if s["veri_seti"] == vs]
        eksik_t1 = sum(not s["t1_kaynak"] for s in alt)
        print(f"{vs}: {len({s['hasta'] for s in alt})} hasta, {len(alt)} vizit "
              f"(örnek {sum(s['ornek'] for s in alt)}), T1 eksik {eksik_t1}, "
              f"FDG {sum(bool(s['fdg_kaynak']) for s in alt)}, amiloid {sum(bool(s['amiloid_kaynak']) for s in alt)}")
    print("->", ENVANTER)


if __name__ == "__main__":
    main()
