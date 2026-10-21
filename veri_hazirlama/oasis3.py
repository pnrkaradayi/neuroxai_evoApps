"""OASIS-3 veri hazırlama — NeuroXAI-Fusion dış doğrulama kohortu.

Kullanım (bu klasörün içinden):
  py oasis3.py meta    klinik ve görüntü metadata dosyalarını indirir (NITRC-IR şifresi sorulur)
  py oasis3.py sec     metadata'dan dış doğrulama kohortunu seçer (çevrimdışı, şifre gerekmez)
  py oasis3.py indir   seçilen T1w + amiloid PET taramalarını bütçe sınırında indirir (şifre sorulur)

Tüm yollar ve sınırlar ayarlar.json'dan okunur; başka bir bilgisayarda yalnızca
"veri_kok" ve "butce_gb" değerlerini değiştirmek yeterlidir. "indir" yarıda kesilirse
aynı komutla kaldığı yerden devam eder.

Etiketleme (OASIS-3'te EMCI/LMCI yok; her PET vizitine en yakın CDR kaydı kullanılır):
  CN         CDR 0 ve tanı "Cognitively normal" / "No dementia"
  AD         CDR >= 0.5 ve AD tipi demans tanısı
  EMCI/LMCI  CDR 0.5 ve demans dışı tanı; ADNI-GO/2 Logical Memory II (MEMUNITS) +
             eğitim yılı eşikleriyle ayrılır, test yoksa "MCI"
  AD dışı demanslar (DLB, FTD, vasküler, Parkinson vb.) dışlanır.
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from nitrc import GirisHatasi, NitrcIR, uyanik_tut

AYARLAR = json.loads((Path(__file__).parent / "ayarlar.json").read_text(encoding="utf-8"))
KOK = Path(AYARLAR["veri_kok"])
O3 = AYARLAR["oasis3"]
META_DIZIN = KOK / "metadata" / "oasis3"
SECIM_DIZIN = META_DIZIN / "secim"
HAM_DIZIN = KOK / "OASIS3" / "raw"
MANIFEST = KOK / "OASIS3" / "indirilenler.csv"
PROJE = "OASIS3"

AD_DISI = ("dlbd", "vascular demt", "frontotemporal", "non ad", "non-ad", "non dat",
           "dementia/pd", "cannot be primary", "proaph", "impair reversible")
MCI_TANI = {"uncertain dementia", "unc: ques. impairment", "0.5 in memory only", "incipient demt ptp"}
CN_TANI = {"cognitively normal", "no dementia"}
SIRA = {"CN": 0, "EMCI": 1, "MCI": 1, "LMCI": 2, "AD": 3}
TAHMINI_MB = {"T1w": 12, "AV45": 210, "PIB": 130}


# ---------------------------------------------------------------- yardımcılar
def gun(etiket: str) -> int:
    return int(re.search(r"_d(\d+)", etiket).group(1))


def standart_etiket(etiket: str) -> str:
    """PET_json bazı seansları 'OAS30899_PIB_d70' yazar; sunucudaki etiket 'OAS30899_PIB_d0070'."""
    on, g = etiket.rsplit("_d", 1)
    return f"{on}_d{int(g):04d}"


def oku(ad: str) -> list[dict]:
    with open(META_DIZIN / ad, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def yaz(yol: Path, satirlar: list[dict]):
    yol.parent.mkdir(parents=True, exist_ok=True)
    with open(yol, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)


def sayi(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def en_yakin(ogeler, g, max_gun, anahtar=lambda x: x):
    aday = [x for x in ogeler if abs(anahtar(x) - g) <= max_gun]
    return min(aday, key=lambda x: abs(anahtar(x) - g)) if aday else None


def dogal(s: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def mci_alt_tipi(lm, egitim):
    """ADNI-GO/2 Logical Memory II eşikleri: LMCI üst sınırı eğitim yılına göre 8 / 4 / 2."""
    if lm is None or egitim is None:
        return "MCI"
    esik = 8 if egitim >= 16 else 4 if egitim >= 8 else 2
    return "LMCI" if lm <= esik else "EMCI"


def etiketle(cdr: float, tani: str, lm, egitim):
    t = (tani or "").strip().lower()
    if any(k in t for k in AD_DISI):
        return None
    if cdr == 0 and t in CN_TANI:
        return "CN"
    if cdr >= 0.5 and t.startswith(("ad dem", "dat")):
        return "AD"
    if cdr == 0.5 and t in MCI_TANI:
        return mci_alt_tipi(lm, egitim)
    return None


def grup(etiket: str) -> str:
    return etiket if etiket in ("CN", "AD") else "MCI"


# ---------------------------------------------------------------- meta
def komut_meta(ir: NitrcIR):
    deneyler = ir.sonuc(f"/data/projects/{PROJE}/experiments?columns=ID,label")
    veri = next(d for d in deneyler if d["label"] == "OASIS3_data_files")
    istenen = set(O3["meta_dosyalar"])
    bulunan = []
    for tarama in ir.sonuc(f"/data/experiments/{veri['ID']}/scans"):
        for f in ir.tarama_dosyalari(veri["ID"], tarama["ID"]):
            if f["Name"] in istenen:
                bulunan.append(f)
    eksik = istenen - {f["Name"] for f in bulunan}
    toplam = sum(int(f["Size"]) for f in bulunan) / 1e6
    print(f"{len(bulunan)} dosya, toplam {toplam:.1f} MB -> {META_DIZIN}")
    for f in bulunan:
        yeni = ir.dosya_indir(f["URI"], META_DIZIN / f["Name"], int(f["Size"]), f.get("digest") or None)
        print(f"  {'indirildi' if yeni else 'zaten var'}: {f['Name']} ({int(f['Size']) / 1e6:.2f} MB)")
    if eksik:
        print("UYARI: sunucuda bulunamayan dosyalar:", ", ".join(sorted(eksik)))


# ---------------------------------------------------------------- sec
def hasta_vizitleri(h, izler, mr_t1, cdr, lm, egitim):
    """Hasta için en çok eşleşmiş vizit veren izleyiciyi ve vizitlerini döndürür."""
    en_iyi = None
    for iz in O3["pet_izleyiciler"]:  # eşitlikte listedeki ilk izleyici tercih edilir
        mr_basina = {}
        for pg, pet_et in sorted(izler.get(iz, {}).items()):
            mg = en_yakin(mr_t1, pg, O3["mr_pet_max_gun"])
            c = en_yakin(cdr, pg, O3["klinik_max_gun"], anahtar=lambda r: int(r["days_to_visit"]))
            if mg is None or c is None:
                continue
            cg = int(c["days_to_visit"])
            l = en_yakin(lm, cg, O3["klinik_max_gun"], anahtar=lambda x: x[0])
            etiket = etiketle(float(c["CDRTOT"]), c["dx1"], l[1] if l else None, egitim)
            if etiket is None:
                continue
            v = {"hasta": h, "izleyici": iz, "pet_seans": pet_et, "pet_gun": pg,
                 "mr_seans": mr_t1[mg], "mr_gun": mg, "klinik_gun": cg, "CDR": c["CDRTOT"],
                 "CDRSUM": c["CDRSUM"], "MMSE": c["MMSE"], "tani": c["dx1"],
                 "LM_II": l[1] if l else "", "egitim": egitim if egitim is not None else "",
                 "etiket": etiket}
            # Aynı MR seansına iki PET düşerse MR'a en yakın PET kalsın
            eski = mr_basina.get(mg)
            if eski is None or abs(pg - mg) < abs(eski["pet_gun"] - mg):
                mr_basina[mg] = v
        vizitler = sorted(mr_basina.values(), key=lambda v: v["pet_gun"])
        if vizitler and (en_iyi is None or len(vizitler) > len(en_iyi)):
            en_iyi = vizitler
    return en_iyi


def komut_sec(_=None):
    mr_t1 = defaultdict(dict)
    for r in oku("OASIS3_MR_json.csv"):
        if r["scan category"] == "T1w":
            mr_t1[r["subject_id"]][gun(r["label"])] = r["label"]
    pet = defaultdict(lambda: defaultdict(dict))
    for r in oku("OASIS3_PET_json.csv"):
        iz = r["Tracer.Name"] or r["TracerName"]
        if iz in O3["pet_izleyiciler"]:
            pet[r["subject_id"]][iz][gun(r["session_id"])] = standart_etiket(r["session_id"])
    cdr = defaultdict(list)
    for r in oku("OASIS3_UDSb4_cdr.csv"):
        if sayi(r["CDRTOT"]) is not None and sayi(r["days_to_visit"]) is not None:
            cdr[r["OASISID"]].append(r)
    lm = defaultdict(list)
    for r in oku("OASIS3_UDSc1_cognitive_assessments.csv"):
        v, g = sayi(r["MEMUNITS"]), sayi(r["days_to_visit"])
        if v is not None and g is not None and 0 <= v <= 25:
            lm[r["OASISID"]].append((int(g), v))
    egitim = {r["OASISID"]: sayi(r["EDUC"]) for r in oku("OASIS3_demographics.csv")}

    adaylar = []
    for h in sorted(pet):
        if h in mr_t1 and h in cdr:
            v = hasta_vizitleri(h, pet[h], mr_t1[h], cdr[h], lm[h], egitim.get(h))
            if v:
                adaylar.append(v)

    # Hasta sınıfı = en ileri vizit etiketi. Her sınıfta önce çok vizitliler (Δ-haritası
    # için), eşitlikte sabit tohumlu rastgele sıra. Doldurulamayan kota bir sonrakine geçer.
    rnd = random.Random(O3["rastgele_tohum"])
    rnd.shuffle(adaylar)
    gruplar = defaultdict(list)
    for v in adaylar:
        gruplar[grup(max((x["etiket"] for x in v), key=SIRA.get))].append(v)
    secilen, artan = [], 0
    for g in ("AD", "MCI", "CN"):
        uyelar = sorted(gruplar[g], key=lambda v: -len(v))
        n = O3["kohort_kotasi"][g] + artan
        secilen += uyelar[:n]
        artan = max(0, n - len(uyelar))
        print(f"{g:>3}: aday {len(uyelar):4d}  seçilen {min(n, len(uyelar)):4d}  "
              f"(>=2 vizitli aday {sum(len(v) >= 2 for v in uyelar)})")

    # İndirme listesi: vizitlerin T1w ve PET seansları + tek vizitlilere ΔMRI için ek T1w
    vizit_satirlari, indirme = [], {}
    for v in secilen:
        h = v[0]["hasta"]
        for i, x in enumerate(v, 1):
            vizit_satirlari.append({"vizit_no": i, **x})
            indirme.setdefault(x["mr_seans"], {"hasta": h, "seans": x["mr_seans"], "tur": "T1w", "rol": "vizit"})
            indirme.setdefault(x["pet_seans"], {"hasta": h, "seans": x["pet_seans"], "tur": "PET", "rol": "vizit"})
        if len(v) == 1:
            mg = v[0]["mr_gun"]
            digerleri = [g for g in mr_t1[h] if abs(g - mg) >= O3["onceki_mr_min_gun"]]
            if digerleri:
                ek = min(digerleri, key=lambda g: (g > mg, abs(g - mg)))  # önce öncekiler
                indirme.setdefault(mr_t1[h][ek], {"hasta": h, "seans": mr_t1[h][ek], "tur": "T1w",
                                                  "rol": "ek_mr_delta"})
    yaz(SECIM_DIZIN / "oasis3_kohort_vizitler.csv", vizit_satirlari)
    yaz(SECIM_DIZIN / "oasis3_indirme_listesi.csv", list(indirme.values()))

    izleyici = Counter(v[0]["izleyici"] for v in secilen)
    tahmin = sum(TAHMINI_MB["T1w"] if s["tur"] == "T1w" else TAHMINI_MB[s["seans"].split("_")[1]]
                 for s in indirme.values()) / 1e3
    print(f"\nSeçilen hasta: {len(secilen)}  vizit: {len(vizit_satirlari)}  "
          f">=2 vizitli: {sum(len(v) >= 2 for v in secilen)}  izleyici: {dict(izleyici)}")
    print("Vizit etiketleri:", dict(Counter(x["etiket"] for x in vizit_satirlari)))
    print(f"İndirilecek seans: {len(indirme)} "
          f"(T1w {sum(s['tur'] == 'T1w' for s in indirme.values())}, "
          f"PET {sum(s['tur'] == 'PET' for s in indirme.values())})  tahmini boyut ~{tahmin:.0f} GB")
    print(f"Dosyalar: {SECIM_DIZIN}")


# ---------------------------------------------------------------- indir
def komut_indir(ir: NitrcIR):
    with open(SECIM_DIZIN / "oasis3_indirme_listesi.csv", encoding="utf-8", newline="") as f:
        liste = list(csv.DictReader(f))
    deneyler = {standart_etiket(d["label"]): d["ID"]
                for d in ir.sonuc(f"/data/projects/{PROJE}/experiments?columns=ID,label")
                if re.search(r"_d\d+$", d["label"])}
    butce = AYARLAR["butce_gb"]["oasis3"] * 1e9
    kullanilan = sum(p.stat().st_size for p in HAM_DIZIN.rglob("*") if p.is_file()) if HAM_DIZIN.exists() else 0
    kayitli = set()
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8", newline="") as f:
            kayitli = {r["yol"] for r in csv.DictReader(f)}
    print(f"{len(liste)} seans; diskte şu an {kullanilan / 1e9:.1f} GB, bütçe {butce / 1e9:.0f} GB")

    hatalar, art_arda = [], 0
    for i, s in enumerate(liste, 1):
        if art_arda >= 10:  # sunucu çökmüş görünüyor: listeyi boşa tüketme, bekle
            print(f"  ! art arda {art_arda} seans hata verdi; NITRC sunucusu için 15 dk bekleniyor "
                  f"({time.strftime('%H:%M')})", flush=True)
            time.sleep(900)
            art_arda = 0
        eid = deneyler.get(standart_etiket(s["seans"]))
        if not eid:
            print(f"[{i}/{len(liste)}] {s['seans']}: sunucuda bulunamadı, atlandı")
            hatalar.append({"seans": s["seans"], "hata": "sunucuda bulunamadı"})
            continue
        try:
            sonuc = seans_indir(ir, s, eid, butce - kullanilan, kayitli)
        except Exception as e:  # tek seansın hatası tüm indirmeyi durdurmasın
            print(f"[{i}/{len(liste)}] {s['seans']}: HATA — {e} (sonraki çalıştırmada tekrar denenecek)")
            hatalar.append({"seans": s["seans"], "hata": str(e)})
            art_arda += 1
            continue
        art_arda = 0
        if sonuc is None:
            break
        kullanilan += sonuc[0]
        print(f"[{i}/{len(liste)}] {s['seans']} {sonuc[1]} — toplam {kullanilan / 1e9:.2f} GB")

    for ad, satirlar in (("indirme_hatalari.csv", hatalar), ("md5_uyarilari.csv", ir.md5_uyarilari)):
        if satirlar:
            yaz(KOK / "OASIS3" / ad, satirlar)
            print(f"{len(satirlar)} kayıt -> {KOK / 'OASIS3' / ad}")
    print("Bitti. Liste:", MANIFEST)


def seans_indir(ir: NitrcIR, s: dict, eid: str, kalan_butce: float, kayitli: set):
    """Seansın taramasını indirir; (yeni inen bayt, durum metni), bütçe yetmezse None döndürür."""
    taramalar = ir.sonuc(f"/data/experiments/{eid}/scans")
    uygun = [t for t in taramalar if (t["type"] == "T1w" if s["tur"] == "T1w" else t["type"].lower() == "pet")]
    if not uygun:
        raise LookupError(f"{s['tur']} taraması yok")
    tarama = min(uygun, key=lambda t: dogal(t["ID"]))  # birden çok T1w varsa ilk koşu
    dosyalar = ir.tarama_dosyalari(eid, tarama["ID"])
    dizin = HAM_DIZIN / s["hasta"] / s["seans"] / tarama["ID"]
    eksik = [f for f in dosyalar if not ((dizin / f["Name"]).exists()
                                         and (dizin / f["Name"]).stat().st_size == int(f["Size"]))]
    yeni_boyut = sum(int(f["Size"]) for f in eksik)
    if yeni_boyut > kalan_butce:
        print(f"BÜTÇE DOLDU: {s['seans']} ({yeni_boyut / 1e6:.0f} MB) sığmıyor. "
              f"ayarlar.json'da butce_gb.oasis3 artırılıp tekrar çalıştırılabilir.")
        return None
    for f in eksik:
        ir.dosya_indir(f["URI"], dizin / f["Name"], int(f["Size"]), f.get("digest") or None)
    yeni_kayit = [{"hasta": s["hasta"], "seans": s["seans"], "tur": s["tur"], "rol": s["rol"],
                   "tarama": tarama["ID"], "dosya": f["Name"], "boyut": f["Size"],
                   "yol": str(dizin / f["Name"])} for f in dosyalar if str(dizin / f["Name"]) not in kayitli]
    if yeni_kayit:
        ilk = not MANIFEST.exists()
        with open(MANIFEST, "a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(yeni_kayit[0]))
            if ilk:
                w.writeheader()
            w.writerows(yeni_kayit)
        kayitli.update(r["yol"] for r in yeni_kayit)
    return yeni_boyut, f"{tarama['ID']} {'indirildi' if eksik else 'zaten var'} ({yeni_boyut / 1e6:.0f} MB)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("komut", choices=["meta", "sec", "indir"])
    a = p.parse_args()
    if a.komut == "sec":
        return komut_sec()
    uyanik_tut()
    try:
        ir = NitrcIR(O3["kullanici"])
    except GirisHatasi as e:
        sys.exit(str(e))
    {"meta": komut_meta, "indir": komut_indir}[a.komut](ir)


if __name__ == "__main__":
    main()
