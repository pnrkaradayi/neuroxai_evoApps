"""ADNI veri hazırlama — NeuroXAI-Fusion eğitim kohortu.

Kullanım (herhangi bir klasörden):
  py veri_hazirlama\\adni.py sec    IDA'dan indirilen tablolardan kohortu ve görüntü listesini seçer
  py veri_hazirlama\\adni.py indir  IDA koleksiyonunun zip dosyalarını indirir (yarıda kalırsa devam eder;
                                    --paralel N ile aynı anda N zip, varsayılan ayarlar.json)
  py veri_hazirlama\\adni.py ac     zip'leri ADNI/raw altına açar, seçilen görüntülerden eksikleri raporlar

Görüntüleri IDA'da toplama (seçimden sonra, tarayıcıda):
  Advanced Search > Image ID alanına secim/ida_image_id_XX.txt içeriği, Modality: MRI+PET,
  Image Types: Original+Pre-processed > SEARCH > Select All > Add To Collection.
  Sonra Data Collections > koleksiyon > All > ADVANCED DOWNLOAD; zip bağlantılarını
  <veri_kok>/ADNI/zip/url_listesi.txt dosyasına satır satır yazın.

Girdi: <veri_kok>/metadata/adni altındaki IDA tabloları (dosya adındaki tarih fark etmez):
  All_Subjects_Key_MRI_*.csv, All_Subjects_Key_PET_*.csv, ARM_*.csv, DXSUM_*.csv

Seçim (proje formu 4.2; seçenekler ayarlar.json > adni):
  - PET vizidi = bir T1w MRI tarihi + ±mr_pet_max_gun içindeki FDG / AV45 (/ tau_pet ise AV1451) PET
    (PET'ler ADNI'nin standart "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res" sürümü)
  - Hasta etiketi (grup_bl): kayıt grubu — ADNI1 MCI = LMCI (ADNIMERGE ile aynı), ADNI-GO/2 Arm
    tablosundan EMCI/LMCI, ADNI3/4 yeni kayıtlarda başlangıç DXSUM tanısı (CN/MCI/AD). SMC dışlanır.
  - Vizit etiketi (dx_vizit): T1w tarihine en yakın DXSUM tanısı (CN/MCI/AD)
  - Hastanın en az min_vizit FDG/AV45'li vizidi olmalı (2 = yalnız boylamsal hastalar; 1 = tek
    vizitliler de, "boylamsal" sütunu ayırır). Δ'da 1.5T/3T karışmasın diye hastanın en çok PET
    vizidi olan alan gücü tutulur.
  - ara_mr_vizitleri: hastanın aynı alan gücündeki PET'siz T1w vizitleri de eklenir (vizit_turu=mr),
    Δ-MRI için daha sık zaman noktası sağlar.
  - Diskte zaten olan görüntüler (ADNI/raw altında I<image_id> klasörü) IDA parçalarına yazılmaz.
"""
import argparse
import csv
import json
import os
import socket
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from nitrc import ag_bekle, uyanik_tut

AYARLAR = json.loads((Path(__file__).parent / "ayarlar.json").read_text(encoding="utf-8"))
KOK = Path(AYARLAR["veri_kok"])
AD_AYAR = AYARLAR["adni"]
META_DIZIN = KOK / "metadata" / "adni"
SECIM_DIZIN = META_DIZIN / "secim"
ZIP_DIZIN = KOK / "ADNI" / "zip"
HAM_DIZIN = KOK / "ADNI" / "raw"

# Arm kodu -> kayıt grubu (ADNI1/GO/2 veri sözlüğü)
ARM_GRUP = {"1": "CN", "4": "CN", "7": "CN", "2": "LMCI", "5": "LMCI", "8": "LMCI",
            "3": "AD", "6": "AD", "9": "AD", "10": "EMCI", "11": "SMC"}
DX = {"1": "CN", "2": "MCI", "3": "AD"}
PET_TANIM = {"18F-FDG": "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res",
             "18F-AV45": "AV45 Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res",
             "18F-AV1451": "AV1451 Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res"}
PET_KISA = {"18F-FDG": "FDG", "18F-AV45": "AV45", "18F-AV1451": "TAU"}
ANA_PET = ("fdg", "av45")  # hastanın seçilmesi bu izleyicilerin vizitlerine göre


def oku(onek: str) -> list[dict]:
    dosyalar = sorted(META_DIZIN.glob(f"{onek}_*.csv"))
    if not dosyalar:
        raise SystemExit(f"{META_DIZIN} içinde {onek}_*.csv bulunamadı")
    with open(dosyalar[-1], encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def yaz(yol: Path, satirlar: list[dict]):
    yol.parent.mkdir(parents=True, exist_ok=True)
    with open(yol, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)


def tarih(s: str) -> date | None:
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def en_yakin(ogeler, t: date, max_gun: int, anahtar):
    aday = [x for x in ogeler if abs((anahtar(x) - t).days) <= max_gun]
    return min(aday, key=lambda x: abs((anahtar(x) - t).days)) if aday else None


def kayit_gruplari(dxsum: dict) -> dict:
    grup = {}
    for r in oku("ARM"):
        # ADNI1 randomize (1), ADNI-GO kayıtlı (2), ADNI2 kayıtlı (3); ilk kayıt fazı geçerli
        if r["ENROLLED"] in ("1", "2", "3") and r["PTID"] not in grup and r["ARM"] in ARM_GRUP:
            grup[r["PTID"]] = ARM_GRUP[r["ARM"]]
    for ptid, dxler in dxsum.items():
        if ptid not in grup and dxler:
            grup[ptid] = dxler[0][1]  # ADNI3/4 yeni kayıt: ilk tanı
    return grup


def t1_sec(goruntuler: list[dict]) -> dict:
    """Aynı gün birden çok T1w varsa: tekrar olmayan, 3D, en küçük image_id."""
    return min(goruntuler, key=lambda r: ("repeat" in r["series_description"].lower(),
                                          r["acquisition_type"] != "3D", int(r["image_id"])))


def vizit_satiri(ptid: str, g: str, mt: date, m: dict, dx, petler: dict) -> dict:
    s = {"hasta": ptid, "grup_bl": g, "mr_tarih": mt.isoformat(), "mr_vizit": m["image_visit"],
         "dx_vizit": dx[1] if dx else "", "alan_gucu": m["magnetic_field_strength"],
         "faz": m["mri_protocol_phase"], "t1_image_id": m["image_id"], "t1_tanim": m["series_description"],
         "vizit_turu": "mr_pet" if petler else "mr"}
    for iz in PET_KISA.values():
        s[f"{iz.lower()}_image_id"] = petler.get(iz, {}).get("image_id", "")
        s[f"{iz.lower()}_tarih"] = petler.get(iz, {}).get("image_date", "")
    return s


def diskteki_goruntuler() -> set[str]:
    """ADNI/raw altında indirilmiş görüntülerin image ID'leri (IDA klasörü I<image_id>)."""
    ids = set()
    if HAM_DIZIN.exists():
        for _, dizinler, _ in os.walk(HAM_DIZIN):
            ids.update(d[1:] for d in dizinler if d[:1] == "I" and d[1:].isdigit())
    return ids


def komut_sec():
    dxsum = defaultdict(list)
    for r in oku("DXSUM"):
        t = tarih(r["EXAMDATE"])
        if t and r["DIAGNOSIS"] in DX:
            dxsum[r["PTID"]].append((t, DX[r["DIAGNOSIS"]]))
    for v in dxsum.values():
        v.sort()
    grup = kayit_gruplari(dxsum)

    t1 = defaultdict(lambda: defaultdict(list))
    for r in oku("All_Subjects_Key_MRI"):
        t = tarih(r["image_date"])
        if r["series_type"] == "T1w" and t:
            t1[r["subject_id"]][t].append(r)
    izleyiciler = {"18F-FDG", "18F-AV45"} | ({"18F-AV1451"} if AD_AYAR["tau_pet"] else set())
    pet = defaultdict(list)
    for r in oku("All_Subjects_Key_PET"):
        t = tarih(r["image_date"])
        radyo = r["radiopharmaceutical"]
        if t and radyo in izleyiciler and PET_TANIM[radyo] == r["pet_description"]:
            pet[r["subject_id"]].append((t, r))

    max_gun = AD_AYAR["mr_pet_max_gun"]
    adaylar, red = {}, Counter()
    for ptid in sorted(set(t1) & set(pet)):
        g = grup.get(ptid)
        if g not in ("CN", "EMCI", "LMCI", "MCI", "AD"):
            red["grup yok / SMC"] += 1
            continue
        # Her PET'i en yakın T1w gününe bağla; aynı güne aynı izleyiciden iki PET düşerse en yakını
        vizit = defaultdict(dict)
        for pt, r in pet[ptid]:
            mt = en_yakin(t1[ptid], pt, max_gun, anahtar=lambda d: d)
            if mt is None:
                continue
            iz = PET_KISA[r["radiopharmaceutical"]]
            eski = vizit[mt].get(iz)
            if eski is None or abs((pt - mt).days) < abs((tarih(eski["image_date"]) - mt).days):
                vizit[mt][iz] = r
        satirlar = []
        for mt, petler in sorted(vizit.items()):
            dx = en_yakin(dxsum[ptid], mt, max_gun, anahtar=lambda x: x[0])
            if dx is not None:
                satirlar.append(vizit_satiri(ptid, g, mt, t1_sec(t1[ptid][mt]), dx, petler))
        ana = [s for s in satirlar if any(s[f"{iz}_image_id"] for iz in ANA_PET)]
        if not ana:
            red["FDG/AV45'li vizit yok"] += 1
            continue
        baskin = Counter(s["alan_gucu"] for s in ana).most_common()
        en_cok = max(n for _, n in baskin)
        alan = max(a for a, n in baskin if n == en_cok)  # eşitlikte 3T
        satirlar = [s for s in satirlar if s["alan_gucu"] == alan]
        n_ana = sum(1 for s in satirlar if any(s[f"{iz}_image_id"] for iz in ANA_PET))
        if n_ana < AD_AYAR["min_vizit"]:
            red[f"< {AD_AYAR['min_vizit']} FDG/AV45'li vizit"] += 1
            continue
        if AD_AYAR["ara_mr_vizitleri"]:
            pet_gunleri = {s["mr_tarih"] for s in satirlar}
            for mt in sorted(t1[ptid]):
                m = t1_sec(t1[ptid][mt])
                if mt.isoformat() not in pet_gunleri and m["magnetic_field_strength"] == alan:
                    dx = en_yakin(dxsum[ptid], mt, max_gun, anahtar=lambda x: x[0])
                    satirlar.append(vizit_satiri(ptid, g, mt, m, dx, {}))
        satirlar.sort(key=lambda s: s["mr_tarih"])
        for i, s in enumerate(satirlar, 1):
            s["vizit_no"] = i
            s["boylamsal"] = int(n_ana >= 2)
        adaylar[ptid] = satirlar
    vizitler = [s for v in adaylar.values() for s in v]

    goruntuler = []
    for s in vizitler:
        goruntuler.append({"image_id": s["t1_image_id"], "hasta": s["hasta"], "tarih": s["mr_tarih"],
                           "modalite": "T1w", "tanim": s["t1_tanim"]})
        for radyo, iz in PET_KISA.items():
            if s[f"{iz.lower()}_image_id"]:
                goruntuler.append({"image_id": s[f"{iz.lower()}_image_id"], "hasta": s["hasta"],
                                   "tarih": s[f"{iz.lower()}_tarih"], "modalite": iz,
                                   "tanim": PET_TANIM[radyo]})
    diskte = diskteki_goruntuler()
    for g in goruntuler:
        g["diskte"] = int(g["image_id"] in diskte)
    yeni = [g for g in goruntuler if not g["diskte"]]

    yaz(SECIM_DIZIN / "adni_kohort_vizitler.csv", vizitler)
    yaz(SECIM_DIZIN / "adni_goruntu_listesi.csv", goruntuler)
    # IDA'ya yapıştırmak için virgülle ayrılmış image ID parçaları — yalnız henüz inmemiş görüntüler
    for eski in SECIM_DIZIN.glob("ida_image_id_*.txt"):
        eski.unlink()
    ids = [g["image_id"] for g in yeni]
    parca = AD_AYAR["ida_parca_boyutu"]
    for i in range(0, len(ids), parca):
        (SECIM_DIZIN / f"ida_image_id_{i // parca + 1:02d}.txt").write_text(",".join(ids[i:i + parca]),
                                                                           encoding="utf-8")

    mb = AD_AYAR["tahmini_mb"]

    def tahmin(gl):
        return sum(mb["T1w" if g["modalite"] == "T1w" else "PET"] for g in gl) / 1e3

    boylamsal = [v for v in adaylar.values() if v[0]["boylamsal"]]
    print(f"Hasta: {len(adaylar)}  {dict(Counter(v[0]['grup_bl'] for v in adaylar.values()))}")
    print(f"  boylamsal (>=2 FDG/AV45 vizit): {len(boylamsal)}  "
          f"{dict(Counter(v[0]['grup_bl'] for v in boylamsal))}")
    print(f"Vizit: {len(vizitler)}  tür: {dict(Counter(s['vizit_turu'] for s in vizitler))}  "
          f"tanı: {dict(Counter(s['dx_vizit'] or '?' for s in vizitler))}")
    print(f"  alan gücü: {dict(Counter(s['alan_gucu'] for s in vizitler))}")
    print(f"Görüntü: {len(goruntuler)} {dict(Counter(g['modalite'] for g in goruntuler))}  "
          f"~{tahmin(goruntuler):.0f} GB")
    print(f"  diskte: {len(goruntuler) - len(yeni)}  indirilecek: {len(yeni)} "
          f"{dict(Counter(g['modalite'] for g in yeni))}  ~{tahmin(yeni):.0f} GB  "
          f"-> {-(-len(ids) // parca)} IDA parçası")
    print(f"Dışlanan hasta: {dict(red)}")
    print(f"Dosyalar: {SECIM_DIZIN}")


def uzak_boyut(url: str) -> int | None:
    """Zip hazırsa boyutu; IDA zip'i henüz hazırlıyorsa (404/503) ya da bağlantı geçersizse None."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        return None


def komut_indir(paralel: int):
    liste = ZIP_DIZIN / "url_listesi.txt"
    urller = [s.strip() for s in liste.read_text(encoding="utf-8").splitlines() if s.strip().startswith("http")]
    # IDA bazen aynı adlı iki farklı zip veriyor (ör. *_dataset.zip): yerel adı numaralandır
    isler, kullanilan = [], Counter()
    for url in urller:
        ad = url.rsplit("/", 1)[1]
        kullanilan[ad] += 1
        if kullanilan[ad] > 1:
            ad = f"{ad[:-4]}_{kullanilan[ad]}.zip"
        isler.append((url, ad))
    # IDA aynı anda birden çok indirmeye çoğunlukla 503 ile yanıt veriyor; varsayılan 1
    with ThreadPoolExecutor(max_workers=paralel) as havuz:
        list(havuz.map(lambda a: zip_indir(a[0], *a[1], len(isler)), enumerate(isler, 1)))


def zip_durumu(yol: Path) -> str:
    """'tam' | 'gecersiz_anahtar' (IDA deneme hakkı doldu) | 'html' | 'yarim'"""
    with open(yol, "rb") as f:
        if f.read(2) != b"PK":
            return "html"  # zip hazır değilken inen hata sayfası
    if not zipfile.is_zipfile(yol):
        return "yarim"
    with zipfile.ZipFile(yol) as zf:
        adlar = zf.namelist()
        if adlar == ["README.txt"]:  # "exceeded the permitted number of attempts ... key invalidated"
            return "gecersiz_anahtar"
    return "tam"


def sil(yol: Path):
    for _ in range(10):  # antivirüs taraması dosyayı kısa süre kilitleyebilir
        try:
            yol.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(3)


def zip_indir(i: int, url: str, ad: str, n: int):
    hedef = ZIP_DIZIN / ad
    isaret = ZIP_DIZIN / f"{ad}.tamam"
    etiket = f"[{i}/{n}] {ad}"
    if isaret.exists():  # doğrulanmış dosyaya bir daha dokunma (yeni bağlantı setiyle bile)
        print(f"{etiket}: zaten var", flush=True)
        return
    if hedef.exists() and zip_durumu(hedef) in ("html", "gecersiz_anahtar"):
        sil(hedef)
    # İnternet kesikse IDA'ya istek atmadan (deneme hakkı yakmadan) bağlantının dönmesini bekle
    ag_bekle(urllib.parse.urlparse(url).hostname)
    boyut = uzak_boyut(url)
    print(f"{etiket}: indiriliyor"
          f"{f' ({boyut / 1e9:.2f} GB)' if boyut else ' (IDA zip hazırlıyor, bekleniyor)'}...", flush=True)
    # IDA zip'i ilk istekte hazırlar (birkaç dk 404/503) ve her isteği "indirme denemesi" sayar;
    # deneme hakkı dolunca bağlantı kalıcı olarak geçersizleşir. Bu yüzden az ve seyrek deneme:
    # -f hata sayfasını dosyaya yazmaz, -C - kopan indirmeyi kaldığı yerden sürdürür.
    # --retry 3 x 180 sn: zip hazırlanırken (~10 dk) bekler ama Wi-Fi'de deneme hakkını hızla tüketmez
    # --speed-limit/--speed-time: 10 dk boyunca 10 KB/s altında kalan (takılmış) bağlantıyı kesip sürdürür
    # ag_arayuzu: birden çok bağlantı varsa (ör. Ethernet + Wi-Fi) indirmeyi belirli arayüze bağlar
    arayuz = ["--interface", AD_AYAR["ag_arayuzu"]] if AD_AYAR.get("ag_arayuzu") else []
    host = urllib.parse.urlparse(url).hostname
    for _ in range(5):  # internet kesintisinde bağlantı dönünce aynı dosyaya devam et
        sonuc = subprocess.run(["curl.exe", "-sS", "-f", "-L", "-C", "-", "--retry", "3", "--retry-all-errors",
                                "--retry-delay", "180", "--connect-timeout", "60", "--speed-limit", "10240",
                                "--speed-time", "600", *arayuz, "-o", str(hedef), url])
        if sonuc.returncode == 0:
            break
        try:
            socket.gethostbyname(host)
            break  # internet var: sorun sunucuda, deneme hakkını korumak için bu dosyada dur
        except OSError:
            ag_bekle(host)
    durum = zip_durumu(hedef) if hedef.exists() else "yok"
    inen = hedef.stat().st_size if hedef.exists() else 0
    if durum == "tam" and (boyut is None or inen == boyut):
        # Tam CRC kontrolü: farklı bağlantı setlerinden birleşen (kaldığı yerden sürdürülen) dosyaları da yakalar
        with zipfile.ZipFile(hedef) as zf:
            bozuk = zf.testzip()
        if bozuk:
            sil(hedef)
            print(f"{etiket}: CRC HATASI ({bozuk}) — dosya silindi, tekrar çalıştırınca baştan inecek", flush=True)
            return
        isaret.touch()
        print(f"{etiket}: tamam ({inen / 1e9:.2f} GB, CRC doğrulandı)", flush=True)
    elif durum == "gecersiz_anahtar":
        sil(hedef)
        print(f"{etiket}: IDA DENEME HAKKI DOLDU — bu dosya için IDA'dan yeni bağlantı gerekli", flush=True)
    else:
        print(f"{etiket}: EKSİK ({inen / 1e9:.2f}{f'/{boyut / 1e9:.2f}' if boyut else ''} GB) — "
              f"tekrar çalıştırın", flush=True)


def komut_ac():
    HAM_DIZIN.mkdir(parents=True, exist_ok=True)
    for z in sorted(ZIP_DIZIN.glob("*.zip")):
        isaret = ZIP_DIZIN / f"{z.name}.acildi"
        if isaret.exists():
            continue
        print(f"{z.name} açılıyor...", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(HAM_DIZIN)
        isaret.touch()
    # IDA klasör yapısı: ADNI/<PTID>/<tanım>/<tarih>/I<image_id>/*.dcm
    bulunan = {p.name[1:] for p in HAM_DIZIN.rglob("I*") if p.is_dir() and p.name[1:].isdigit()}
    with open(SECIM_DIZIN / "adni_goruntu_listesi.csv", encoding="utf-8", newline="") as f:
        istenen = list(csv.DictReader(f))
    eksik = [g for g in istenen if g["image_id"] not in bulunan]
    print(f"Seçilen {len(istenen)} görüntünün {len(istenen) - len(eksik)} tanesi diskte.")
    if eksik:
        yaz(SECIM_DIZIN / "adni_eksik_goruntuler.csv", eksik)
        print(f"Eksik {len(eksik)} görüntü: {SECIM_DIZIN / 'adni_eksik_goruntuler.csv'} "
              f"({dict(Counter(g['modalite'] for g in eksik))})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("komut", choices=["sec", "indir", "ac"])
    p.add_argument("--paralel", type=int, default=AD_AYAR["paralel_indirme"],
                   help="indir: aynı anda indirilecek zip sayısı")
    a = p.parse_args()
    if a.komut == "indir":
        uyanik_tut()
        komut_indir(a.paralel)
    else:
        {"sec": komut_sec, "ac": komut_ac}[a.komut]()


if __name__ == "__main__":
    main()
