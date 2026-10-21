"""NITRC-IR (XNAT) REST istemcisi — yalnızca Python standart kütüphanesi.

Şifre yalnızca bellekte tutulur; diske, loga veya ekrana yazılmaz.
"""
import base64
import getpass
import hashlib
import http.cookiejar
import json
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.nitrc.org/ir"


class GirisHatasi(RuntimeError):
    pass


def uyanik_tut():
    """İndirme sürerken Windows'un ekranı kapatıp Modern Bekleme'ye (arka plan neredeyse durur) ya da uykuya
    geçmesini engeller. Sistem ayarı değiştirmez; istek bu işlem bitince kendiliğinden kalkar."""
    import ctypes
    import sys
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)


def ag_bekle(host: str, en_fazla_saat: float = 12):
    """İnternet kesikse (alan adı çözülemiyorsa) geri gelene kadar bekler; sunucuya istek atmaz."""
    bitis = time.time() + en_fazla_saat * 3600
    uyarildi = False
    while True:
        try:
            socket.gethostbyname(host)
            if uyarildi:
                print(f"  ağ bağlantısı geri geldi ({time.strftime('%H:%M')})", flush=True)
            return
        except OSError:
            if time.time() > bitis:
                raise
            if not uyarildi:
                print(f"  ! ağ bağlantısı yok ({time.strftime('%H:%M')}), geri gelmesi bekleniyor...", flush=True)
                uyarildi = True
            time.sleep(60)


class NitrcIR:
    def __init__(self, kullanici: str, sifre: str | None = None):
        self.kullanici = kullanici
        self._sifre = sifre if sifre is not None else getpass.getpass(
            f"NITRC-IR şifresi ({kullanici}) [ekranda görünmez]: ")
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.md5_uyarilari: list[dict] = []
        self.giris()

    def giris(self):
        token = base64.b64encode(f"{self.kullanici}:{self._sifre}".encode()).decode()
        req = urllib.request.Request(BASE + "/data/JSESSION",
                                     headers={"Authorization": "Basic " + token})
        try:
            with self._opener.open(req, timeout=60) as r:
                r.read()
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise GirisHatasi("Giriş başarısız: kullanıcı adı veya şifre hatalı.") from None
            raise

    def _ac(self, yol: str, basliklar: dict | None = None):
        url = yol if yol.startswith("http") else BASE + yol
        req = urllib.request.Request(url, headers=basliklar or {})
        bekleme = (10, 30, 60, 120, 300)  # sunucu hatalarında artan aralıklarla yeniden dene
        deneme, girildi = 0, False
        while True:
            try:
                return self._opener.open(req, timeout=300)
            except urllib.error.HTTPError as e:
                if e.code == 401 and not girildi:
                    self.giris()  # uzun indirmelerde oturum düşebilir
                    girildi = True
                    continue
                if e.code < 500 or deneme >= len(bekleme):
                    raise
                # NITRC düşen oturumu bazen 500 ile bildiriyor; ilk 500'de yeniden giriş de dene
                if not girildi:
                    try:
                        self.giris()
                        girildi = True
                    except Exception:
                        pass
                time.sleep(bekleme[deneme])
                deneme += 1
            except urllib.error.URLError as e:
                if isinstance(e.reason, socket.gaierror):
                    # İnternet kesildiyse listeyi hızla tüketme: bağlantı gelene kadar bekle (12 sa sınırı)
                    ag_bekle(urllib.parse.urlparse(url).hostname)
                elif isinstance(e.reason, ssl.SSLError) and deneme < len(bekleme):
                    # NITRC'nin bir sunucusu arada eski sertifika veriyor: biraz bekleyip başka sunucuya düş
                    time.sleep(bekleme[deneme])
                    deneme += 1
                else:
                    raise

    def sonuc(self, yol: str) -> list[dict]:
        ayirac = "&" if "?" in yol else "?"
        with self._ac(f"{yol}{ayirac}format=json") as r:
            return json.load(r)["ResultSet"]["Result"]

    def tarama_dosyalari(self, deney_id: str, tarama_id: str) -> list[dict]:
        tid = urllib.parse.quote(tarama_id, safe="")
        return self.sonuc(f"/data/experiments/{deney_id}/scans/{tid}/files")

    def dosya_indir(self, uri: str, hedef: Path, boyut: int | None = None,
                    md5: str | None = None, deneme: int = 3) -> bool:
        """Dosyayı indirir; zaten tam ve doğruysa atlar. .part ile kaldığı yerden devam eder.
        True: yeni indirildi, False: zaten vardı."""
        if hedef.exists() and (boyut is None or hedef.stat().st_size == boyut):
            return False
        hedef.parent.mkdir(parents=True, exist_ok=True)
        part = hedef.with_name(hedef.name + ".part")
        for i in range(1, deneme + 1):
            try:
                mevcut = part.stat().st_size if part.exists() else 0
                basliklar = {"Range": f"bytes={mevcut}-"} if mevcut else {}
                with self._ac(urllib.parse.quote(uri, safe="/:?=&"), basliklar) as r:
                    mod = "ab" if mevcut and r.status == 206 else "wb"
                    with open(part, mod) as f:
                        while blok := r.read(1 << 20):
                            f.write(blok)
                if boyut is not None and part.stat().st_size != boyut:
                    raise IOError(f"boyut uyuşmuyor ({part.stat().st_size} != {boyut})")
                bulunan = _md5(part) if md5 else None
                if md5 and bulunan != md5:
                    if i < deneme:
                        part.unlink()
                        raise IOError("MD5 uyuşmuyor, dosya silindi")
                    # Boyut doğru ve MD5 her denemede aynı biçimde tutmuyorsa sorun büyük olasılıkla
                    # sunucudaki eski özet kaydıdır: dosyayı sakla, uyarı listesine yaz.
                    self.md5_uyarilari.append({"dosya": str(hedef), "sunucu_md5": md5, "inen_md5": bulunan})
                    print(f"    ! {hedef.name}: MD5 {deneme} denemede de tutmadı, boyut doğru — saklandı, listeye yazıldı")
                part.replace(hedef)
                return True
            except (urllib.error.URLError, IOError, TimeoutError) as e:
                if i == deneme:
                    raise
                print(f"    ! {hedef.name}: {e} — {5 * i} sn sonra tekrar denenecek")
                time.sleep(5 * i)
        return True


def _md5(yol: Path) -> str:
    h = hashlib.md5()
    with open(yol, "rb") as f:
        while blok := f.read(1 << 20):
            h.update(blok)
    return h.hexdigest()
