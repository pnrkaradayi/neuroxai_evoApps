"""Yama verisini yükleme, örnek seçimi ve hasta bazlı çapraz doğrulama bölmeleri (proje formu 4.7)."""
import csv

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .ortak import ISLENMIS

SINIFLAR = {"dort": ["CN", "EMCI", "LMCI", "AD"], "uc": ["CN", "MCI", "AD"]}
UC_KARSILIK = {"CN": "CN", "EMCI": "MCI", "LMCI": "MCI", "AD": "AD"}


class YamaVerisi:
    """<islenmis>/yamalar/<veri_seti>.npz + meta.csv üzerinde seçilmiş alt küme.

    etiket_tutarli: 4 sınıflı görevde EMCI/LMCI ayrımı yalnız kayıt grubunda var; takip vizitlerinde hasta
    ilerlemiş olabilir (ör. LMCI kayıtlı, vizit tanısı AD). Kayıt grubu o vizitin tanısıyla (dx3) uyuşmayan
    vizitler etiket gürültüsü olduğundan alınmaz.
    """

    def __init__(self, veri_seti: str, gorev: str = "dort", pet: str = "amiloid", delta_gerekli: bool = False,
                 etiket_tutarli: bool = True):
        dizin = ISLENMIS / "yamalar"
        with open(dizin / f"{veri_seti}_meta.csv", encoding="utf-8", newline="") as f:
            meta = list(csv.DictReader(f))
        diziler = np.load(dizin / f"{veri_seti}.npz", mmap_mode="r")
        etiket_sutun = "etiket4" if gorev == "dort" else "dx3"
        siniflar = SINIFLAR[gorev]

        def tutarli(m):
            return not (etiket_tutarli and gorev == "dort") or UC_KARSILIK.get(m["etiket4"]) == m["dx3"]

        sec = [i for i, m in enumerate(meta)
               if m["ornek"] == "1" and m[etiket_sutun] in siniflar and m.get("mri_var") == "1"
               and m.get(f"{pet}_var") == "1" and (not delta_gerekli or m.get("mri_delta_var") == "1")
               and tutarli(m)]
        self.meta = [meta[i] for i in sec]
        self.mri = np.asarray(diziler["mri"][sec], dtype=np.float32)
        self.pet = np.asarray(diziler[pet][sec], dtype=np.float32)
        self.y = np.array([siniflar.index(m[etiket_sutun]) for m in self.meta])
        self.hasta = np.array([m["hasta"] for m in self.meta])
        self.delta_var = np.array([m.get("mri_delta_var") == "1" for m in self.meta])
        self.siniflar, self.pet_adi = siniflar, pet

    def __len__(self):
        return len(self.y)

    def ozet(self) -> str:
        sayim = np.bincount(self.y, minlength=len(self.siniflar))
        return (f"{len(self)} vizit ({int(self.delta_var.sum())} Δ'lı), {len(set(self.hasta))} hasta | "
                + ", ".join(f"{s}={n}" for s, n in zip(self.siniflar, sayim)))


def bolmeler(veri: YamaVerisi, k: int = 5, tohum: int = 2026):
    """Aynı hastanın vizitleri hep aynı katmanda (sızıntı yok), sınıf oranları korunur."""
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=tohum).split(veri.y, veri.y, veri.hasta))


def egitim_oncelleri(veri: YamaVerisi, indis: np.ndarray, k: int = 5, tohum: int = 2026) -> np.ndarray:
    """Her test vizidi için kendi dış katının EĞİTİM kısmındaki sınıf oranları (N, K)."""
    n = len(veri.siniflar)
    oran = {}
    for egitim, test in bolmeler(veri, k, tohum):
        p = np.bincount(veri.y[egitim], minlength=n) / len(egitim)
        for v in test:
            oran[int(v)] = p
    return np.stack([oran[int(v)] for v in indis])


def capraz_uretim(veri: YamaVerisi, egitim_idx: np.ndarray, k: int = 3, tohum: int = 2026):
    """Eğitim katmanını hasta bazlı k parçaya böler: (parçasız eğitim, parça) çiftleri. CNN her seferinde parçayı
    görmeden eğitilir; parçanın embedding'leri XGBoost için 'katman dışı' (sızıntısız) özellik olur."""
    if k == 1:  # çapraz üretim yok (sızıntı gösterimi): CNN eğitim katmanının tamamını görür ve onu gömer
        return [(egitim_idx, egitim_idx)]
    alt = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=tohum + 1)
    return [(egitim_idx[tr], egitim_idx[va])
            for tr, va in alt.split(egitim_idx, veri.y[egitim_idx], veri.hasta[egitim_idx])]


def ic_dogrulama(veri: YamaVerisi, egitim_idx: np.ndarray, oran: float = 0.15, tohum: int = 2026):
    """Eğitim katmanından erken durdurma için hasta bazlı küçük doğrulama kümesi ayırır."""
    k = max(2, round(1 / oran))
    alt = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=tohum)
    tr, va = next(alt.split(egitim_idx, veri.y[egitim_idx], veri.hasta[egitim_idx]))
    return egitim_idx[tr], egitim_idx[va]
