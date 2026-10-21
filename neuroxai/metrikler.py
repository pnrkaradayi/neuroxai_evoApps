"""Değerlendirme ölçütleri (proje formu 4.7, 4.9): ROC-AUC, macro-F1, dengeli doğruluk, ECE, duyarlılık/özgüllük,
PR-AUC, bootstrap %95 GA; modeller arası karşılaştırma için McNemar."""
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)


def ece(y: np.ndarray, olasilik: np.ndarray, kutu: int = 15) -> float:
    """Beklenen kalibrasyon hatası (en olası sınıfın güveni üzerinden)."""
    guven = olasilik.max(1)
    dogru = (olasilik.argmax(1) == y).astype(float)
    sinirlar = np.linspace(0, 1, kutu + 1)
    toplam = 0.0
    for a, b in zip(sinirlar[:-1], sinirlar[1:]):
        m = (guven > a) & (guven <= b)
        if m.any():
            toplam += m.mean() * abs(dogru[m].mean() - guven[m].mean())
    return float(toplam)


def temel(y: np.ndarray, olasilik: np.ndarray, oncel: np.ndarray | None = None) -> dict:
    """oncel: (N, K) eğitim katmanı sınıf oranları. Verilirse sınıf kararı argmax(p / oncel) olur — kalibre
    olasılıklarda dengeli hata için Bayes kararı (yalnız eğitim verisinden). AUC ve ECE p'den hesaplanır."""
    n = olasilik.shape[1]
    tahmin = (olasilik / oncel).argmax(1) if oncel is not None else olasilik.argmax(1)
    tek = np.eye(n)[y]
    cm = confusion_matrix(y, tahmin, labels=range(n))
    duyarlilik = np.diag(cm) / np.maximum(cm.sum(1), 1)
    ozgulluk = np.array([(cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]) / max(cm.sum() - cm[i].sum(), 1)
                         for i in range(n)])
    return {
        "roc_auc": float(roc_auc_score(tek, olasilik, average="macro", multi_class="ovr")),
        "pr_auc": float(average_precision_score(tek, olasilik, average="macro")),
        "macro_f1": float(f1_score(y, tahmin, average="macro")),
        "dengeli_dogruluk": float(balanced_accuracy_score(y, tahmin)),
        "dogruluk": float((tahmin == y).mean()),
        "ece": ece(y, olasilik),
        "duyarlilik_ort": float(duyarlilik.mean()),
        "ozgulluk_ort": float(ozgulluk.mean()),
        "sinif_duyarlilik": duyarlilik.round(4).tolist(),
        "sinif_ozgulluk": ozgulluk.round(4).tolist(),
        "karisiklik": cm.tolist(),
    }


def bootstrap(y: np.ndarray, olasilik: np.ndarray, grup: np.ndarray, n: int = 1000, tohum: int = 2026,
              oncel: np.ndarray | None = None) -> dict:
    """Hasta bazlı (küme) bootstrap ile %95 güven aralıkları."""
    rng = np.random.default_rng(tohum)
    hastalar = np.unique(grup)
    indis = {h: np.where(grup == h)[0] for h in hastalar}
    ornekler = {k: [] for k in ("roc_auc", "macro_f1", "dengeli_dogruluk", "ece")}
    for _ in range(n):
        sec = np.concatenate([indis[h] for h in rng.choice(hastalar, len(hastalar), replace=True)])
        if len(np.unique(y[sec])) < olasilik.shape[1]:
            continue
        m = temel(y[sec], olasilik[sec], None if oncel is None else oncel[sec])
        for k in ornekler:
            ornekler[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in ornekler.items() if v}


def ikili(y: np.ndarray, olasilik: np.ndarray, siniflar: list[str]) -> dict:
    """Çok sınıflı modelden ikili ayrımlar: yalnız iki sınıfın vizitleri, skor = p_b / (p_a + p_b).
    Ek eğitim yok; ör. CN–AD ayrımı aynı 4 sınıflı modelin olasılıklarından ölçülür."""
    sonuc = {}
    for a in range(len(siniflar)):
        for b in range(a + 1, len(siniflar)):
            m = (y == a) | (y == b)
            if len(np.unique(y[m])) < 2:
                continue
            skor = olasilik[m, b] / (olasilik[m, a] + olasilik[m, b] + 1e-12)
            hedef = (y[m] == b).astype(int)
            sonuc[f"{siniflar[a]}-{siniflar[b]}"] = {
                "n": int(m.sum()), "auc": float(roc_auc_score(hedef, skor)),
                "dengeli_dogruluk": float(balanced_accuracy_score(hedef, (skor >= 0.5).astype(int)))}
    return sonuc


def mcnemar(y: np.ndarray, tahmin_a: np.ndarray, tahmin_b: np.ndarray) -> dict:
    """İki modelin doğru/yanlış uyumsuzluğu (süreklilik düzeltmeli)."""
    a, b = tahmin_a == y, tahmin_b == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    istatistik = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return {"n01": n01, "n10": n10, "chi2": float(istatistik), "p": float(chi2.sf(istatistik, 1))}
