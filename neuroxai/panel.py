"""Evrimsel çok amaçlı bölgesel biyobelirteç paneli seçimi (EvoApps 2027 bildirisi, ikinci evrimsel deney).

Genotip: 233 bit — 116 AAL bölgesi × {MRI, FDG} + demografi. Bir bit açıksa o bölge-modalitenin tüm ölçüleri
(MRI: ortalama z, BOS oranı ve yıllık değişimleri; FDG: SUVR ve yıllık değişimi) modele girer.
Amaçlar (dış eğitim katmanı içinde, 3 katlı hasta bazlı iç CV, sabit XGBoost):
  1) macro-F1 (maks.)   2) panel büyüklüğü = açık bit sayısı (min.)
Algoritmalar (aynı bütçe, aynı başlangıç nüfusu):
  nsga2     : ikili NSGA-II — hızlı baskın olmayan sıralama, kalabalık mesafesi, ikili turnuva,
              tek düze çaprazlama, bit çevirme mutasyonu (1/L)
  motpe     : çok amaçlı TPE (Optuna, Bayes)
  rastgele  : rastgele yoğunluklu rastgele maskeler
  filtre    : tek değişkenli ANOVA-F sıralaması, ilk-k (k = 1..L), öğrenmesiz klasik referans
Başlangıç nüfusu: her birey için yoğunluk ~ U(0.02, 0.5), sonra bitler; tüm algoritmalar aynı başlangıcı paylaşır.
Seçim (dış test için): iç macro-F1'i en iyinin 0.01 altına kadar olan baskın olmayan çözümlerden EN KÜÇÜK panel.
Dış test: seçilen panelle XGBoost + çapraz üretilmiş izotonik kalibrasyon (deney.xgb_kalibreli).

Kullanım: .venv\\Scripts\\python -m neuroxai.panel [--butce 1000 --nufus 40 --tohumlar 0,1]
"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import optuna
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from .bolgesel import bolgesel_matris
from .deney import xgb_kalibreli, xgb_model
from .evrimsel import baskin_olmayan, hipervolum
from .metrikler import temel
from .ortak import SONUC, uyanik_tut
from .veri import YamaVerisi, bolmeler, capraz_uretim

SABIT_XGB = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.8,
             "colsample_bytree": 0.5, "min_child_weight": 3, "reg_lambda": 1.0, "max_bin": 64}


def genler(sutunlar: list[str]):
    """Sütunları bölge-modalite genlerine gruplar: gen adı -> sütun indisleri."""
    grup = {}
    for i, s in enumerate(sutunlar):
        ad = s[2:] if s.startswith("d_") else s
        if ad.startswith("mri_z_") or ad.startswith("mri_bos_"):
            g = "MRI:" + ad.split("_", 2)[2]
        elif ad.startswith("fdg_"):
            g = "FDG:" + ad[4:]
        else:
            g = "Demographics"
        grup.setdefault(g, []).append(i)
    adlar = sorted(grup)
    return adlar, [grup[g] for g in adlar]


class Degerlendirici:
    def __init__(self, X, y, grup, gen_sutun, n_sinif, tohum):
        self.X, self.y, self.gen_sutun, self.n, self.tohum = X, y, gen_sutun, n_sinif, tohum
        self.ic = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=tohum).split(X, y, grup))
        self.agirlik = [compute_sample_weight("balanced", y[tr]) for tr, _ in self.ic]
        self.onbellek, self.sayac = {}, 0

    def __call__(self, maske: np.ndarray) -> tuple[float, int]:
        anahtar = maske.tobytes()
        if anahtar not in self.onbellek:
            self.sayac += 1
            boyut = int(maske.sum())
            if boyut == 0:
                self.onbellek[anahtar] = (0.0, 0)
            else:
                sutun = [c for g in np.where(maske)[0] for c in self.gen_sutun[g]]
                oof = np.zeros((len(self.y), self.n))
                for i, (tr, va) in enumerate(self.ic):
                    m = xgb_model(SABIT_XGB, self.n, self.tohum)
                    m.fit(self.X[tr][:, sutun], self.y[tr], sample_weight=self.agirlik[i])
                    oof[va] = m.predict_proba(self.X[va][:, sutun])
                self.onbellek[anahtar] = (temel(self.y, oof)["macro_f1"], boyut)
        return self.onbellek[anahtar]


def baslangic(L: int, nufus: int, rng) -> np.ndarray:
    yog = rng.uniform(0.02, 0.5, nufus)
    return (rng.random((nufus, L)) < yog[:, None])


# ----------------------------------------------------------------- ikili NSGA-II
def _siralama(F: np.ndarray):
    """F: (n, 2) en aza indirilecek. Cephe numaraları ve kalabalık mesafeleri."""
    n = len(F)
    rank = np.zeros(n, dtype=int)
    kalan = np.arange(n)
    r = 0
    while len(kalan):
        on = kalan[baskin_olmayan(F[kalan])]
        rank[on] = r
        kalan = np.setdiff1d(kalan, on)
        r += 1
    kalabalik = np.zeros(n)
    for r_ in np.unique(rank):
        idx = np.where(rank == r_)[0]
        for j in range(F.shape[1]):
            s = idx[np.argsort(F[idx, j])]
            kalabalik[s[0]] = kalabalik[s[-1]] = np.inf
            aralik = F[s[-1], j] - F[s[0], j] or 1.0
            kalabalik[s[1:-1]] += (F[s[2:], j] - F[s[:-2], j]) / aralik
    return rank, kalabalik


def nsga2(deg, L, butce, nufus, rng, P0):
    P = P0.copy()
    F = np.array([[1 - f, b / L] for f, b in (deg(p) for p in P)])
    gecmis = [deg(p) for p in P]
    nesil = 0
    while deg.sayac < butce and nesil < 20 * butce // nufus:  # tekrar eden çocuklar sayılmaz; sonsuz döngü koruması
        nesil += 1
        rank, kal = _siralama(F)

        def turnuva():
            a, b = rng.integers(0, len(P), 2)
            return a if (rank[a], -kal[a]) < (rank[b], -kal[b]) else b

        cocuk = []
        while len(cocuk) < nufus and deg.sayac < butce:  # bütçe dolunca kuşak yarıda kesilir (eşit bütçe)
            p1, p2 = P[turnuva()], P[turnuva()]
            maske = rng.random(L) < 0.5 if rng.random() < 0.9 else np.zeros(L, bool)
            c = np.where(maske, p2, p1)
            c = c ^ (rng.random(L) < 1.0 / L)
            gecmis.append(deg(c))
            cocuk.append(c)
        if not cocuk:
            break
        Q = np.array(cocuk)
        FQ = np.array([[1 - f, b / L] for f, b in (deg(q) for q in Q)])
        R, FR = np.vstack([P, Q]), np.vstack([F, FQ])
        rank, kal = _siralama(FR)
        sira = np.lexsort((-kal, rank))[:nufus]
        P, F = R[sira], FR[sira]
    return gecmis


def rastgele(deg, L, butce, nufus, rng, P0):
    gecmis = [deg(p) for p in P0]
    for _ in range(20 * butce):
        if deg.sayac >= butce:
            break
        gecmis.append(deg(baslangic(L, 1, rng)[0]))
    return gecmis


def motpe(deg, L, butce, nufus, rng, P0):
    gecmis = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    calisma = optuna.create_study(directions=["maximize", "minimize"],
                                  sampler=optuna.samplers.TPESampler(seed=int(rng.integers(1 << 30)),
                                                                     n_startup_trials=nufus))
    for p in P0:
        calisma.enqueue_trial({f"g{j}": int(p[j]) for j in range(L)})

    def amac(t):
        m = np.array([t.suggest_categorical(f"g{j}", [0, 1]) for j in range(L)], dtype=bool)
        f, b = deg(m)
        gecmis.append((f, b))
        return f, b

    for _ in range(3 * butce):  # tekrar eden öneriler bütçeyi artırmaz; sonsuz döngü koruması
        if deg.sayac >= butce:
            break
        calisma.optimize(amac, n_trials=1)
    return gecmis


def filtre(deg, L, X, y, gen_sutun):
    F, _ = f_classif(np.nan_to_num(X), y)
    skor = np.array([np.nanmax(F[s]) for s in gen_sutun])
    sira = np.argsort(-skor)
    gecmis = []
    for k in list(range(1, 21)) + list(range(22, 61, 2)) + list(range(65, L + 1, 5)):
        m = np.zeros(L, bool)
        m[sira[:k]] = True
        gecmis.append(deg(m))
    return gecmis


ALG = {"nsga2": nsga2, "motpe": motpe, "rastgele": rastgele}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ad", default="ana")
    p.add_argument("--butce", type=int, default=1000)
    p.add_argument("--nufus", type=int, default=40)
    p.add_argument("--tohumlar", default="0,1")
    p.add_argument("--algoritmalar", default="nsga2,motpe,rastgele,filtre")
    p.add_argument("--tolerans", type=float, default=0.01)
    p.add_argument("--sadece-kat", type=int, default=None)
    a = p.parse_args()
    uyanik_tut()
    kok = SONUC / a.ad
    ayar = json.loads((kok / "ayarlar.json").read_text(encoding="utf-8"))
    cikti = kok / "panel"
    cikti.mkdir(exist_ok=True)
    veri = YamaVerisi("ADNI", ayar["gorev"], ayar["pet"], ayar["delta_gerekli"], ayar.get("etiket_tutarli", True))
    R, sutunlar, _ = bolgesel_matris(veri, ayar["pet"])
    gen_adlari, gen_sutun = genler(sutunlar)
    L, n = len(gen_adlari), len(veri.siniflar)
    print(f"=== panel ({a.ad}): {veri.ozet()} | {L} gen, bütçe {a.butce}", flush=True)
    for k, (egitim, test) in enumerate(bolmeler(veri, ayar["kat"], ayar["tohum"])):
        if a.sadece_kat is not None and k != a.sadece_kat:
            continue
        X, y = R[egitim], veri.y[egitim]
        parcalar = capraz_uretim(veri, egitim, ayar.get("ic_kat", 3), ayar["tohum"])
        konum = {v: i for i, v in enumerate(egitim.tolist())}
        cv = [(np.array([konum[v] for v in tr]), np.array([konum[v] for v in pr])) for tr, pr in parcalar]
        for s in [int(t) for t in a.tohumlar.split(",")]:
            P0 = baslangic(L, a.nufus, np.random.default_rng(1000 * k + s))
            for alg in a.algoritmalar.split(","):
                if alg == "filtre" and s != int(a.tohumlar.split(",")[0]):
                    continue  # belirlenimci
                hedef = cikti / f"kat{k}_{alg}_t{s}.json"
                if hedef.exists():
                    continue
                deg = Degerlendirici(X, y, veri.hasta[egitim], gen_sutun, n, ayar["tohum"])
                t = time.time()
                rng = np.random.default_rng(10_000 + 1000 * k + s)
                gecmis = filtre(deg, L, X, y, gen_sutun) if alg == "filtre" else ALG[alg](deg, L, a.butce, a.nufus, rng, P0)
                maskeler = {m: v for m, v in deg.onbellek.items()}
                nok = np.array([[1 - f, b / L] for f, b in maskeler.values()])
                on = baskin_olmayan(nok)
                adaylar = [(m, v) for (m, v), o in zip(maskeler.items(), on) if o]
                en_iyi = max(v[0] for _, v in adaylar)
                m_sec, v_sec = min(((m, v) for m, v in adaylar if v[0] >= en_iyi - a.tolerans), key=lambda x: x[1][1])
                secili = np.frombuffer(m_sec, dtype=bool)
                sutun = [c for g in np.where(secili)[0] for c in gen_sutun[g]]
                model, kalibre_et = xgb_kalibreli(R[egitim][:, sutun], y, cv, SABIT_XGB, n, ayar["tohum"])
                ol = kalibre_et(model.predict_proba(R[test][:, sutun]))
                olcu = temel(veri.y[test], ol)
                sonuc = {"kat": k, "algoritma": alg, "tohum": s, "sure_sn": round(time.time() - t, 1),
                         "degerlendirme": deg.sayac, "hv": hipervolum(nok, np.array([1.0, 1.0])),
                         "gecmis": gecmis, "secilen_genler": [gen_adlari[g] for g in np.where(secili)[0]],
                         "ic_f1": v_sec[0], "panel": int(v_sec[1]), "test_olasilik": ol.tolist(),
                         "test_indis": test.tolist(), "test": {k_: olcu[k_] for k_ in
                                                               ("roc_auc", "macro_f1", "dengeli_dogruluk", "ece")},
                         "cephe": sorted([list(v) for _, v in adaylar], key=lambda x: x[1])}
                hedef.write_text(json.dumps(sonuc), encoding="utf-8")
                print(f"kat {k + 1} {alg:8s} t{s}: HV {sonuc['hv']:.4f} | panel {sonuc['panel']} gen, iç F1 "
                      f"{v_sec[0]:.3f} | test F1 {olcu['macro_f1']:.3f} AUC {olcu['roc_auc']:.3f} ECE {olcu['ece']:.3f}"
                      f" ({sonuc['sure_sn'] / 60:.1f} dk, {deg.sayac} değ.)", flush=True)
    ozetle(cikti, veri)


def ozetle(cikti, veri):
    from scipy.stats import wilcoxon
    from .veri import egitim_oncelleri
    kosular = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cikti.glob("kat*_*_t*.json"))]
    for r in kosular:  # kalibre olasılıklarda dengeli karar: argmax(p / eğitim katmanı sınıf oranı)
        idx, ol = np.array(r["test_indis"]), np.array(r["test_olasilik"])
        o_ = temel(veri.y[idx], ol, egitim_oncelleri(veri, idx))
        r["test"] = {k: o_[k] for k in ("roc_auc", "macro_f1", "dengeli_dogruluk", "ece")}
    ozet = {"algoritmalar": {}, "wilcoxon": {}, "gen_sikligi": {}}
    for alg in sorted({r["algoritma"] for r in kosular}):
        s = [r for r in kosular if r["algoritma"] == alg]
        ozet["algoritmalar"][alg] = {
            "n": len(s), "hv_ort": float(np.mean([r["hv"] for r in s])), "hv_ss": float(np.std([r["hv"] for r in s])),
            "panel_ort": float(np.mean([r["panel"] for r in s])), "panel_ss": float(np.std([r["panel"] for r in s])),
            **{f"test_{k}_ort": float(np.mean([r["test"][k] for r in s])) for k in ("roc_auc", "macro_f1", "dengeli_dogruluk", "ece")},
            **{f"test_{k}_ss": float(np.std([r["test"][k] for r in s])) for k in ("roc_auc", "macro_f1", "dengeli_dogruluk", "ece")},
            "sure_dk_ort": float(np.mean([r["sure_sn"] for r in s]) / 60)}
        sayac = Counter(g for r in s for g in r["secilen_genler"])
        ozet["gen_sikligi"][alg] = {g: c / len(s) for g, c in sayac.most_common(30)}
    for alg in ("motpe", "rastgele"):
        for olcut in ("hv", "panel"):
            a_ = {(r["kat"], r["tohum"]): r[olcut] for r in kosular if r["algoritma"] == "nsga2"}
            b_ = {(r["kat"], r["tohum"]): r[olcut] for r in kosular if r["algoritma"] == alg}
            ortak = sorted(set(a_) & set(b_))
            fark = [a_[i] - b_[i] for i in ortak]
            if len(ortak) >= 5 and any(fark):
                ozet["wilcoxon"][f"nsga2_vs_{alg}_{olcut}"] = {"n": len(ortak), "ort_fark": float(np.mean(fark)),
                                                               "p": float(wilcoxon(fark).pvalue)}
    (cikti / "ozet_panel.json").write_text(json.dumps(ozet, indent=1, ensure_ascii=False), encoding="utf-8")
    for alg, o in ozet["algoritmalar"].items():
        print(f"{alg:8s} HV {o['hv_ort']:.4f}±{o['hv_ss']:.4f}  panel {o['panel_ort']:.1f}  test F1 "
              f"{o['test_macro_f1_ort']:.3f}  AUC {o['test_roc_auc_ort']:.3f}  ECE {o['test_ece_ort']:.3f}")
    for k_, v in ozet["wilcoxon"].items():
        print(f"  {k_}: fark {v['ort_fark']:+.4f} p={v['p']:.4f}")
    print("->", cikti / "ozet_panel.json")


if __name__ == "__main__":
    main()
