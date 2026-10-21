"""İÇ KULLANIM: bildiri metninin "hedef anotasyonlu" kopyası (evoapps_hedefli.tex).

Gerçek sonuçlar DEĞİŞTİRİLMEZ; her performans ölçüsünün yanına kırmızı "(hedef: X)" eklenir. Hedefler sabit,
tutarlı kurallarla üretilir (aynı ölçü türüne tüm modellerde aynı artış -> modellerin sıralaması korunur;
dengeli doğruluk = duyarlılık/özgüllük ortalaması korunur). Betimleyici sayılara (kohort, p değerleri, seçim
yüzdeleri, SHAP/Grad-CAM payları, 4 sınıf karıştırıcı analizi, sızıntı gösterimi) hedef konmaz.
Bu dosyanın ürettiği belge hiçbir yere gönderilmez; yalnızca SCI aşaması için hedef takibidir.

Kullanım: .venv\\Scripts\\python -m neuroxai.hedefler   (önce neuroxai.bildiri_tablolar)
"""
import json
import re
from pathlib import Path

from .ortak import SONUC

KOK = Path(__file__).resolve().parents[1] / "bildiri"
URET = KOK / "uretilen"

# 3 sınıf görevleri için ölçü bazında artış (literatürde hasta bazlı değerlendirmeli MRI+PET çalışmalarının üst
# aralığı hedeflenir); ECE için hedef: 0.005 daha düşük (en az 0.02)
D3 = {"auc": 0.075, "f1": 0.10, "ba": 0.08, "acc": 0.08, "prauc": 0.08}
# İkili görevler: AUC artışı ve eşik tabanlı ölçülerin (doğruluk, dengeli doğruluk, duyarlılık, özgüllük) artışı
DB = {"CN-AD": (0.036, 0.037), "MCI-AD": (0.042, 0.042), "CN-MCI": (0.122, 0.10)}


def _bicim(orj: str, deger: float) -> str:
    """Hedefi orijinal sayının ondalık basamak sayısıyla yazar."""
    m = re.search(r"\.(\d+)", orj)
    return f"{deger:.{len(m.group(1)) if m else 0}f}"


def _sinirla(x, ust=0.99):
    return min(ust, x)


def hedef_anahtar(k: str, v: str) -> str | None:
    """Bir \\R anahtarı için hedef değer (metin) ya da None."""
    if k.startswith(("d4-", "sizinti", "durust", "xai", "shap", "n-", "wil-", "pwil-", "kar-", "alan3")):
        return None
    try:
        # aralıklar: "a–b"
        if "–" in v:
            a, b = (float(x) for x in v.split("–"))
            d = _artis(k.rsplit("-ci", 1)[0])
            return None if d is None else f"{_bicim(v.split('–')[0], _sinirla(a + d))}–{_bicim(v.split('–')[1], _sinirla(b + d))}"
        x = float(v)
    except ValueError:
        return None
    if k == "panel-nsga2-boyut":
        return "15.0"
    if k.endswith("-ece"):
        return _bicim(v, max(0.02, x - 0.005))
    d = _artis(k)
    if d is None:
        return None
    yuzde = k.endswith("-yuzde") or re.search(r"-duy-", k) is not None
    return _bicim(v, _sinirla(x + (100 * d if yuzde else d), 99.0 if yuzde else 0.99))


def _artis(k: str) -> float | None:
    m = re.match(r"^ikili-(CN-AD|MCI-AD|CN-MCI)-(auc|acc|ba|sens|spec)(-yuzde|-15T|-3T)?$", k)
    if m:
        d_auc, d_esik = DB[m.group(1)]
        return d_auc if m.group(2) == "auc" else d_esik
    m = re.match(r"^(bol|ana)-[a-z]+-(auc|f1|ba|acc|prauc)$", k)
    if m:
        return D3[m.group(2)]
    if re.match(r"^(bol|ana)-[a-z]+-duy-(CN|MCI|AD)$", k):
        return D3["ba"]  # sınıf duyarlılıkları dengeli doğrulukla aynı artış -> ortalama tutarlı
    m = re.match(r"^(evo|panel)-[a-z0-9]+-(f1|auc|ba)$", k)
    if m:
        return D3[m.group(2)]
    return None


def _hucre(deger: float, metrik: str, ikili: str | None = None) -> str:
    if metrik == "ece":
        h = max(0.02, deger - 0.005)
    elif ikili:
        h = deger + (DB[ikili][0] if metrik == "auc" else DB[ikili][1])
    else:
        h = deger + D3[metrik]
    return f"{deger:.3f} {{\\color{{red}}({_sinirla(h):.3f})}}"


# ----------------------------------------------------------------- hedefli tablolar
def tablo_ana():
    ana = json.loads((SONUC / "uc_sinif" / "ozet.json").read_text(encoding="utf-8"))
    bol = json.loads((SONUC / "uc_sinif" / "bolgesel" / "ozet.json").read_text(encoding="utf-8"))
    satirlar = [("Fusion (CNN emb.\\ + regional) XGBoost", bol["modeller"]["fuzyon_bolgesel"]),
                ("Regional measures only", bol["modeller"]["bolgesel"]), None,
                ("Fusion (MRI+FDG) XGBoost", ana["modeller"]["fuzyon"]), None,
                ("MRI-only XGBoost", ana["modeller"]["mri"]), ("FDG-only XGBoost", ana["modeller"]["pet"]),
                ("CNN, MRI", ana["modeller"]["cnn_mri"]), ("CNN, FDG", ana["modeller"]["cnn_pet"]),
                ("CNN average (MRI, FDG)", ana["modeller"]["cnn_ortalama"])]
    with open(URET / "hedef_tablo_ana.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l cccc}\n\\toprule\nModel & ROC-AUC & Macro-F1 & Bal. acc. & ECE \\\\\n\\midrule\n")
        for s in satirlar:
            if s is None:
                f.write("\\midrule\n")
                continue
            ad, m = s
            f.write(f"{ad} & {_hucre(m['roc_auc'], 'auc')} & {_hucre(m['macro_f1'], 'f1')} & "
                    f"{_hucre(m['dengeli_dogruluk'], 'ba')} & {_hucre(m['ece'], 'ece')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def tablo_ikili():
    o = json.loads((SONUC / "uc_sinif" / "ikili" / "ozet.json").read_text(encoding="utf-8"))
    with open(URET / "hedef_tablo_ikili.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l rr ccccc cc}\n\\toprule\n"
                " & & & & & & & & \\multicolumn{2}{c}{AUC by field strength} \\\\\n\\cmidrule(lr){9-10}\n"
                "Task & Visits & Subj. & AUC & Acc. & Bal. acc. & Sens. & Spec. & 1.5T & 3T \\\\\n\\midrule\n")
        for cift, d in o["ciftler"].items():
            fs = d.get("alan_gucu", {})
            gorev = cift.replace("-", " vs.\\ ")
            hucre = [_hucre(d[k], m, cift) for k, m in (("auc", "auc"), ("dogruluk", "acc"), ("dengeli_dogruluk", "ba"),
                                                        ("duyarlilik", "acc"), ("ozgulluk", "acc"))]
            hucre += [_hucre(fs[g]["auc"], "auc", cift) if g in fs else "--" for g in ("1.5T", "3T")]
            f.write(f"{gorev} & {d['n']} & {d['n_hasta']} & " + " & ".join(hucre) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def tablo_evrimsel():
    o = json.loads((SONUC / "uc_sinif" / "evrimsel" / "ozet_evrimsel.json").read_text(encoding="utf-8"))
    ad = {"nsga2": "NSGA-II (MO, evolutionary)", "motpe": "MOTPE (MO, Bayesian)", "rastgele": "Random search (MO)",
          "tpe": "TPE (SO, macro-F1)"}
    with open(URET / "hedef_tablo_evrimsel.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l c ccc c}\n\\toprule\n"
                "Algorithm & Inner HV & Macro-F1 & ECE & ROC-AUC & $k_\\mathrm{MRI}/k_\\mathrm{FDG}$ \\\\\n\\midrule\n")
        for alg, a_ in ad.items():
            x = o["algoritmalar"][alg]
            f.write(f"{a_} & {x['hv_ort']:.3f} & {_hucre(x['test_f1_ort'], 'f1')} & {_hucre(x['test_ece_ort'], 'ece')} & "
                    f"{_hucre(x['test_auc_ort'], 'auc')} & {x['k_mri_ort']:.0f}/{x['k_pet_ort']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def tablo_panel():
    o = json.loads((SONUC / "uc_sinif" / "panel" / "ozet_panel.json").read_text(encoding="utf-8"))
    ad = {"nsga2": "NSGA-II", "motpe": "MOTPE", "rastgele": "Random", "filtre": "ANOVA filter"}
    with open(URET / "hedef_tablo_panel.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l c c ccc}\n\\toprule\n"
                "Method & Inner HV & Panel size & Macro-F1 & ROC-AUC & ECE \\\\\n\\midrule\n")
        for alg, a_ in ad.items():
            x = o["algoritmalar"][alg]
            boyut = f"{x['panel_ort']:.1f}" + (" {\\color{red}(15.0)}" if alg == "nsga2" else "")
            f.write(f"{a_} & {x['hv_ort']:.3f} & {boyut} & {_hucre(x['test_macro_f1_ort'], 'f1')} & "
                    f"{_hucre(x['test_roc_auc_ort'], 'auc')} & {_hucre(x['test_ece_ort'], 'ece')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ----------------------------------------------------------------- metin kopyası
UYARI = r"""\begin{center}
\fcolorbox{red}{white}{\parbox{0.95\textwidth}{\color{red}\small\textbf{İÇ KULLANIM --- HEDEF ANOTASYONLU KOPYA. GÖNDERİLMEZ.}
Siyah sayılar elde edilmiş gerçek sonuçlardır. Kırmızı parantez içindeki değerler SCI aşaması için \emph{hedeflerdir};
elde edilmiş sonuç değildir ve hiçbir yayında, raporda veya sunumda sonuç olarak kullanılamaz.}}
\end{center}
"""


HEDEF_DIZIN = KOK / "hedef"


def _cubuk(ax, etiketler, gercek, hedef, baslik, ylim=(0, 1), yuzde=False):
    """Gerçek (dolu, gri-mavi) ve hedef (kırmızı taralı) çubuklar yan yana."""
    import numpy as np
    x = np.arange(len(etiketler))
    w = 0.38
    ax.bar(x - w / 2, gercek, w, color="#34495e", label="actual (obtained)")
    ax.bar(x + w / 2, hedef, w, color="white", edgecolor="#c0392b", hatch="///", linewidth=1.2,
           label="TARGET (not obtained)")
    for xi, g, h in zip(x, gercek, hedef):
        f = (lambda v: f"{v:.1f}") if yuzde else (lambda v: f"{v:.3f}")
        ax.text(xi - w / 2, g, f(g), ha="center", va="bottom", fontsize=6)
        ax.text(xi + w / 2, h, f(h), ha="center", va="bottom", fontsize=6, color="#c0392b")
    ax.set_xticks(x, etiketler, fontsize=7)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=7)
    ax.set_title(baslik, fontsize=8)
    ax.text(0.99, 0.02, "TARGET", transform=ax.transAxes, ha="right", va="bottom", fontsize=14,
            color="#c0392b", alpha=0.25, fontweight="bold")


def grafikler():
    """Gerçek ve hedef karşılaştırma grafikleri (deney grafikleri yeniden çizilmez)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HEDEF_DIZIN.mkdir(exist_ok=True)
    ikili = json.loads((SONUC / "uc_sinif" / "ikili" / "ozet.json").read_text(encoding="utf-8"))["ciftler"]
    bol = json.loads((SONUC / "uc_sinif" / "bolgesel" / "ozet.json").read_text(encoding="utf-8"))["modeller"]
    ana = json.loads((SONUC / "uc_sinif" / "ozet.json").read_text(encoding="utf-8"))["modeller"]
    evo = json.loads((SONUC / "uc_sinif" / "evrimsel" / "ozet_evrimsel.json").read_text(encoding="utf-8"))["algoritmalar"]
    pan = json.loads((SONUC / "uc_sinif" / "panel" / "ozet_panel.json").read_text(encoding="utf-8"))["algoritmalar"]

    def h(v, metrik, cift=None):
        if metrik == "ece":
            return max(0.02, v - 0.005)
        return _sinirla(v + ((DB[cift][0] if metrik == "auc" else DB[cift][1]) if cift else D3[metrik]))

    # 1) İkili görevler
    fig, eks = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, (cift, d) in zip(eks, ikili.items()):
        ad = ["AUC", "Acc.", "Bal. acc.", "Sens.", "Spec."]
        g = [d["auc"], d["dogruluk"], d["dengeli_dogruluk"], d["duyarlilik"], d["ozgulluk"]]
        t = [h(g[0], "auc", cift)] + [h(v, "acc", cift) for v in g[1:]]
        _cubuk(ax, ad, g, t, cift.replace("-", " vs. "))
    eks[0].legend(fontsize=6, loc="lower left")
    plt.tight_layout()
    plt.savefig(HEDEF_DIZIN / "hedef_sekil_ikili.pdf", bbox_inches="tight")
    plt.close(fig)
    # 2) 3 sınıf modeller
    modeller = [("Fusion+reg.", bol["fuzyon_bolgesel"]), ("Regional", bol["bolgesel"]), ("Fusion", ana["fuzyon"]),
                ("MRI", ana["mri"]), ("FDG", ana["pet"]), ("CNN avg.", ana["cnn_ortalama"])]
    fig, eks = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (k, metrik, ad) in zip(eks, (("roc_auc", "auc", "macro ROC-AUC"), ("macro_f1", "f1", "macro-F1"),
                                         ("dengeli_dogruluk", "ba", "balanced accuracy"))):
        g = [m[k] for _, m in modeller]
        _cubuk(ax, [a for a, _ in modeller], g, [h(v, metrik) for v in g], f"Three-class: {ad}")
        ax.tick_params(axis="x", labelrotation=30)
    eks[0].legend(fontsize=6, loc="lower left")
    plt.tight_layout()
    plt.savefig(HEDEF_DIZIN / "hedef_sekil_uc_sinif.pdf", bbox_inches="tight")
    plt.close(fig)
    # 3) Sınıf duyarlılıkları (en iyi model) — ortalama = dengeli doğruluk
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    g = [100 * v for v in bol["fuzyon_bolgesel"]["sinif_duyarlilik"]]
    _cubuk(ax, ["CN", "MCI", "AD"], g, [min(99.0, v + 100 * D3["ba"]) for v in g],
           "Fusion + regional: sensitivity per class (%)", (0, 100), yuzde=True)
    ax.legend(fontsize=6, loc="upper left")
    plt.tight_layout()
    plt.savefig(HEDEF_DIZIN / "hedef_sekil_duyarlilik.pdf", bbox_inches="tight")
    plt.close(fig)
    # 4) Evrimsel: Problem A (F1, ECE) ve Problem B (panel büyüklüğü)
    fig, eks = plt.subplots(1, 3, figsize=(10, 2.8))
    algs = ["nsga2", "motpe", "rastgele", "tpe"]
    ad = ["NSGA-II", "MOTPE", "Random", "TPE (SO)"]
    g = [evo[a]["test_f1_ort"] for a in algs]
    _cubuk(eks[0], ad, g, [h(v, "f1") for v in g], "Problem A: outer-test macro-F1")
    g = [evo[a]["test_ece_ort"] for a in algs]
    _cubuk(eks[1], ad, g, [h(v, "ece") for v in g], "Problem A: outer-test ECE", (0, 0.1))
    palg = ["nsga2", "motpe", "rastgele", "filtre"]
    g = [pan[a]["panel_ort"] for a in palg]
    _cubuk(eks[2], ["NSGA-II", "MOTPE", "Random", "Filter"], g, [15.0] + g[1:],
           "Problem B: panel size (genes; target only for NSGA-II)", (0, 120), yuzde=True)
    eks[0].legend(fontsize=6, loc="lower left")
    plt.tight_layout()
    plt.savefig(HEDEF_DIZIN / "hedef_sekil_evrimsel.pdf", bbox_inches="tight")
    plt.close(fig)


EK_BOLUM = r"""
\clearpage
\appendix
\section*{EK (İÇ KULLANIM): Gerçek ve hedef karşılaştırma grafikleri}
\noindent{\color{red}\small Aşağıdaki grafiklerde koyu çubuklar elde edilmiş gerçek sonuçlardır; kırmızı taralı çubuklar
\textbf{hedeftir, elde edilmemiştir}. Deney grafikleri (Grad-CAM, karışıklık matrisi, Pareto cepheleri) yalnız gerçek
verilerle çizilir ve hedefe göre yeniden çizilmez.}

\begin{figure}[h]\centering\includegraphics[width=\textwidth]{hedef_sekil_ikili.pdf}
\par\smallskip{\color{red}\small Binary tasks: actual vs.\ TARGET.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=\textwidth]{hedef_sekil_uc_sinif.pdf}
\par\smallskip{\color{red}\small Three-class models: actual vs.\ TARGET.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=0.55\textwidth]{hedef_sekil_duyarlilik.pdf}
\par\smallskip{\color{red}\small Per-class sensitivity of the fusion model with regional measures: actual vs.\ TARGET (mean = balanced accuracy).}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=\textwidth]{hedef_sekil_evrimsel.pdf}
\par\smallskip{\color{red}\small Evolutionary problems: actual vs.\ TARGET.}\end{figure}
"""


def metin():
    s = (KOK / "evoapps.tex").read_text(encoding="utf-8")
    s = s.replace("\\InputIfFileExists{uretilen/sayilar.tex}{}{}",
                  "\\InputIfFileExists{uretilen/sayilar.tex}{}{}\n"
                  "\\newcommand{\\SetH}[2]{\\expandafter\\def\\csname H@#1\\endcsname{#2}}\n"
                  "\\InputIfFileExists{uretilen/hedefler.tex}{}{}\n"
                  "\\renewcommand{\\R}[1]{\\ifcsname R@#1\\endcsname\\csname R@#1\\endcsname\\else\\textcolor{red}{??}\\fi"
                  "\\ifcsname H@#1\\endcsname~{\\color{red}(hedef: \\csname H@#1\\endcsname)}\\fi}", 1)
    s = s.replace("\\maketitle", "\\maketitle\n" + UYARI, 1)
    for t in ("tablo_ana", "tablo_ikili", "tablo_evrimsel", "tablo_panel"):
        s = s.replace(f"uretilen/{t}.tex", f"uretilen/hedef_{t}.tex")
    s = "% OTOMATİK ÜRETİLDİ (neuroxai.hedefler) — İÇ KULLANIM, GÖNDERİLMEZ. Elle düzenlemeyin; evoapps.tex'i düzenleyin.\n" + s
    # hedef/ klasöründen derlenir: ortak dosyalara göreli yol
    s = s.replace("{uretilen/", "{../uretilen/").replace("\\bibliography{kaynaklar}", "\\bibliography{../kaynaklar}")
    s = s.replace("\\bibliographystyle{splncs04}", EK_BOLUM + "\n\\bibliographystyle{splncs04}", 1)
    HEDEF_DIZIN.mkdir(exist_ok=True)
    (HEDEF_DIZIN / "hedef.tex").write_text(s, encoding="utf-8")


def hedef_isaretli_sekiller():
    """Gerçek deney grafikleri + kırmızı hedef işaretleri (veri değiştirilmez, hedef verisi üretilmez)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .evrimsel import baskin_olmayan
    cikti = HEDEF_DIZIN / "sekiller" / "hedef_isaretli"
    cikti.mkdir(parents=True, exist_ok=True)
    renk = {"nsga2": "#c0392b", "motpe": "#2471a3", "rastgele": "#7f8c8d", "tpe": "#27ae60", "filtre": "#8e44ad"}
    ad = {"nsga2": "NSGA-II", "motpe": "MOTPE", "rastgele": "Random", "tpe": "TPE (SO)", "filtre": "ANOVA filter"}

    # 1) Karışıklık matrisi: gerçek oranlar; köşegende kırmızı sınıf hedefi (+8 puan, dengeli doğruluk hedefiyle tutarlı)
    bol = json.loads((SONUC / "uc_sinif" / "bolgesel" / "ozet.json").read_text(encoding="utf-8"))
    cm = np.array(bol["modeller"]["fuzyon_bolgesel"]["karisiklik"], dtype=float)
    siniflar = bol["siniflar"]
    oran = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.imshow(oran, cmap="Blues", vmin=0, vmax=1)
    for a in range(len(cm)):
        for b in range(len(cm)):
            renk_ = "white" if oran[a, b] > 0.5 else "black"
            ax.text(b, a - (0.12 if a == b else 0), f"{100 * oran[a, b]:.0f}%\n({int(cm[a, b])})", ha="center",
                    va="center", fontsize=7, color=renk_)
            if a == b:
                ax.text(b, a + 0.3, f"hedef {min(99, 100 * oran[a, b] + 100 * D3['ba']):.0f}%", ha="center",
                        va="center", fontsize=6.5, color="#c0392b", fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.0))
    ax.set_xticks(range(len(siniflar)), siniflar, fontsize=8)
    ax.set_yticks(range(len(siniflar)), siniflar, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title("actual (black) / TARGET (red, not obtained)", fontsize=7, color="#c0392b")
    plt.tight_layout()
    plt.savefig(cikti / "sekil_karisiklik_hedefli.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2) Problem A Pareto: gerçek cepheler + hedef çalışma noktası ve hedef bölgesi
    evo_dizin = SONUC / "uc_sinif" / "evrimsel"
    evo = json.loads((evo_dizin / "ozet_evrimsel.json").read_text(encoding="utf-8"))["algoritmalar"]
    h_f1, h_ece = _sinirla(evo["nsga2"]["test_f1_ort"] + D3["f1"]), max(0.02, evo["nsga2"]["test_ece_ort"] - 0.005)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for p in sorted(evo_dizin.glob("kat0_*_t0.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        nok = np.array([[x["f1"], x["ece"]] for x in r["kayit"]])
        ax.scatter(nok[:, 0], nok[:, 1], s=6, alpha=0.25, color=renk[r["algoritma"]])
        on = nok[baskin_olmayan(np.c_[1 - nok[:, 0], nok[:, 1]])]
        on = on[np.argsort(on[:, 0])]
        ax.step(on[:, 0], on[:, 1], where="post", color=renk[r["algoritma"]], lw=1.4, label=ad[r["algoritma"]])
    ax.axhline(0.05, ls=":", color="k", lw=0.8)
    x0, x1 = ax.get_xlim()
    ax.fill_between([h_f1, max(x1, h_f1 + 0.01)], 0, h_ece, color="#c0392b", alpha=0.12, hatch="//",
                    edgecolor="#c0392b", label="TARGET region (not obtained)")
    ax.scatter([h_f1], [h_ece], marker="*", s=140, color="#c0392b", zorder=5, label="TARGET point (outer test)")
    ax.set_xlim(x0, max(x1, h_f1 + 0.02))
    ax.set_xlabel("macro-F1 (inner CV; target refers to outer test)", fontsize=7.5)
    ax.set_ylabel("ECE", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left")
    plt.tight_layout()
    plt.savefig(cikti / "sekil_pareto_hedefli.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3) Problem B panel cepheleri: gerçek cepheler + hedef panel büyüklüğü (15 gen)
    pan_dizin = SONUC / "uc_sinif" / "panel"
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for p in sorted(pan_dizin.glob("kat0_*_t0.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = np.array(r["cephe"])
        ax.step(c[:, 1], c[:, 0], where="post", color=renk[r["algoritma"]], lw=1.4, label=ad[r["algoritma"]])
    ax.axvline(15, ls="--", color="#c0392b", lw=1.3, label="TARGET panel size (15)")
    ax.set_xscale("log")
    ax.set_xlabel("panel size (region–modality genes, log scale)", fontsize=8)
    ax.set_ylabel("inner macro-F1", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    plt.savefig(cikti / "sekil_panel_hedefli.pdf", bbox_inches="tight")
    plt.close(fig)


def duzlestir():
    """hedef.tex -> hedef_tam.tex: makrolar, tablo dosyaları ve kaynakça içine gömülü, tek başına derlenebilir.
    Şekiller hedef/sekiller/ altına kopyalanır. Sayılar ve kırmızı hedef anotasyonları hedef.tex ile aynıdır."""
    import shutil
    oku = lambda p: p.read_text(encoding="utf-8")

    def sozluk(dosya, komut):
        d = {}
        for satir in oku(dosya).splitlines():
            m = re.match(rf"\\{komut}\{{([^}}]+)\}}\{{(.*)\}}$", satir)
            if m:
                d[m.group(1)] = m.group(2)
        return d

    R, H = sozluk(URET / "sayilar.tex", "SetR"), sozluk(URET / "hedefler.tex", "SetH")
    s = oku(HEDEF_DIZIN / "hedef.tex")
    # 1) makro tanımları ve sayılar dosyası satırlarını kaldır
    s = re.sub(r"\\newcommand\{\\SetR\}.*\n", "", s)
    s = re.sub(r"\\newcommand\{\\R\}.*\n", "", s)
    s = re.sub(r"\\newcommand\{\\SetH\}.*\n", "", s)
    s = re.sub(r"\\renewcommand\{\\R\}.*\n", "", s)
    s = re.sub(r"\\InputIfFileExists\{\.\./uretilen/(sayilar|hedefler)\.tex\}\{\}\{\}\n", "", s)
    s = re.sub(r"% ---- Otomatik sonuçlar.*\n", "", s)

    # 2) \R{anahtar} -> değer (+ kırmızı hedef)
    def r_degis(m):
        k = m.group(1)
        v = R.get(k, "??")
        return v + (f"~{{\\color{{red}}(hedef: {H[k]})}}" if k in H else "")
    s = re.sub(r"\\R\{([^}]+)\}", r_degis, s)

    # 3) tablo dosyalarını göm
    def t_degis(m):
        yol = URET / m.group(1)
        return oku(yol).rstrip("\n") if yol.exists() else "\\textcolor{red}{(tablo yok)}"
    s = re.sub(r"\\InputIfFileExists\{\.\./uretilen/([^}]+)\}\{\}\{[^\n]*?\}(?=\n|$)", t_degis, s)

    # 4) şekilleri hedef/sekiller/ altına kopyala, \IfFileExists sarmalayıcısını kaldır
    sek = HEDEF_DIZIN / "sekiller"
    sek.mkdir(exist_ok=True)

    def s_degis(m):
        dosya, secenek = m.group(1), m.group(2)
        kaynak = URET / dosya
        if kaynak.exists():
            shutil.copy(kaynak, sek / dosya)
        return f"\\includegraphics[{secenek}]{{sekiller/{dosya}}}"
    s = re.sub(r"\\IfFileExists\{\.\./uretilen/([^}]+)\}\{\\includegraphics\[([^\]]*)\]\{\.\./uretilen/[^}]+\}\}\{[^\n]*?\}(?=\n|$)",
               s_degis, s)
    for ad in ("hedef_sekil_ikili.pdf", "hedef_sekil_uc_sinif.pdf", "hedef_sekil_duyarlilik.pdf",
               "hedef_sekil_evrimsel.pdf"):
        shutil.copy(HEDEF_DIZIN / ad, sek / ad)
        s = s.replace(f"{{{ad}}}", f"{{sekiller/{ad}}}")

    # 5) kaynakçayı göm (bibtex gerekmesin)
    bbl = HEDEF_DIZIN / "hedef.bbl"
    if bbl.exists():
        s = s.replace("\\bibliographystyle{splncs04}\n\\bibliography{../kaynaklar}", oku(bbl).rstrip("\n"))
    s = s.replace("% OTOMATİK ÜRETİLDİ (neuroxai.hedefler)", "% TEK DOSYA SÜRÜMÜ (neuroxai.hedefler.duzlestir)", 1)
    (HEDEF_DIZIN / "hedef_tam.tex").write_text(s, encoding="utf-8")
    kalan = re.findall(r"\\R\{|uretilen/|\\InputIfFileExists", s)
    print(f"hedef_tam.tex yazıldı; kalan bağımlılık: {len(kalan)}")


def main():
    sayilar = {}
    for satir in (URET / "sayilar.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\SetR\{([^}]+)\}\{(.*)\}$", satir)
        if m:
            sayilar[m.group(1)] = m.group(2)
    hedef = {k: h for k, v in sayilar.items() if (h := hedef_anahtar(k, v)) is not None}
    with open(URET / "hedefler.tex", "w", encoding="utf-8") as f:
        f.write("% İÇ KULLANIM: hedef değerler (elde edilmiş sonuç değildir) — neuroxai.hedefler\n")
        for k, v in hedef.items():
            f.write(f"\\SetH{{{k}}}{{{v}}}\n")
    tablo_ana()
    tablo_ikili()
    tablo_evrimsel()
    tablo_panel()
    grafikler()
    metin()
    print(f"{len(hedef)} hedef -> {URET / 'hedefler.tex'}; metin -> {HEDEF_DIZIN / 'hedef.tex'}")


if __name__ == "__main__":
    main()
