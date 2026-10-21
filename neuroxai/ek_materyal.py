"""Bildiri ek materyali ve panel deneyi için tablo/şekiller (bildiri_tablolar.main tarafından çağrılır).

Çıktı: 1005 revize/bildiri/uretilen/
  tablo_panel.tex, sekil_panel.pdf, sekil_panel_bolgeler.pdf      (evrimsel bölge paneli; ana metin)
  ek_tablo_kat.tex          kat bazında ölçüler (ana modeller)
  ek_tablo_nsga2.tex        NSGA-II'nin her kat × tohumda seçtiği yapılandırma
  ek_sekil_pareto_tum.pdf   tüm katlar × tohumlar için baskın olmayan cepheler
  ek_sekil_guvenilirlik.pdf güvenilirlik (kalibrasyon) grafikleri
  ek_sekil_qc.pdf           kayıt kalite kontrolü (NCC) dağılımları
  ek_tablo_sizinti.tex, ek_tablo_uc.tex  (deneyler bittiyse)
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrikler import temel
from .bildiri_tablolar import SIN, D, _d, _k
from .ortak import ISLENMIS, SONUC

CIKTI = Path(__file__).resolve().parents[1] / "bildiri" / "uretilen"
ALG_AD = {"nsga2": "NSGA-II", "motpe": "MOTPE", "rastgele": "Random", "tpe": "TPE (SO)", "filtre": "ANOVA filter"}
ALG_RENK = {"nsga2": "#c0392b", "motpe": "#2471a3", "rastgele": "#7f8c8d", "tpe": "#27ae60", "filtre": "#8e44ad"}


def _oku(yol):
    return json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else None


def f3(x):
    return "--" if x is None else f"{x:.3f}"


# ----------------------------------------------------------------- panel deneyi (ana metin)
def panel(S: dict):
    from .evrimsel import baskin_olmayan
    dizin = SONUC / D() / "panel"
    ozet = _oku(dizin / "ozet_panel.json")
    kosular = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(dizin.glob("kat*_*_t*.json"))] \
        if dizin.exists() else []
    with open(_d("tablo_panel.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l c c ccc}\n\\toprule\n"
                " & Inner & Panel size & \\multicolumn{3}{c}{Outer test (mean $\\pm$ SD over runs)} \\\\\n"
                "\\cmidrule(lr){4-6}\n"
                "Method & HV & (of 233) & Macro-F1 & ROC-AUC & ECE \\\\\n\\midrule\n")
        for alg in ("nsga2", "motpe", "rastgele", "filtre"):
            o = (ozet or {}).get("algoritmalar", {}).get(alg)
            if not o:
                f.write(f"{ALG_AD[alg]} & -- & -- & -- & -- & -- \\\\\n")
                continue
            f.write(f"{ALG_AD[alg]} & {o['hv_ort']:.3f} $\\pm$ {o['hv_ss']:.3f} & "
                    f"{o['panel_ort']:.1f} $\\pm$ {o['panel_ss']:.1f} & "
                    f"{o['test_macro_f1_ort']:.3f} $\\pm$ {o['test_macro_f1_ss']:.3f} & "
                    f"{o['test_roc_auc_ort']:.3f} $\\pm$ {o['test_roc_auc_ss']:.3f} & "
                    f"{o['test_ece_ort']:.3f} \\\\\n")
            S[_k(f"panel-{alg}-sure")] = f"{o['sure_dk_ort']:.0f}"
            for k, kk in (("hv_ort", "hv"), ("panel_ort", "boyut"), ("test_macro_f1_ort", "f1"),
                          ("test_roc_auc_ort", "auc"), ("test_ece_ort", "ece"), ("test_dengeli_dogruluk_ort", "ba")):
                S[_k(f"panel-{alg}-{kk}")] = f"{o[k]:.1f}" if kk == "boyut" else f"{o[k]:.3f}"
        f.write("\\bottomrule\n\\end{tabular}\n")
    for k, v in ((ozet or {}).get("wilcoxon") or {}).items():
        S[_k(f"pwil-{k.replace('_', '-')}")] = f"{v['p']:.3f}" if v["p"] >= 0.001 else "<0.001"
    if not kosular:
        return
    # Şekil: kat 1, tohum 0 — iç macro-F1 / panel büyüklüğü cepheleri
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for r in kosular:
        if r["kat"] != 0 or r["tohum"] != 0:
            continue
        c = np.array(r["cephe"])
        ax.step(c[:, 1], c[:, 0], where="post", color=ALG_RENK[r["algoritma"]], lw=1.4, label=ALG_AD[r["algoritma"]])
    ax.set_xscale("log")
    ax.set_xlabel("panel size (region–modality genes, log scale)", fontsize=8)
    ax.set_ylabel("inner macro-F1", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(_d("sekil_panel.pdf"), bbox_inches="tight")
    plt.close(fig)
    # Şekil: NSGA-II panellerinde en sık seçilen bölge-modalite genleri
    s = [r for r in kosular if r["algoritma"] == "nsga2"]
    if s:
        say = Counter(g for r in s for g in r["secilen_genler"])
        en = say.most_common(15)
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        adlar = [g.replace("_", " ") for g, _ in en][::-1]
        ax.barh(adlar, [100 * c / len(s) for _, c in en][::-1],
                color=["#c0392b" if g.startswith("FDG") else "#2c3e50" for g, _ in en][::-1])
        ax.set_xlabel("selected in % of NSGA-II runs", fontsize=8)
        ax.tick_params(labelsize=6.5)
        plt.tight_layout()
        plt.savefig(_d("sekil_panel_bolgeler.pdf"), bbox_inches="tight")
        plt.close(fig)
        for i, (g, c) in enumerate(en[:5]):
            S[_k(f"panelgen-{i + 1}")] = g.replace("_", " ")
            S[_k(f"panelgen-{i + 1}-oran")] = f"{100 * c / len(s):.0f}"


# ----------------------------------------------------------------- ek materyal
def kat_tablosu(veri):
    from .veri import bolmeler
    satirlar = []
    for k, (_, test) in enumerate(bolmeler(veri, 5, 2026)):
        t = SONUC / D() / f"kat{k}" / "tahminler.npz"
        if not t.exists():
            continue
        s = dict(np.load(t))
        hucre = [k + 1, len(test)]
        oncel = np.tile(np.bincount(veri.y[_], minlength=len(SIN())) / len(_), (len(test), 1))  # eğitim oranları
        for v in ("fuzyon", "mri", "pet", "cnn_ortalama"):
            hucre.append(temel(veri.y[test], s[v], oncel if v != "cnn_ortalama" else None)["macro_f1"])
        b = SONUC / D() / "bolgesel" / f"kat{k}_fuzyon_bolgesel.npz"
        hucre.append(temel(veri.y[test], np.load(b)["olasilik"], oncel)["macro_f1"] if b.exists() else None)
        satirlar.append(hucre)
    with open(_d("ek_tablo_kat.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{c r ccccc}\n\\toprule\n"
                "Fold & $n_\\mathrm{test}$ & Fusion & MRI-only & FDG-only & CNN avg. & Fusion + regional \\\\\n\\midrule\n")
        for h in satirlar:
            f.write(f"{h[0]} & {h[1]} & " + " & ".join(f3(x) for x in h[2:]) + " \\\\\n")
        if satirlar:
            ort = np.array([[x if x is not None else np.nan for x in h[2:]] for h in satirlar], dtype=float)
            f.write("\\midrule\nMean $\\pm$ SD & & " + " & ".join(
                f"{np.nanmean(c):.3f} $\\pm$ {np.nanstd(c):.3f}" for c in ort.T) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def nsga2_tablosu():
    kosular = sorted((json.loads(p.read_text(encoding="utf-8"))
                      for p in (SONUC / D() / "evrimsel").glob("kat*_nsga2_t*.json")),
                     key=lambda r: (r["kat"], r["tohum"]))
    with open(_d("ek_tablo_nsga2.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{cc rrrrrrr rr cccc cc}\n\\toprule\n"
                "F & S & trees & depth & lr & sub & col & mcw & $\\lambda$ & $k_\\mathrm{M}$ & $k_\\mathrm{F}$ & "
                "MR & FR & $\\Delta$ & Dem & F1$_\\mathrm{in}$ & ECE$_\\mathrm{in}$ \\\\\n\\midrule\n")
        for r in kosular:
            s = r["secilen"]
            p = s.get("params")
            if p is None:  # eski kayıtlarda parametreler yalnız 'kayit' dışında tutuldu
                p = {}
            b = s.get("bloklar", {})
            f.write(f"{r['kat'] + 1} & {r['tohum']} & {p.get('n_estimators', '--')} & {p.get('max_depth', '--')} & "
                    f"{p.get('learning_rate', float('nan')):.3f} & {p.get('subsample', float('nan')):.2f} & "
                    f"{p.get('colsample_bytree', float('nan')):.2f} & {p.get('min_child_weight', float('nan')):.1f} & "
                    f"{p.get('reg_lambda', float('nan')):.2f} & {s['k_mri']} & {s['k_pet']} & "
                    + " & ".join("\\checkmark" if b.get(x) else "" for x in ("mri_roi", "fdg_roi", "delta", "demog"))
                    + f" & {s['f1']:.3f} & {s['ece']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def pareto_tum():
    from .evrimsel import baskin_olmayan
    kosular = [json.loads(p.read_text(encoding="utf-8")) for p in (SONUC / D() / "evrimsel").glob("kat*_*_t*.json")]
    if not kosular:
        return
    katlar, tohumlar = sorted({r["kat"] for r in kosular}), sorted({r["tohum"] for r in kosular})
    fig, eks = plt.subplots(len(tohumlar), len(katlar), figsize=(2.3 * len(katlar), 2.0 * len(tohumlar)),
                            sharex=True, sharey=True, squeeze=False)
    for r in kosular:
        ax = eks[tohumlar.index(r["tohum"]), katlar.index(r["kat"])]
        nok = np.array([[x["f1"], x["ece"]] for x in r["kayit"]])
        on = nok[baskin_olmayan(np.c_[1 - nok[:, 0], nok[:, 1]])]
        on = on[np.argsort(on[:, 0])]
        ax.step(on[:, 0], on[:, 1], where="post", color=ALG_RENK[r["algoritma"]], lw=1.1, label=ALG_AD[r["algoritma"]])
        ax.axhline(0.05, ls=":", color="k", lw=0.6)
        ax.set_title(f"fold {r['kat'] + 1}, seed {r['tohum']}", fontsize=7)
        ax.tick_params(labelsize=6)
    eks[0, 0].legend(fontsize=5.5)
    fig.supxlabel("inner macro-F1", fontsize=8)
    fig.supylabel("inner ECE", fontsize=8)
    plt.tight_layout()
    plt.savefig(_d("ek_sekil_pareto_tum.pdf"), bbox_inches="tight")
    plt.close(fig)


def guvenilirlik(veri):
    """Güvenilirlik grafikleri: en olası sınıfın güveni (15 kutu) — gömme füzyonu kalibresiz/kalibreli,
    gömme+bölgesel (izotonik), NSGA-II seçimi (kalibrasyonsuz), tek amaçlı TPE seçimi."""
    adaylar = {}
    ana = [dict(np.load(p)) for p in sorted((SONUC / D()).glob("kat*/tahminler.npz"))]
    if ana:
        idx = np.concatenate([s["indis"] for s in ana])
        adaylar["Fusion, uncalibrated"] = (idx, np.concatenate([s["fuzyon_kalibresiz"] for s in ana]))
        adaylar["Fusion, isotonic"] = (idx, np.concatenate([s["fuzyon"] for s in ana]))
    b = sorted((SONUC / D() / "bolgesel").glob("kat*_fuzyon_bolgesel.npz"))
    if ana and len(b) == len(ana):
        adaylar["Fusion + regional, isotonic"] = (idx, np.concatenate([np.load(p)["olasilik"] for p in b]))
    for alg, ad in (("nsga2", "NSGA-II selection (no post-hoc)"), ("tpe", "TPE-SO selection (no post-hoc)")):
        r = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((SONUC / D() / "evrimsel").glob(f"kat*_{alg}_t0.json"))]
        if r:
            adaylar[ad] = (np.concatenate([x["test_indis"] for x in r]), np.concatenate([x["test_olasilik"] for x in r]))
    if not adaylar:
        return
    fig, eks = plt.subplots(1, len(adaylar), figsize=(2.4 * len(adaylar), 2.5), sharey=True, squeeze=False)
    for ax, (ad, (idx, ol)) in zip(eks[0], adaylar.items()):
        y = veri.y[idx]
        guven, dogru = ol.max(1), (ol.argmax(1) == y)
        kutu = np.linspace(0, 1, 16)
        orta, oran, say = [], [], []
        for a_, b_ in zip(kutu[:-1], kutu[1:]):
            m = (guven > a_) & (guven <= b_)
            if m.sum() >= 5:
                orta.append(guven[m].mean())
                oran.append(dogru[m].mean())
                say.append(m.sum())
        ax.plot([0, 1], [0, 1], ls=":", color="k", lw=0.8)
        ax.plot(orta, oran, "o-", ms=3, color="#c0392b")
        ax.set_title(f"{ad}\nECE {temel(y, ol)['ece']:.3f}", fontsize=6.5)
        ax.set_xlim(0.2, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("confidence", fontsize=7)
    eks[0, 0].set_ylabel("accuracy", fontsize=7)
    plt.tight_layout()
    plt.savefig(_d("ek_sekil_guvenilirlik.pdf"), bbox_inches="tight")
    plt.close(fig)


def qc_sekil(veri):
    t1, fdg = [], []
    for m in veri.meta:
        q = ISLENMIS / "ADNI" / m["hasta"] / m["vizit_kodu"] / "qc.json"
        if q.exists():
            d = json.loads(q.read_text(encoding="utf-8"))
            t1.append(d.get("t1_ncc", np.nan))
            fdg.append(d.get("fdg_ncc_t1", np.nan))
    fig, eks = plt.subplots(1, 2, figsize=(6.0, 2.2))
    eks[0].hist([x for x in t1 if x == x], bins=40, color="#2c3e50")
    eks[0].set_xlabel("T1-to-template NCC", fontsize=7)
    eks[1].hist([x for x in fdg if x == x], bins=40, color="#c0392b")
    eks[1].set_xlabel("FDG-to-T1 NCC", fontsize=7)
    for ax in eks:
        ax.tick_params(labelsize=6)
        ax.set_ylabel("visits", fontsize=7)
    plt.tight_layout()
    plt.savefig(_d("ek_sekil_qc.pdf"), bbox_inches="tight")
    plt.close(fig)


def ek_deneyler(S: dict):
    sz = _oku(SONUC / "sizinti_gosterim" / "ozet.json")
    ana = _oku(SONUC / "ana" / "ozet.json")  # sızıntı gösterimi 4 sınıfta yapıldı
    with open(_d("ek_tablo_sizinti.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l cccc}\n\\toprule\n"
                "Protocol & ROC-AUC & Macro-F1 & Bal. acc. & ECE \\\\\n\\midrule\n")
        for ad, o in (("Participant-level split + cross-fitting (ours)", ana),
                      ("Visit-level split + naive stacking (leaky)", sz)):
            m = (o or {}).get("modeller", {}).get("fuzyon")
            f.write(f"{ad} & {f3(m and m['roc_auc'])} & {f3(m and m['macro_f1'])} & "
                    f"{f3(m and m['dengeli_dogruluk'])} & {f3(m and m['ece'])} \\\\\n")
            if m and o is ana:
                S[_k("durust-ece")] = f3(m["ece"])
            if m and o is sz:
                S[_k("sizinti-ece")] = f3(m["ece"])
                S[_k("sizinti-auc")], S[_k("sizinti-f1")], S[_k("sizinti-ba")] = \
                    f3(m["roc_auc"]), f3(m["macro_f1"]), f3(m["dengeli_dogruluk"])
        f.write("\\bottomrule\n\\end{tabular}\n")


def etiket_tablosu():
    """Kayıt grubu × vizit tanısı (FDG'li, 4 sınıf adayı vizitler; tutarlılık filtresinden önce)."""
    from .veri import YamaVerisi
    v = YamaVerisi("ADNI", "dort", "fdg", delta_gerekli=False, etiket_tutarli=False)
    sinif, dx = ["CN", "EMCI", "LMCI", "AD"], ["CN", "MCI", "AD"]
    say = {(a, b): 0 for a in sinif for b in dx}
    for m in v.meta:
        if m["dx3"] in dx:
            say[(m["etiket4"], m["dx3"])] += 1
    tutarli = {("CN", "CN"), ("EMCI", "MCI"), ("LMCI", "MCI"), ("AD", "AD")}
    with open(_d("ek_tablo_etiket.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l rrr r}\n\\toprule\n"
                "Enrolment group & \\multicolumn{3}{c}{Diagnosis at visit} & Total \\\\\n\\cmidrule(lr){2-4}\n"
                " & CN & MCI & AD & \\\\\n\\midrule\n")
        for a in sinif:
            hucre = [f"\\textbf{{{say[(a, b)]}}}" if (a, b) in tutarli else str(say[(a, b)]) for b in dx]
            f.write(f"{a} & " + " & ".join(hucre) + f" & {sum(say[(a, b)] for b in dx)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def shap_uzun():
    o = _oku(SONUC / D() / "bolgesel" / "shap_ozet.json")
    if not o:
        return
    onem, ad = np.array(o["onem"]), o["ad"]
    sira = np.argsort(-onem)[:30]
    toplam = onem.sum()

    def guzel(s):
        s = s.replace("d_", "$\\Delta$ ").replace("mri_bos_", "MRI CSF ").replace("mri_z_", "MRI int.\\ ")
        s = s.replace("fdg_", "FDG ").replace("yas", "Age").replace("kadin", "Sex").replace("egitim", "Education")
        return s.replace("_", " ")

    with open(_d("ek_tablo_shap30.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{r l r @{\\hspace{2em}} r l r}\n\\toprule\n"
                "\\# & Feature & \\% & \\# & Feature & \\% \\\\\n\\midrule\n")
        for i in range(15):
            a_, b_ = sira[i], sira[i + 15]
            f.write(f"{i + 1} & {guzel(ad[a_])} & {100 * onem[a_] / toplam:.2f} & "
                    f"{i + 16} & {guzel(ad[b_])} & {100 * onem[b_] / toplam:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def panel_siklik():
    ozet = _oku(SONUC / D() / "panel" / "ozet_panel.json")
    with open(_d("ek_tablo_panel_siklik.tex"), "w", encoding="utf-8") as f:
        algs = [a for a in ("nsga2", "motpe", "rastgele", "filtre") if ozet and a in ozet.get("gen_sikligi", {})]
        if not algs:
            f.write("\\textit{(available after the panel experiment)}\n")
            return
        f.write("\\begin{tabular}{" + "l r " * len(algs) + "}\n\\toprule\n"
                + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ALG_AD[a]}}}" for a in algs) + " \\\\\n\\midrule\n")
        listeler = [list(ozet["gen_sikligi"][a].items())[:15] for a in algs]
        for i in range(15):
            hucre = []
            for li in listeler:
                hucre += [li[i][0].replace("_", " "), f"{100 * li[i][1]:.0f}"] if i < len(li) else ["", ""]
            f.write(" & ".join(hucre) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def uc_sinif_tablosu(S: dict):
    o = _oku(SONUC / "uc_sinif" / "ozet.json")
    with open(_d("ek_tablo_uc.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l cccc}\n\\toprule\nModel & ROC-AUC & Macro-F1 & Bal. acc. & ECE \\\\\n\\midrule\n")
        for v, ad in (("fuzyon", "Fusion (MRI+FDG) XGBoost"), ("mri", "MRI-only XGBoost"), ("pet", "FDG-only XGBoost"),
                      ("cnn_ortalama", "CNN average")):
            m = (o or {}).get("modeller", {}).get(v)
            f.write(f"{ad} & {f3(m and m['roc_auc'])} & {f3(m and m['macro_f1'])} & "
                    f"{f3(m and m['dengeli_dogruluk'])} & {f3(m and m['ece'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if o:
        S[_k("uc-veri-tam")] = o["veri"].split("|")[0].strip().replace("vizit", "visits").replace("hasta", "participants") \
            .replace("Δ'lı", "with $\\Delta$")


def karistirici_tablo(veri, S: dict):
    """Sınıflara göre alan gücü ve ADNI fazı (EMCI/LMCI ayrımında olası çekim karıştırıcısı)."""
    import csv
    from .ortak import AYARLAR, ENVANTER
    alan, faz = {}, {}
    with open(ENVANTER, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            alan[(r["hasta"], r["vizit_kodu"])] = r.get("alan_gucu", "")
    with open(AYARLAR["adni_vizitler"], encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            faz[(r["hasta"], r["mr_tarih"])] = r.get("faz", "")
    siniflar = SIN()
    alanlar, fazlar = ["1.5", "3"], sorted({faz.get((m["hasta"], m["vizit_kodu"]), "") for m in veri.meta} - {""})
    with open(_d("ek_tablo_karistirici.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l rr " + "r" * len(fazlar) + "}\n\\toprule\n"
                f" & \\multicolumn{{2}}{{c}}{{Field strength (T)}} & \\multicolumn{{{len(fazlar)}}}{{c}}{{ADNI phase of the MRI}} \\\\\n"
                f"\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-{3 + len(fazlar)}}}\n"
                "Class & 1.5 & 3 & " + " & ".join(fazlar) + " \\\\\n\\midrule\n")
        for s, ad in enumerate(siniflar):
            idx = np.where(veri.y == s)[0]
            a = Counter(str(alan.get((veri.meta[i]["hasta"], veri.meta[i]["vizit_kodu"]), "")).split(".")[0] if
                        str(alan.get((veri.meta[i]["hasta"], veri.meta[i]["vizit_kodu"]), "")).startswith("3") else "1.5"
                        for i in idx)
            p = Counter(faz.get((veri.meta[i]["hasta"], veri.meta[i]["vizit_kodu"]), "") for i in idx)
            f.write(f"{ad} & {a.get('1.5', 0)} & {a.get('3', 0)} & " + " & ".join(str(p.get(x, 0)) for x in fazlar)
                    + " \\\\\n")
            S[_k(f"alan3-{ad}")] = f"{100 * a.get('3', 0) / max(len(idx), 1):.0f}"
        f.write("\\bottomrule\n\\end{tabular}\n")


def sinif_tablosu():
    """Sınıf bazında duyarlılık / özgüllük (ana modeller)."""
    satir = []
    ana = _oku(SONUC / D() / "ozet.json")
    bol = _oku(SONUC / D() / "bolgesel" / "ozet.json")
    for ad, o, v in (("Fusion (CNN emb.)", ana, "fuzyon"), ("Fusion + regional", bol, "fuzyon_bolgesel"),
                     ("Regional only", bol, "bolgesel"), ("CNN average", ana, "cnn_ortalama")):
        m = (o or {}).get("modeller", {}).get(v)
        if m:
            satir.append((ad, m["sinif_duyarlilik"], m["sinif_ozgulluk"]))
    with open(_d("ek_tablo_sinif.tex"), "w", encoding="utf-8") as f:
        n_ = len(SIN())
        f.write("\\begin{tabular}{l " + "c" * n_ + " " + "c" * n_ + "}\n\\toprule\n"
                f" & \\multicolumn{{{n_}}}{{c}}{{Sensitivity}} & \\multicolumn{{{n_}}}{{c}}{{Specificity}} \\\\\n"
                f"\\cmidrule(lr){{2-{1 + n_}}}\\cmidrule(lr){{{2 + n_}-{1 + 2 * n_}}}\n"
                "Model & " + " & ".join(SIN() + SIN()) + " \\\\\n\\midrule\n")
        for ad, d, o in satir:
            f.write(f"{ad} & " + " & ".join(f"{x:.2f}" for x in d) + " & " + " & ".join(f"{x:.2f}" for x in o) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def karisiklik_uclu():
    ana = _oku(SONUC / D() / "ozet.json")
    bol = _oku(SONUC / D() / "bolgesel" / "ozet.json")
    modeller = [(ad, o["modeller"][v]["karisiklik"]) for ad, o, v in
                (("Fusion (CNN emb.)", ana, "fuzyon"), ("Regional only", bol, "bolgesel"),
                 ("Fusion + regional", bol, "fuzyon_bolgesel")) if o and v in o.get("modeller", {})]
    if not modeller:
        return
    fig, eks = plt.subplots(1, len(modeller), figsize=(3.0 * len(modeller), 2.8), squeeze=False)
    sin = SIN()
    for ax, (ad, cm) in zip(eks[0], modeller):
        cm = np.array(cm, dtype=float)
        oran = cm / cm.sum(1, keepdims=True)
        ax.imshow(oran, cmap="Blues", vmin=0, vmax=1)
        for a in range(len(sin)):
            for b in range(len(sin)):
                ax.text(b, a, f"{100 * oran[a, b]:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if oran[a, b] > 0.5 else "black")
        ax.set_xticks(range(len(sin)), sin, fontsize=7)
        ax.set_yticks(range(len(sin)), sin, fontsize=7)
        ax.set_title(ad, fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
    eks[0, 0].set_ylabel("true", fontsize=7)
    plt.tight_layout()
    plt.savefig(_d("ek_sekil_karisiklik3.pdf"), bbox_inches="tight")
    plt.close(fig)


def harmonize_tablosu():
    """Alan gücü harmonizasyonu (CN referanslı, katman içi) ile ve olmadan bölgesel modeller."""
    a = _oku(SONUC / D() / "bolgesel" / "ozet.json")
    b = _oku(SONUC / D() / "bolgesel_harmonize" / "ozet.json")
    if not (a and b):
        return
    with open(_d("ek_tablo_harmonize.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l cccc}\n\\toprule\nModel & Features & ROC-AUC & Macro-F1 & Bal. acc. & ECE \\\\\n"
                "\\midrule\n")
        for v, ad in (("fuzyon_bolgesel", "Fusion + regional"), ("bolgesel", "Regional only")):
            for o, h in ((a, "original"), (b, "harmonised")):
                m = o["modeller"][v]
                f.write(f"{ad} & {h} & {m['roc_auc']:.3f} & {m['macro_f1']:.3f} & {m['dengeli_dogruluk']:.3f} & "
                        f"{m['ece']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main(S: dict, veri):
    harmonize_tablosu()
    karistirici_tablo(veri, S)
    sinif_tablosu()
    karisiklik_uclu()
    etiket_tablosu()
    shap_uzun()
    panel_siklik()
    uc_sinif_tablosu(S)
    panel(S)
    kat_tablosu(veri)
    nsga2_tablosu()
    pareto_tum()
    guvenilirlik(veri)
    qc_sekil(veri)
    ek_deneyler(S)
