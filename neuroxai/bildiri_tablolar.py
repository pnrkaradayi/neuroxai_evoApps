"""Bildiri tabloları, sayıları ve şekilleri — deney çıktılarından otomatik (elle rakam kopyalanmaz).

Girdi : <sonuc>/{ana,delta_yok,kenar_yok,uc_sinif}/ozet.json, <sonuc>/ana/xai/xai_ozet.json, yamalar, ADNI PTDEMOG
Çıktı : 1005 revize/bildiri/uretilen/
  sayilar.tex          \\SetR{anahtar}{değer} satırları (main.tex'te \\R{anahtar} ile kullanılır)
  tablo_kohort.tex, tablo_ana.tex, tablo_ablasyon.tex, tablo_xai.tex
  sekil_yamalar.pdf, sekil_karisiklik.pdf, sekil_gradcam.png
Henüz bitmemiş deneylerin hücreleri "--" yazılır; belge her zaman derlenir.

Kullanım: .venv\\Scripts\\python -m neuroxai.bildiri_tablolar
"""
import csv
import json
import re
import shutil
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ortak import AYARLAR, SONUC
from .veri import YamaVerisi

CIKTI = Path(__file__).resolve().parents[1] / "bildiri" / "uretilen"

# Geçiş bağlamı: ana metin = 3 sınıf (uc_sinif, önek yok); ek materyal = 4 sınıf (ana, önek "d4")
_ORTAM = {"deney": "uc_sinif", "on": "", "siniflar": ["CN", "MCI", "AD"]}


def D() -> str:
    return _ORTAM["deney"]


def SIN() -> list[str]:
    return _ORTAM["siniflar"]


def _d(ad: str) -> Path:
    return CIKTI / (f"{_ORTAM['on']}_{ad}" if _ORTAM["on"] else ad)


def _k(anahtar: str) -> str:
    return f"{_ORTAM['on']}-{anahtar}" if _ORTAM["on"] else anahtar

VARYANT_AD = {"fuzyon": "Fusion (MRI+FDG) XGBoost", "mri": "MRI-only XGBoost", "pet": "FDG-only XGBoost",
              "cnn_mri": "CNN, MRI", "cnn_pet": "CNN, FDG", "cnn_ortalama": "CNN average (MRI, FDG)"}


def oku(yol: Path):
    return json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else None


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def ga(m, k):
    if not m or k not in m.get("ga95", {}):
        return "--"
    a, b = m["ga95"][k]
    return f"{a:.3f}--{b:.3f}"


def sayilar_yaz(satirlar: dict):
    with open(_d("sayilar.tex"), "w", encoding="utf-8") as f:
        f.write("% neuroxai.bildiri_tablolar tarafından üretildi — elle düzenlemeyin\n")
        for k, v in satirlar.items():
            f.write(f"\\SetR{{{k}}}{{{v}}}\n")


# ----------------------------------------------------------------- kohort
def kohort(S: dict):
    ayar = json.loads((SONUC / D() / "ayarlar.json").read_text(encoding="utf-8"))
    veri = YamaVerisi("ADNI", ayar["gorev"], ayar["pet"], ayar["delta_gerekli"], ayar.get("etiket_tutarli", True))
    tum = YamaVerisi("ADNI", ayar["gorev"], ayar["pet"], ayar["delta_gerekli"], etiket_tutarli=False)
    meta_yol = Path(AYARLAR["veri_kok"]) / "metadata" / "adni"
    demog = {}
    with open(sorted(meta_yol.glob("PTDEMOG_*.csv"))[-1], encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            d = demog.setdefault(r["PTID"], {})
            if r["PTGENDER"] in ("1", "2"):
                d.setdefault("kadin", r["PTGENDER"] == "2")
            if r["PTDOB"] and "/" in r["PTDOB"]:
                ay, yil = r["PTDOB"].split("/")
                d.setdefault("dogum", date(int(yil), int(ay), 15))
            if r["PTEDUCAT"].isdigit() and 0 < int(r["PTEDUCAT"]) < 30:
                d.setdefault("egitim", int(r["PTEDUCAT"]))
    satir = []
    for s, ad in enumerate(SIN()):
        idx = np.where(veri.y == s)[0]
        hastalar = sorted(set(veri.hasta[idx]))
        yas = [(date.fromisoformat(veri.meta[i]["vizit_kodu"]) - demog[veri.hasta[i]]["dogum"]).days / 365.25
               for i in idx if "dogum" in demog.get(veri.hasta[i], {})]
        kadin = [demog[h]["kadin"] for h in hastalar if "kadin" in demog.get(h, {})]
        egitim = [demog[h]["egitim"] for h in hastalar if "egitim" in demog.get(h, {})]
        aralik = [int(veri.meta[i]["mri_delta_gun"]) / 30.44 for i in idx if veri.meta[i].get("mri_delta_gun")]
        satir.append((ad, len(hastalar), len(idx), int(veri.delta_var[idx].sum()),
                      f"{np.mean(yas):.1f} $\\pm$ {np.std(yas):.1f}", f"{100 * np.mean(kadin):.0f}",
                      f"{np.mean(egitim):.1f} $\\pm$ {np.std(egitim):.1f}",
                      f"{np.median(aralik):.0f}" if aralik else "--"))
        S[_k(f"n-{ad}-hasta")], S[_k(f"n-{ad}-vizit")] = len(hastalar), len(idx)
    S[_k("n-vizit")], S[_k("n-hasta")], S[_k("n-delta")] = len(veri), len(set(veri.hasta)), int(veri.delta_var.sum())
    S[_k("n-tutarsiz")] = len(tum) - len(veri)
    S[_k("n-tum-vizit")] = len(tum)
    with open(_d("tablo_kohort.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrccc r}\n\\toprule\n"
                "Class & Subj. & Visits & $\\Delta$ & Age (y) & F (\\%) & Educ. (y) & $\\Delta t$ (mo) \\\\\n\\midrule\n")
        for r in satir:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write(f"\\midrule\nTotal & {len(set(veri.hasta))} & {len(veri)} & {int(veri.delta_var.sum())} "
                "& & & & \\\\\n\\bottomrule\n\\end{tabular}\n")
    return veri


# ----------------------------------------------------------------- sonuç tabloları
BOLGESEL_AD = {"fuzyon_bolgesel": "Fusion (CNN emb.\\ + regional) XGBoost", "bolgesel": "Regional measures only"}
CIFT_AD = ["CN-EMCI", "CN-LMCI", "CN-AD", "EMCI-LMCI", "EMCI-AD", "LMCI-AD"]


def ana_tablo(S: dict, ana: dict | None, bol: dict | None = None):
    def satir(f, ad, m, kalin_mi=False, ci=False):
        kalin = (lambda s: f"\\textbf{{{s}}}") if kalin_mi else (lambda s: s)
        hucre = [kalin(f3(m and m["roc_auc"])), kalin(f3(m and m["macro_f1"])),
                 kalin(f3(m and m["dengeli_dogruluk"])), f3(m and m["ece"])]
        f.write(f"{ad} & " + " & ".join(hucre) + " \\\\\n")
        if ci:
            f.write(f"\\quad 95\\% CI & {ga(m, 'roc_auc')} & {ga(m, 'macro_f1')} & "
                    f"{ga(m, 'dengeli_dogruluk')} & {ga(m, 'ece')} \\\\\n")

    with open(_d("tablo_ana.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l ccc c}\n\\toprule\n"
                "Model & ROC-AUC & Macro-F1 & Bal. acc. & ECE \\\\\n\\midrule\n")
        for v, ad in BOLGESEL_AD.items():
            satir(f, ad, (bol or {}).get("modeller", {}).get(v), v == "fuzyon_bolgesel", v == "fuzyon_bolgesel")
        f.write("\\midrule\n")
        for v, ad in VARYANT_AD.items():
            satir(f, ad, (ana or {}).get("modeller", {}).get(v), False, v == "fuzyon")
            if v == "fuzyon":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if bol:
        for v in BOLGESEL_AD:
            m = bol["modeller"][v]
            kk = v.replace("_", "")
            for k, k2 in (("roc_auc", "auc"), ("macro_f1", "f1"), ("dengeli_dogruluk", "ba"), ("ece", "ece")):
                S[_k(f"bol-{kk}-{k2}")] = f3(m[k])
            for k, k2 in (("roc_auc", "auc"), ("macro_f1", "f1")):
                S[_k(f"bol-{kk}-{k2}-ci")] = ga(m, k).replace("--", "–")
            for s, ad in enumerate(SIN()):
                S[_k(f"bol-{kk}-duy-{ad}")] = f"{100 * m['sinif_duyarlilik'][s]:.1f}"
        mc = bol.get("mcnemar_bolgesel_vs_ana")
        if mc:
            S[_k("bol-mcnemar-p")] = f"{mc['p']:.3f}" if mc["p"] >= 0.001 else "<0.001"
        S[_k("bol-sutun")] = bol.get("bolgesel_sutun", "")
        ikili_ayri = oku(SONUC / D() / "ikili" / "ozet.json")
        if ikili_ayri:
            ikili_tablo_ayri(S, ikili_ayri)
        else:
            ikili_tablo_dolayli(S, bol)
        karistirici_tablosu(S, bol)
    _ana_sayilar(S, ana)


def ikili_tablo_ayri(S: dict, o: dict):
    """Her ikili görev için ayrı eğitilmiş modeller (neuroxai.ikili_deney): literatürle karşılaştırma tablosu."""
    with open(_d("tablo_ikili.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l rr c ccccc cc}\n\\toprule\n"
                " & & & & & & & & & \\multicolumn{2}{c}{AUC by field strength} \\\\\n"
                "\\cmidrule(lr){10-11}\n"
                "Task & Visits & Subj. & AUC [95\\% CI] & Acc. & Bal. acc. & Sens. & Spec. & & 1.5T & 3T \\\\\n\\midrule\n")
        for cift, d in o["ciftler"].items():
            ci = d["ga95"]["auc"]
            fs = d.get("alan_gucu", {})
            gorev = cift.replace("-", " vs.\\ ")
            f.write(f"{gorev} & {d['n']} & {d['n_hasta']} & \\textbf{{{d['auc']:.3f}}} "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] & {d['dogruluk']:.3f} & {d['dengeli_dogruluk']:.3f} & "
                    f"{d['duyarlilik']:.3f} & {d['ozgulluk']:.3f} & & {f3(fs.get('1.5T', {}).get('auc'))} & "
                    f"{f3(fs.get('3T', {}).get('auc'))} \\\\\n")
            for k, kk in (("auc", "auc"), ("dogruluk", "acc"), ("dengeli_dogruluk", "ba"), ("duyarlilik", "sens"),
                          ("ozgulluk", "spec")):
                S[_k(f"ikili-{cift}-{kk}")] = f"{d[k]:.3f}"
                S[_k(f"ikili-{cift}-{kk}-yuzde")] = f"{100 * d[k]:.1f}"
            S[_k(f"ikili-{cift}-auc-ci")] = f"{ci[0]:.3f}–{ci[1]:.3f}"
            for g_ in ("1.5T", "3T"):
                if g_ in fs:
                    S[_k(f"ikili-{cift}-auc-{g_.replace('.', '')}")] = f"{fs[g_]['auc']:.3f}"
        f.write("\\bottomrule\n\\end{tabular}\n")


def karistirici_tablosu(S: dict, bol: dict):
    """Birleşik ve alan gücü katmanlarında ikili AUC (bölgesel füzyon modeli) + CN içinde 3T 'tanıma' AUC'si."""
    k = (bol.get("karistirici") or {}).get("fuzyon_bolgesel")
    if not k:
        return
    havuz = bol["ikili"]["fuzyon_bolgesel"]
    with open(_d("tablo_karistirici_ikili.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l ccc}\n\\toprule\nPair & Pooled & 1.5T only & 3T only \\\\\n\\midrule\n")
        for cift, d in havuz.items():
            a_ = k.get("1.5T", {}).get(cift, {}).get("auc")
            b_ = k.get("3T", {}).get(cift, {}).get("auc")
            f.write(f"{cift.replace('-', '--')} & {d['auc']:.3f} & {f3(a_)} & {f3(b_)} \\\\\n")
            if b_ is not None:
                S[_k(f"kar-{cift}-3T")] = f"{b_:.3f}"
            if a_ is not None:
                S[_k(f"kar-{cift}-15T")] = f"{a_:.3f}"
        f.write("\\bottomrule\n\\end{tabular}\n")
    if "cn_icinde_pEMCI_3T_auc" in k:
        S[_k("kar-cn-3t-auc")] = f"{k['cn_icinde_pEMCI_3T_auc']:.2f}"


def ikili_tablo_dolayli(S: dict, bol: dict):
    if True:
        with open(_d("tablo_ikili.tex"), "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{l r cc cc}\n\\toprule\n"
                    " & & \\multicolumn{2}{c}{Fusion + regional} & \\multicolumn{2}{c}{Fusion (CNN emb.)} \\\\\n"
                    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
                    "Pair & $n$ & AUC & Bal. acc. & AUC & Bal. acc. \\\\\n\\midrule\n")
            for cift in CIFT_AD:
                a_ = bol["ikili"].get("fuzyon_bolgesel", {}).get(cift)
                b_ = bol["ikili"].get("fuzyon_ana", {}).get(cift)
                if not a_:
                    continue
                f.write(f"{cift.replace('-', '--')} & {a_['n']} & \\textbf{{{a_['auc']:.3f}}} & "
                        f"{a_['dengeli_dogruluk']:.3f} & {f3(b_ and b_['auc'])} & "
                        f"{f3(b_ and b_['dengeli_dogruluk'])} \\\\\n")
                S[_k(f"ikili-{cift}-auc")], S[_k(f"ikili-{cift}-ba")] = f"{a_['auc']:.3f}", f"{a_['dengeli_dogruluk']:.3f}"
            f.write("\\bottomrule\n\\end{tabular}\n")


def _ana_sayilar(S: dict, ana: dict | None):
    if not ana:
        return
    for v in VARYANT_AD:
        m = ana["modeller"].get(v)
        if m:
            for k, kk in (("roc_auc", "auc"), ("macro_f1", "f1"), ("dengeli_dogruluk", "ba"), ("ece", "ece"),
                          ("dogruluk", "acc"), ("pr_auc", "prauc")):
                S[_k(f"ana-{v}-{kk}")] = f3(m[k])
            for k, kk in (("roc_auc", "auc"), ("macro_f1", "f1")):
                if k in m.get("ga95", {}):
                    S[_k(f"ana-{v}-{kk}-ci")] = ga(m, k).replace("--", "–")
            for s, ad in enumerate(SIN()):
                S[_k(f"ana-{v}-duy-{ad}")] = f"{100 * m['sinif_duyarlilik'][s]:.1f}"
    for diger in ("mri", "pet", "cnn_ortalama"):
        mc = ana.get(f"mcnemar_fuzyon_vs_{diger}")
        if mc:
            S[_k(f"ana-mcnemar-{diger}-p")] = f"{mc['p']:.3f}" if mc["p"] >= 0.001 else "<0.001"


def ablasyon_tablo(S: dict, deneyler: dict):
    satirlar = (("ana", "Original + $\\Delta$ + edge (full)"), ("delta_yok", "w/o $\\Delta$ channel"),
                ("kenar_yok", "w/o edge channel"))
    with open(_d("tablo_ablasyon.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l ccc cc}\n\\toprule\n"
                " & \\multicolumn{3}{c}{All visits} & \\multicolumn{2}{c}{Visits with $\\Delta$} \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
                "Channels & AUC & F1 & BA & AUC & F1 \\\\\n\\midrule\n")
        for ad, etiket in satirlar:
            o = deneyler.get(ad) or {}
            m = o.get("modeller", {}).get("fuzyon")
            d = o.get("delta_var_altkume", {}).get("fuzyon")
            f.write(f"{etiket} & {f3(m and m['roc_auc'])} & {f3(m and m['macro_f1'])} & "
                    f"{f3(m and m['dengeli_dogruluk'])} & {f3(d and d['roc_auc'])} & {f3(d and d['macro_f1'])} \\\\\n")
            if m:
                S[_k(f"{ad.replace('_', '')}-fuzyon-auc")], S[_k(f"{ad.replace('_', '')}-fuzyon-f1")] = \
                    f3(m["roc_auc"]), f3(m["macro_f1"])
            if d:
                S[_k(f"{ad.replace('_', '')}-delta-auc")], S[_k(f"{ad.replace('_', '')}-delta-f1")] = \
                    f3(d["roc_auc"]), f3(d["macro_f1"])
        f.write("\\bottomrule\n\\end{tabular}\n")
    uc = deneyler.get("uc_sinif")
    if uc:
        for v in ("fuzyon", "mri", "pet", "cnn_ortalama"):
            m = uc["modeller"].get(v)
            if m:
                S[_k(f"uc-{v}-auc")], S[_k(f"uc-{v}-f1")], S[_k(f"uc-{v}-ba")] = \
                    f3(m["roc_auc"]), f3(m["macro_f1"]), f3(m["dengeli_dogruluk"])
        sayilar = [int(x) for x in re.findall(r"\d+", uc["veri"].split("|")[0])]  # "N vizit (D Δ'lı), H hasta"
        S[_k("uc-veri")] = f"{sayilar[0]} visits of {sayilar[-1]} participants"


def xai_tablo(S: dict, xai: dict | None):
    with open(_d("tablo_xai.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l rr}\n\\toprule\n & MRI & FDG \\\\\n\\midrule\n"
                "\\multicolumn{3}{l}{\\textit{Grad-CAM mass by AAL region (\\%, mean of classes)}} \\\\\n")
        if not xai:
            f.write("-- & -- & -- \\\\\n\\bottomrule\n\\end{tabular}\n")
            return
        bolge = {}
        for m in ("mri", "pet"):
            for sinif, d in xai["gradcam_bolge_yuzde"][m].items():
                for b, y in d.items():
                    bolge.setdefault(b, {}).setdefault(m, []).append(y)
        sira = sorted(bolge, key=lambda b: -np.mean(bolge[b].get("mri", [0]) + bolge[b].get("pet", [0])))
        ingilizce = {"Hipokampus": "Hippocampus", "Parahipokampal": "Parahippocampal", "Amigdala": "Amygdala",
                     "Fusiform": "Fusiform", "Temporal korteks": "Temporal cortex", "Talamus": "Thalamus",
                     "İnsula": "Insula", "Bazal gangliyon": "Basal ganglia", "Lingual": "Lingual",
                     "Singulat": "Cingulate", "Diğer": "Other"}
        for b in sira[:7]:
            mr, pt = np.mean(bolge[b].get("mri", [0])), np.mean(bolge[b].get("pet", [0]))
            f.write(f"{ingilizce.get(b, b)} & {mr:.1f} & {pt:.1f} \\\\\n")
            S[_k(f"xai-bolge-{ingilizce.get(b, b).replace(' ', '')}-mri")] = f"{mr:.1f}"
            S[_k(f"xai-bolge-{ingilizce.get(b, b).replace(' ', '')}-pet")] = f"{pt:.1f}"
        f.write("\\midrule\n\\multicolumn{3}{l}{\\textit{Channel contribution by occlusion (\\%)}} \\\\\n")
        for k, ad in (("orj", "Original"), ("delta", "$\\Delta$"), ("kenar", "Edge")):
            mr = xai["kanal_katki_yuzde"]["mri"].get(k)
            pt = xai["kanal_katki_yuzde"]["pet"].get(k)
            f.write(f"{ad} & {mr if mr is not None else '--'} & {pt if pt is not None else '--'} \\\\\n")
            S[_k(f"xai-kanal-{k}-mri")], S[_k(f"xai-kanal-{k}-pet")] = mr, pt
        f.write("\\midrule\n\\multicolumn{3}{l}{\\textit{TreeSHAP share in the fusion model (\\%)}} \\\\\n")
        for sinif, d in xai["shap_modalite_yuzde"].items():
            f.write(f"{sinif} & {d['MRI']:.1f} & {d['PET']:.1f} \\\\\n")
            S[_k(f"xai-shap-{sinif}-mri")], S[_k(f"xai-shap-{sinif}-pet")] = f"{d['MRI']:.1f}", f"{d['PET']:.1f}"
        f.write("\\bottomrule\n\\end{tabular}\n")


# ----------------------------------------------------------------- şekiller
def sekil_yamalar(veri: YamaVerisi):
    aday = [i for i, m in enumerate(veri.meta) if veri.y[i] == SIN().index("AD") and m.get("mri_delta_var") == "1"
            and m.get("fdg_delta_var") == "1"]
    if not aday:
        return
    i = aday[len(aday) // 2]
    fig, eksen = plt.subplots(2, 3, figsize=(6.6, 4.5))
    for r, (X, ad) in enumerate(((veri.mri, "T1 MRI"), (veri.pet, "FDG-PET"))):
        x = X[i, 1]
        eksen[r, 0].imshow(x[0], cmap="gray", vmin=-3, vmax=4)
        eksen[r, 1].imshow(x[1], cmap="RdBu_r", vmin=-3, vmax=3)
        eksen[r, 2].imshow(x[2], cmap="gray", vmin=0, vmax=1)
        for c, t in enumerate(("original (z)", "$\\Delta$ (σ units)", "edge")):
            eksen[r, c].set_title(f"{ad}: {t}", fontsize=8)
            eksen[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(_d("sekil_yamalar.pdf"), bbox_inches="tight")
    plt.close(fig)


def sekil_karisiklik(ozet: dict | None, varyant: str = "fuzyon"):
    if not ozet:
        return
    cm = np.array(ozet["modeller"][varyant]["karisiklik"], dtype=float)
    oran = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(oran, cmap="Blues", vmin=0, vmax=1)
    for a in range(len(cm)):
        for b in range(len(cm)):
            ax.text(b, a, f"{100 * oran[a, b]:.0f}%\n({int(cm[a, b])})", ha="center", va="center", fontsize=7,
                    color="white" if oran[a, b] > 0.5 else "black")
    ax.set_xticks(range(len(SIN())), SIN(), fontsize=8)
    ax.set_yticks(range(len(SIN())), SIN(), fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    plt.tight_layout()
    plt.savefig(_d("sekil_karisiklik.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- evrimsel deney (EvoApps)
ALG_AD = {"nsga2": "NSGA-II (MO, evolutionary)", "motpe": "MOTPE (MO, Bayesian)",
          "rastgele": "Random search (MO)", "tpe": "TPE (SO, macro-F1)"}
ALG_RENK = {"nsga2": "#c0392b", "motpe": "#2471a3", "rastgele": "#7f8c8d", "tpe": "#27ae60"}


def evrimsel(S: dict):
    from .evrimsel import baskin_olmayan, hipervolum
    dizin = SONUC / D() / "evrimsel"
    ozet = oku(dizin / "ozet_evrimsel.json")
    kosular = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(dizin.glob("kat*_*_t*.json"))] \
        if dizin.exists() else []
    with open(_d("tablo_evrimsel.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l c ccc c c}\n\\toprule\n"
                " & Inner & \\multicolumn{3}{c}{Outer test (mean $\\pm$ SD over seeds)} & & \\\\\n"
                "\\cmidrule(lr){3-5}\n"
                "Algorithm & HV & Macro-F1 & ECE & ROC-AUC & $k_\\mathrm{MRI}/k_\\mathrm{FDG}$ & Time (min) \\\\\n\\midrule\n")
        for alg, ad in ALG_AD.items():
            o = (ozet or {}).get("algoritmalar", {}).get(alg)
            if not o:
                f.write(f"{ad} & -- & -- & -- & -- & -- & -- \\\\\n")
                continue
            f.write(f"{ad} & {o['hv_ort']:.3f} $\\pm$ {o['hv_ss']:.3f} & "
                    f"{o['test_f1_ort']:.3f} $\\pm$ {o['test_f1_ss']:.3f} & "
                    f"{o['test_ece_ort']:.3f} $\\pm$ {o['test_ece_ss']:.3f} & "
                    f"{o['test_auc_ort']:.3f} $\\pm$ {o['test_auc_ss']:.3f} & "
                    f"{o['k_mri_ort']:.0f}/{o['k_pet_ort']:.0f} & {o['sure_dk_ort']:.1f} \\\\\n")
            for b_, v_ in o.get("blok_orani", {}).items():
                S[_k(f"evo-{alg}-blok-{b_.replace('_', '')}")] = f"{100 * v_:.0f}"
            for k_, kk in (("hv_ort", "hv"), ("test_f1_ort", "f1"), ("test_ece_ort", "ece"), ("test_auc_ort", "auc"),
                           ("test_ba_ort", "ba"), ("k_mri_ort", "kmri"), ("k_pet_ort", "kpet")):
                S[_k(f"evo-{alg}-{kk}")] = f"{o[k_]:.0f}" if kk.startswith("k") else f"{o[k_]:.3f}"
        f.write("\\bottomrule\n\\end{tabular}\n")
    for k_, v in ((ozet or {}).get("wilcoxon") or {}).items():
        S[_k(f"wil-{k_.replace('_', '-')}")] = f"{v['p']:.3f}" if v["p"] >= 0.001 else "<0.001"
    if ozet:
        S[_k("evo-butce")], S[_k("evo-nufus")] = ozet["ayarlar"]["butce"], ozet["ayarlar"]["nufus"]
        S[_k("evo-tohum-sayisi")] = len(ozet["ayarlar"]["tohumlar"].split(","))
    if not kosular:
        return
    ref = np.array([1.0, (ozet or {}).get("ayarlar", {}).get("hv_ref_ece", 0.3)])
    # Şekil: kat 1, tohum 0 — tüm değerlendirmeler ve baskın olmayan cepheler
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for r in kosular:
        if r["kat"] != 0 or r["tohum"] != 0:
            continue
        nok = np.array([[x["f1"], x["ece"]] for x in r["kayit"]])
        ax.scatter(nok[:, 0], nok[:, 1], s=6, alpha=0.25, color=ALG_RENK[r["algoritma"]])
        on = nok[baskin_olmayan(np.c_[1 - nok[:, 0], nok[:, 1]])]
        on = on[np.argsort(on[:, 0])]
        ax.step(on[:, 0], on[:, 1], where="post", color=ALG_RENK[r["algoritma"]], lw=1.4,
                label=ALG_AD[r["algoritma"]])
    ax.axhline(0.05, ls=":", color="k", lw=0.8)
    ax.set_xlabel("inner macro-F1 (maximise)", fontsize=8)
    ax.set_ylabel("inner ECE (minimise)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(_d("sekil_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)
    # Şekil: değerlendirme sayısına göre hipervolüm (kat × tohum ortalaması ± SS)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for alg in ALG_AD:
        egriler = []
        for r in kosular:
            if r["algoritma"] != alg:
                continue
            nok = np.array([[1 - x["f1"], x["ece"]] for x in r["kayit"]])
            egriler.append([hipervolum(nok[:i], ref) for i in range(1, len(nok) + 1)])
        if not egriler:
            continue
        uz = min(len(e) for e in egriler)
        e = np.array([x[:uz] for x in egriler])
        x = np.arange(1, uz + 1)
        ax.plot(x, e.mean(0), color=ALG_RENK[alg], lw=1.3, label=ALG_AD[alg])
        ax.fill_between(x, e.mean(0) - e.std(0), e.mean(0) + e.std(0), color=ALG_RENK[alg], alpha=0.12)
    ax.set_xlabel("evaluations", fontsize=8)
    ax.set_ylabel("hypervolume (inner CV)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(_d("sekil_hv.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- bölgesel füzyon modelinin TreeSHAP özeti
def bolgesel_shap(S: dict, veri: YamaVerisi):
    """Her katın 'gömme + bölgesel' XGBoost'u için eğitim vizitlerinde |SHAP|; bloklara ve bölgelere toplanır.
    Sonuç önbelleğe alınır (<sonuc>/ana/bolgesel/shap_ozet.json)."""
    dizin = SONUC / D() / "bolgesel"
    onbellek = dizin / "shap_ozet.json"
    if not onbellek.exists():
        if not (dizin / "ozet.json").exists() or \
                not all((dizin / f"kat{k}_xgb_fuzyon_bolgesel.json").exists() for k in range(5)):
            return  # bölgesel deney henüz bitmedi
        import shap
        import xgboost as xgb
        from .bolgesel import bolgesel_matris
        from .veri import bolmeler
        R, sutunlar, bloklar = bolgesel_matris(veri, "fdg")
        toplam = None
        for k, (egitim, _) in enumerate(bolmeler(veri, 5, 2026)):
            g = np.load(SONUC / D() / f"kat{k}" / "gommeler.npz")
            X = np.concatenate([g["egitim"], R[egitim]], 1)
            m = xgb.XGBClassifier()
            m.load_model(dizin / f"kat{k}_xgb_fuzyon_bolgesel.json")
            d = np.abs(np.array(shap.TreeExplainer(m).shap_values(X)))
            d = d.mean(axis=tuple(i for i in range(d.ndim) if d.shape[i] != X.shape[1]))  # özellik başına
            toplam = d if toplam is None else toplam + d
        emb = g["egitim"].shape[1]
        adlar = [f"CNN-MRI PC{i + 1}" for i in range(emb // 2)] + [f"CNN-FDG PC{i + 1}" for i in range(emb // 2)] + sutunlar
        blok = ["cnn_mri"] * (emb // 2) + ["cnn_fdg"] * (emb // 2) + bloklar
        onbellek.write_text(json.dumps({"ad": adlar, "blok": blok, "onem": (toplam / 5).tolist()}), encoding="utf-8")
    o = json.loads(onbellek.read_text(encoding="utf-8"))
    onem, blok, ad = np.array(o["onem"]), np.array(o["blok"]), o["ad"]
    pay = {b: 100 * onem[blok == b].sum() / onem.sum() for b in dict.fromkeys(blok)}
    blok_ad = {"cnn_mri": "CNN embedding, MRI", "cnn_fdg": "CNN embedding, FDG", "mri_roi": "Regional MRI",
               "fdg_roi": "Regional FDG", "delta": "Annualised change", "demog": "Demographics"}
    for b, v in pay.items():
        S[_k(f"shapblok-{b}")] = f"{v:.1f}"

    def guzel(s):
        s = s.replace("d_", "$\\Delta$ ").replace("mri_bos_", "MRI CSF ").replace("mri_z_", "MRI int.\\ ")
        s = s.replace("fdg_", "FDG ").replace("yas", "Age").replace("kadin", "Sex").replace("egitim", "Education")
        return s.replace("_", " ")

    sira = np.argsort(-onem)
    with open(_d("tablo_shap_bolge.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l r @{\\hspace{1.2em}} l r}\n\\toprule\n"
                "Feature block & Share (\\%) & Top features (mean $|\\phi|$ rank) & \\\\\n\\midrule\n")
        ust = [i for i in sira[:12]]
        bl = sorted(pay.items(), key=lambda x: -x[1])
        for r in range(max(len(bl), len(ust))):
            sol = f"{blok_ad[bl[r][0]]} & {bl[r][1]:.1f}" if r < len(bl) else " & "
            sag = f"{r + 1}. {guzel(ad[ust[r]])} & " if r < len(ust) else " & "
            f.write(f"{sol} & {sag} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for r, i in enumerate(sira[:5]):
        S[_k(f"shapust-{r + 1}")] = guzel(ad[i])


def gecis(S: dict, deney: str, on: str, deneyler: dict):
    """Bir deney (görev) için tüm tablo/şekil/sayıları üretir; on='' ana metin, 'd4' ek materyal (4 sınıf)."""
    ozet = deneyler.get(deney) or oku(SONUC / deney / "ozet.json")
    if not ozet:
        return
    _ORTAM.update(deney=deney, on=on, siniflar=ozet["siniflar"])
    bol = oku(SONUC / deney / "bolgesel" / "ozet.json")
    veri = kohort(S)
    ana_tablo(S, ozet, bol)
    if deney == "ana":
        ablasyon_tablo(S, deneyler)
    xai_tablo(S, oku(SONUC / deney / "xai" / "xai_ozet.json"))
    sekil_yamalar(veri)
    sekil_karisiklik(bol, "fuzyon_bolgesel") if bol else sekil_karisiklik(ozet)
    evrimsel(S)
    bolgesel_shap(S, veri)
    from . import ek_materyal
    ek_materyal.main(S, veri)
    gradcam = SONUC / deney / "xai" / "gradcam_sinif_ortalama.png"
    if gradcam.exists():
        shutil.copy(gradcam, _d("sekil_gradcam.png"))


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)
    S = {}
    deneyler = {ad: oku(SONUC / ad / "ozet.json") for ad in ("ana", "delta_yok", "kenar_yok", "uc_sinif")}
    gecis(S, "ana", "d4", deneyler)          # 4 sınıf -> ek materyal (d4_* dosyalar, d4-* sayılar)
    gecis(S, "uc_sinif", "", deneyler)       # 3 sınıf -> ana metin
    S["durum"] = ", ".join(f"{ad}: {'tamam' if o else 'bekleniyor'}" for ad, o in deneyler.items())
    sayilar_yaz(S)
    print(f"{len(S)} sayı -> {CIKTI}\n{S['durum']}")


if __name__ == "__main__":
    # -m ile çalışınca bu dosya __main__ olur; ek_materyal ise neuroxai.bildiri_tablolar'ı ayrıca yükler.
    # Geçiş bağlamının (_ORTAM) tek olması için paket modülünün main'i çağrılır.
    from neuroxai import bildiri_tablolar
    bildiri_tablolar.main()
