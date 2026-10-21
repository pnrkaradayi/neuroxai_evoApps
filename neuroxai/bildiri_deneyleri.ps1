# Bildiri deneylerini sirayla calistirir (Claude uygulamasi kapansa da surer). Biten adim tekrar calistirilmaz
# (sonuclar\<ad>\ozet.json varsa atlanir). Kullanim:
#   powershell -ExecutionPolicy Bypass -File neuroxai\bildiri_deneyleri.ps1
# Not: Windows PowerShell 5.1 BOM'suz UTF-8 dosyalardaki Turkce karakterleri bozdugu icin bu dosya ASCII tutuldu.
$ErrorActionPreference = "Continue"
$proje = Split-Path -Parent $PSScriptRoot
Set-Location $proje
$env:PYTHONIOENCODING = "utf-8"
$gunluk = "C:\NeuroXAI_data\sonuclar\bildiri.log"
New-Item -ItemType Directory -Force (Split-Path $gunluk) | Out-Null
$py = Join-Path $proje ".venv\Scripts\python.exe"

function Adim($ad, $hedef, [string[]]$arg) {
    if ($hedef -and (Test-Path $hedef)) {
        "=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') $ad zaten bitmis, atlandi" | Tee-Object -FilePath $gunluk -Append
        return
    }
    "=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') $ad basliyor" | Tee-Object -FilePath $gunluk -Append
    & $py -W ignore -u -m @arg 2>&1 | Tee-Object -FilePath $gunluk -Append
    "=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') $ad bitti (cikis kodu $LASTEXITCODE)" | Tee-Object -FilePath $gunluk -Append
}

$S = "C:\NeuroXAI_data\sonuclar"
# Bildiri PET'i FDG: ADNI v1'de onceki vizitli (delta) amiloid orneklerinde AD yalniz 34 vizit; FDG'de
# CN 234 / EMCI 180 / LMCI 343 / AD 101 (858 vizit, 538 hasta). Amiloid + tau (tam A/T/N) ve OASIS dis
# dogrulamasi (OASIS'te yalniz amiloid PET var) SCI asamasinda.
# 1) Ana model: 4 sinif, T1 + FDG-PET, orijinal + delta + kenar kanallari (MRI-only / PET-only / fuzyon ayni kosuda)
Adim "ana" "$S\ana\ozet.json" @("neuroxai.deney", "--ad", "ana", "--pet", "fdg")
# 2) Aciklanabilirlik: Grad-CAM ROI yuzdeleri, kanal katkilari, TreeSHAP modalite payi
Adim "xai_ana" "$S\ana\xai\xai_ozet.json" @("neuroxai.xai", "--ad", "ana")
# 3-4) Kanal ablasyonlari (CNN'ler yeniden egitilir)
Adim "delta_yok" "$S\delta_yok\ozet.json" @("neuroxai.deney", "--ad", "delta_yok", "--pet", "fdg", "--kanallar", "orj,kenar")
Adim "kenar_yok" "$S\kenar_yok\ozet.json" @("neuroxai.deney", "--ad", "kenar_yok", "--pet", "fdg", "--kanallar", "orj,delta")
# Bolgesel (atlas) oznitelikler + embedding fuzyonu (CNN yeniden egitilmez)
if (-not (Test-Path "C:\NeuroXAI_data\islenmis\bolgesel\ADNI_bolgesel.csv")) { & $py -W ignore -u -m neuroxai.bolgesel cikar 2>&1 | Tee-Object -FilePath $gunluk -Append }
Adim "bolgesel" "$S\ana\bolgesel\ozet.json" @("neuroxai.bolgesel", "deney", "--ad", "ana")
# Evrimsel bolgesel biyobelirtec paneli secimi (233 bitlik genotip; NSGA-II / MOTPE / rastgele / filtre)
Adim "panel" "$S\ana\panel\ozet_panel.json" @("neuroxai.panel", "--ad", "ana", "--butce", "600", "--nufus", "30", "--tohumlar", "0,1")
# Sizinti gosterimi (YALNIZ KARSILASTIRMA): vizit bazli bolme + capraz uretimsiz yigma -> literaturdeki sisik sonuclar
Adim "sizinti_gosterim" "$S\sizinti_gosterim\ozet.json" @("neuroxai.deney", "--ad", "sizinti_gosterim", "--pet", "fdg", "--sizinti")
# Ek gorev: 3 sinif (CN / MCI / AD; etiket vizit tanisi)
Adim "uc_sinif" "$S\uc_sinif\ozet.json" @("neuroxai.deney", "--ad", "uc_sinif", "--pet", "fdg", "--gorev", "uc")
# 5) SCI hazirligi: T1 + amiloid (ADNI ici) ve OASIS-3 on-islemesinin kalan kismi (kaldigi yerden devam eder)
Adim "amiloid" "$S\amiloid\ozet.json" @("neuroxai.deney", "--ad", "amiloid", "--pet", "amiloid")
Adim "oasis_onisleme" "" @("neuroxai.onisleme", "--veri-seti", "OASIS3", "--paralel", "4")
# 6) SCI hazirligi: indirilen ADNI ek zip'lerini ac (acilmis zip'ler .acildi isaretiyle atlanir)
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') adni_ac basliyor" | Tee-Object -FilePath $gunluk -Append
& py -u veri_hazirlama\adni.py ac 2>&1 | Tee-Object -FilePath $gunluk -Append
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') tum adimlar bitti" | Tee-Object -FilePath $gunluk -Append
