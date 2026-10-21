# EvoApps bildirisinin ana gorevi 3 sinif (CN / MCI / AD): 4 sinifta EMCI/LMCI ayrimi alan gucu (ADNI fazi)
# ile karistigi icin. Bu betik uc_sinif deneyinin uzerine bolgesel fuzyon, harmonize karsilastirma, evrimsel
# yapilandirma (Problem A), panel secimi (Problem B) ve XAI'yi calistirir; sonra kalan kuyrugu (amiloid, OASIS,
# ADNI ek zip) bildiri_deneyleri.ps1 ile surdurur. Biten adimlar atlanir.
# Not: Windows PowerShell 5.1 BOM'suz UTF-8 dosyalardaki Turkce karakterleri bozdugu icin bu dosya ASCII tutuldu.
$ErrorActionPreference = "Continue"
$proje = Split-Path -Parent $PSScriptRoot
Set-Location $proje
$env:PYTHONIOENCODING = "utf-8"
$gunluk = "C:\NeuroXAI_data\sonuclar\bildiri.log"
$py = Join-Path $proje ".venv\Scripts\python.exe"
$S = "C:\NeuroXAI_data\sonuclar\uc_sinif"

function Adim($ad, $hedef, [string[]]$arg) {
    if ($hedef -and (Test-Path $hedef)) {
        "=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') $ad zaten bitmis, atlandi" | Tee-Object -FilePath $gunluk -Append
        return
    }
    "=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') $ad basliyor" | Tee-Object -FilePath $gunluk -Append
    & $py -W ignore -u -m @arg 2>&1 | Tee-Object -FilePath $gunluk -Append
    "=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') $ad bitti (cikis kodu $LASTEXITCODE)" | Tee-Object -FilePath $gunluk -Append
}

Adim "uc_bolgesel" "$S\bolgesel\ozet.json" @("neuroxai.bolgesel", "deney", "--ad", "uc_sinif")
Adim "uc_bolgesel_harmonize" "$S\bolgesel_harmonize\ozet.json" @("neuroxai.bolgesel", "deney", "--ad", "uc_sinif", "--harmonize")
Adim "uc_xai" "$S\xai\xai_ozet.json" @("neuroxai.xai", "--ad", "uc_sinif")
Adim "uc_panel" "$S\panel\ozet_panel.json" @("neuroxai.panel", "--ad", "uc_sinif", "--butce", "600", "--nufus", "30", "--tohumlar", "0,1")
Adim "uc_evrimsel" "$S\evrimsel\ozet_evrimsel.json" @("neuroxai.evrimsel", "--ad", "uc_sinif", "--butce", "100", "--nufus", "20", "--tohumlar", "0,1")
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') 3 sinif zinciri bitti; kalan kuyruk" | Tee-Object -FilePath $gunluk -Append
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bildiri_deneyleri.ps1")
