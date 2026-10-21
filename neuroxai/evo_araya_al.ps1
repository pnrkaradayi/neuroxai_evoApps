# EvoApps 2027 icin evrimsel deneyi kuyrugun onune alir: kenar_yok bitince calisan bildiri_deneyleri.ps1'i
# durdurur, neuroxai.evrimsel'i calistirir, sonra bildiri_deneyleri.ps1'i yeniden baslatir (biten adimlar atlanir).
# Not: Windows PowerShell 5.1 BOM'suz UTF-8 dosyalardaki Turkce karakterleri bozdugu icin bu dosya ASCII tutuldu.
param([int]$KosucuPid)
$ErrorActionPreference = "Continue"
$proje = Split-Path -Parent $PSScriptRoot
Set-Location $proje
$env:PYTHONIOENCODING = "utf-8"
$gunluk = "C:\NeuroXAI_data\sonuclar\bildiri.log"
$py = Join-Path $proje ".venv\Scripts\python.exe"

while (-not (Test-Path "C:\NeuroXAI_data\sonuclar\kenar_yok\ozet.json")) { Start-Sleep 60 }
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') kenar_yok bitti; kosucu durduruluyor, evrimsel deney araya aliniyor" | Tee-Object -FilePath $gunluk -Append
if ($KosucuPid) { taskkill /T /F /PID $KosucuPid 2>&1 | Out-Null }
Start-Sleep 5
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') evrimsel basliyor" | Tee-Object -FilePath $gunluk -Append
& $py -W ignore -u -m neuroxai.evrimsel --ad ana --butce 100 --nufus 20 --tohumlar 0,1,2 2>&1 | Tee-Object -FilePath $gunluk -Append
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') evrimsel bitti (cikis kodu $LASTEXITCODE)" | Tee-Object -FilePath $gunluk -Append
& $py -W ignore -m neuroxai.bildiri_tablolar 2>&1 | Tee-Object -FilePath $gunluk -Append
# Kalan kuyruk: uc_sinif, amiloid, OASIS on-isleme, ADNI ek zip'leri
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bildiri_deneyleri.ps1")
