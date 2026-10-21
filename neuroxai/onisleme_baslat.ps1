# On-islemeyi ayri bir pencerede calistirir (Claude uygulamasi kapansa da surer; kesilirse kaldigi yerden devam eder).
# Kullanim: sag tik > PowerShell ile calistir, ya da:  powershell -ExecutionPolicy Bypass -File neuroxai\onisleme_baslat.ps1
# Not: Windows PowerShell 5.1 BOM'suz UTF-8 dosyalardaki Turkce karakterleri bozdugu icin bu dosya ASCII tutuldu.
$ErrorActionPreference = "Continue"
$proje = Split-Path -Parent $PSScriptRoot
Set-Location $proje
$env:PYTHONIOENCODING = "utf-8"
$gunluk = "C:\NeuroXAI_data\islenmis\onisleme.log"
$py = Join-Path $proje ".venv\Scripts\python.exe"
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') ADNI basliyor" | Tee-Object -FilePath $gunluk -Append
& $py -u -m neuroxai.onisleme --veri-seti ADNI --paralel 6 2>&1 | Tee-Object -FilePath $gunluk -Append
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') OASIS3 basliyor (4D PET bellek kullandigi icin 4 paralel)" | Tee-Object -FilePath $gunluk -Append
& $py -u -m neuroxai.onisleme --veri-seti OASIS3 --paralel 4 2>&1 | Tee-Object -FilePath $gunluk -Append
"=== $(Get-Date -Format 'dd.MM.yyyy HH:mm') bitti" | Tee-Object -FilePath $gunluk -Append
