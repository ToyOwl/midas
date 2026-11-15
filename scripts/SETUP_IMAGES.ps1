param([string]$TorchSitePackages = "")
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
& "$projectRoot\.venv\Scripts\python.exe" -m pip install 'Pillow>=11,<13'
if ($LASTEXITCODE) { throw 'Image reader installation failed' }
& "$projectRoot\.venv\Scripts\python.exe" -m venv "$projectRoot\.venv-images"
if ($LASTEXITCODE) { throw 'Image environment creation failed' }
if ($TorchSitePackages -and (Test-Path -LiteralPath "$TorchSitePackages\torch")) {
    Set-Content -LiteralPath "$projectRoot\.venv-images\Lib\site-packages\portagent_torch.pth" -Value $TorchSitePackages -Encoding ascii
} else {
    & "$projectRoot\.venv-images\Scripts\python.exe" -m pip install -r "$projectRoot\requirements-images.txt"
    if ($LASTEXITCODE) { throw 'Image dependencies installation failed' }
}
& "$projectRoot\.venv-images\Scripts\python.exe" -m pip install 'transformers==4.57.6' 'safetensors>=0.6,<1'
if ($LASTEXITCODE) { throw 'Embedding dependencies installation failed' }
Write-Output 'Image environment ready. CLIP weights must already exist in the local Hugging Face cache; inference never downloads them.'
