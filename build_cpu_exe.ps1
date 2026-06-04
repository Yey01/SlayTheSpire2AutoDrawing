$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $env:LOCALAPPDATA "autodrawer-cpu-build-venv"
$Build = Join-Path $env:LOCALAPPDATA "autodrawer_cpu_build"
$Dist = Join-Path $env:LOCALAPPDATA "autodrawer_cpu_dist"
$Wheelhouse = Join-Path $env:LOCALAPPDATA "autodrawer_cpu_wheelhouse"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock] $Command
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Download-Wheel {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Url,
        [Parameter(Mandatory = $true)]
        [string] $FileName,
        [Parameter(Mandatory = $true)]
        [int64] $ExpectedBytes
    )
    New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null
    $Target = Join-Path $Wheelhouse $FileName
    if ((Test-Path $Target) -and ((Get-Item -LiteralPath $Target).Length -eq $ExpectedBytes)) {
        Write-Host "Using cached wheel: $Target"
        return
    }
    for ($Attempt = 1; $Attempt -le 40; $Attempt++) {
        Write-Host "Downloading wheel: $FileName (attempt $Attempt)"
        & curl.exe -L -C - --retry 10 --retry-delay 5 --connect-timeout 60 --speed-time 180 --speed-limit 1024 -o $Target $Url
        $ActualBytes = 0
        if (Test-Path $Target) {
            $ActualBytes = (Get-Item -LiteralPath $Target).Length
        }
        if ($ActualBytes -eq $ExpectedBytes) {
            return
        }
        Write-Host "Partial wheel: $ActualBytes / $ExpectedBytes bytes"
        Start-Sleep -Seconds 5
    }
    $FinalBytes = 0
    if (Test-Path $Target) {
        $FinalBytes = (Get-Item -LiteralPath $Target).Length
    }
    throw "Downloaded wheel size mismatch for $FileName. Expected $ExpectedBytes, got $FinalBytes"
}

if (-not (Test-Path $Venv)) {
    Invoke-Checked { python -m venv $Venv }
}

$Python = Join-Path $Venv "Scripts\python.exe"
Invoke-Checked { & $Python -m pip install --upgrade pip --retries 10 --timeout 120 }

Download-Wheel `
    -Url "https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp313-cp313-win_amd64.whl" `
    -FileName "torch-2.7.1+cpu-cp313-cp313-win_amd64.whl" `
    -ExpectedBytes 215990934
Download-Wheel `
    -Url "https://download.pytorch.org/whl/cpu/torchvision-0.22.1%2Bcpu-cp313-cp313-win_amd64.whl" `
    -FileName "torchvision-0.22.1+cpu-cp313-cp313-win_amd64.whl" `
    -ExpectedBytes 1708165

Invoke-Checked { & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-cpu.txt") --find-links $Wheelhouse --retries 10 --timeout 120 --resume-retries 10 }

Invoke-Checked { & $Python -c "import torch; assert torch.version.cuda is None, f'Expected CPU-only torch, got CUDA {torch.version.cuda}'; print('CPU-only torch:', torch.__version__)" }

if (Test-Path $Build) {
    Remove-Item -LiteralPath $Build -Recurse -Force
}
if (Test-Path $Dist) {
    Remove-Item -LiteralPath $Dist -Recurse -Force
}

Push-Location $ProjectRoot
try {
    Invoke-Checked { & $Python -m PyInstaller AutoDrawer.spec --clean --noconfirm --workpath $Build --distpath $Dist }
}
finally {
    Pop-Location
}

Write-Host "CPU-only exe output: $Dist\AutoDrawer.exe"
