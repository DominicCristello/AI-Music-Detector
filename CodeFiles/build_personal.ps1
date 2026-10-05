param(
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$CodeFilesDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackagingDir = Join-Path $CodeFilesDir 'packaging'
$DistDir = Join-Path $CodeFilesDir 'dist'
$GuiDir = Join-Path $DistDir 'AI Music Detector'
$ServerExe = Join-Path $DistDir 'AI Music Detector Server.exe'
$BundledServerExe = Join-Path $GuiDir 'AI Music Detector Server.exe'
$PrivateEmailSource = Join-Path $CodeFilesDir 'Server\email_secrets.py'
$PrivateDataDir = Join-Path $env:LOCALAPPDATA 'AI Music Detector'
$PrivateEmailDestination = Join-Path $PrivateDataDir 'email_secrets.py'

if (-not $Python) {
    $LocalPython = Join-Path $env:LOCALAPPDATA `
        'Programs\Python\Python311\python.exe'
    if (Test-Path -LiteralPath $LocalPython -PathType Leaf) {
        $Python = $LocalPython
    }
    else {
        $PythonCommand = Get-Command python3.11, python, py `
            -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($PythonCommand) {
            $Python = $PythonCommand.Source
        }
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python 3.11 was not found. Pass its path with -Python.'
}

& $Python -c 'import PyInstaller' 2>$null
if ($LASTEXITCODE -ne 0) {
    # A Microsoft Store uninstall can leave the user's installed packages in
    # LocalCache. Reuse that compatible Python 3.11 package directory when it
    # exists, rather than forcing a large dependency reinstall.
    $PackageRoot = Join-Path $env:LOCALAPPDATA 'Packages'
    $LegacySite = Get-ChildItem -LiteralPath $PackageRoot -Directory `
        -Filter 'PythonSoftwareFoundation.Python.3.11_*' `
        -ErrorAction SilentlyContinue |
        ForEach-Object {
            Join-Path $_.FullName `
                'LocalCache\local-packages\Python311\site-packages'
        } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
        Select-Object -First 1

    if ($LegacySite) {
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = $LegacySite + [IO.Path]::PathSeparator + `
                              $env:PYTHONPATH
        }
        else {
            $env:PYTHONPATH = $LegacySite
        }
        & $Python -c 'import PyInstaller' 2>$null
    }

    if ($LASTEXITCODE -ne 0) {
        throw ('PyInstaller is not installed. Run: ' + $Python +
               ' -m pip install pyinstaller')
    }
}

Push-Location $CodeFilesDir
try {
    Write-Host 'Building AI Music Detector Server...'
    & $Python -m PyInstaller --noconfirm --clean `
        (Join-Path $PackagingDir 'AI_Music_Detector_Server.spec')
    if ($LASTEXITCODE -ne 0) {
        throw 'The server executable build failed.'
    }

    Write-Host 'Building AI Music Detector interface...'
    & $Python -m PyInstaller --noconfirm --clean `
        (Join-Path $PackagingDir 'AI_Music_Detector.spec')
    if ($LASTEXITCODE -ne 0) {
        throw 'The interface executable build failed.'
    }

    if (-not (Test-Path -LiteralPath $ServerExe -PathType Leaf)) {
        throw "The server executable was not created at: $ServerExe"
    }
    if (-not (Test-Path -LiteralPath $GuiDir -PathType Container)) {
        throw "The interface application folder was not created at: $GuiDir"
    }

    Copy-Item -LiteralPath $ServerExe -Destination $BundledServerExe -Force

    # Personal builds keep SMTP credentials outside both executables.  The
    # source file is gitignored, and its LocalAppData copy remains private to
    # this Windows account while still being discoverable by the frozen server.
    if (Test-Path -LiteralPath $PrivateEmailSource -PathType Leaf) {
        New-Item -ItemType Directory -Path $PrivateDataDir -Force | Out-Null
        Copy-Item -LiteralPath $PrivateEmailSource `
            -Destination $PrivateEmailDestination -Force
        Write-Host 'Private email settings installed in LocalAppData.'
    }

    Write-Host ''
    Write-Host 'Personal build complete.' -ForegroundColor Green
    Write-Host ('Launch: ' + (Join-Path $GuiDir 'AI Music Detector.exe'))
    Write-Host 'The server executable beside it starts and stops automatically.'
}
finally {
    Pop-Location
}
