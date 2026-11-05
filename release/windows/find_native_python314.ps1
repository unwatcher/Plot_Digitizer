param(
    [ValidateSet("Console", "Windowed")]
    [string]$Mode = "Console"
)

$ErrorActionPreference = "SilentlyContinue"

function Resolve-PathText([string]$PathText) {
    if (-not $PathText) { return $null }
    try { return (Resolve-Path -LiteralPath $PathText -ErrorAction Stop).Path } catch { return $null }
}

function Test-ConsolePython314([string]$PathText) {
    $resolved = Resolve-PathText $PathText
    if (-not $resolved -or -not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $false }
    & $resolved -c "import struct, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) and struct.calcsize('P') == 8 else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Add-Candidate([System.Collections.Generic.List[string]]$Candidates, [string]$PathText) {
    $resolved = Resolve-PathText $PathText
    if ($resolved -and -not $Candidates.Contains($resolved)) { $Candidates.Add($resolved) | Out-Null }
}

$candidates = [System.Collections.Generic.List[string]]::new()
Add-Candidate $candidates $env:PLOTDIGITIZER_PYTHON
Add-Candidate $candidates (Join-Path $PSScriptRoot "python314\python.exe")

$pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($pyLauncher) {
    $reported = @(& $pyLauncher.Source -3.14 -c "import sys; print(sys.executable)" 2>$null)
    if ($LASTEXITCODE -eq 0 -and $reported) { Add-Candidate $candidates ([string]$reported[0]) }
}

foreach ($command in Get-Command python.exe -All -ErrorAction SilentlyContinue) { Add-Candidate $candidates $command.Source }

$roots = @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)})
foreach ($root in $roots) {
    if (-not $root) { continue }
    Add-Candidate $candidates (Join-Path $root "Programs\Python\Python314\python.exe")
    Add-Candidate $candidates (Join-Path $root "Python314\python.exe")
    Add-Candidate $candidates (Join-Path $root "Python 3.14\python.exe")
}

foreach ($candidate in $candidates) {
    if (-not (Test-ConsolePython314 $candidate)) { continue }
    if ($Mode -eq "Console") { Write-Output $candidate; exit 0 }
    $pythonw = Join-Path (Split-Path -Parent $candidate) "pythonw.exe"
    if (Test-Path -LiteralPath $pythonw -PathType Leaf) { Write-Output (Resolve-PathText $pythonw); exit 0 }
}

[Console]::Error.WriteLine("ERROR: No native 64-bit CPython 3.14 installation was found. Install Python 3.14 x64 or set PLOTDIGITIZER_PYTHON to python.exe.")
exit 1
