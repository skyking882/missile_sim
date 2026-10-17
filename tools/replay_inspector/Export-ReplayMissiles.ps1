param(
    [Parameter(Mandatory = $true)][string]$GameDirectory,
    [Parameter(Mandatory = $true)][string]$ReplayPath,
    [string]$OutputDirectory,
    [string]$Weapon,
    [string]$Entity,
    [string]$LocalPackage,
    [string]$Distribution = 'Ubuntu',
    [string]$ModuleDirectory
)
$ErrorActionPreference = 'Stop'
$resolvedGame = (Resolve-Path -LiteralPath $GameDirectory).Path
$resolvedReplay = (Resolve-Path -LiteralPath $ReplayPath).Path
if (-not $ModuleDirectory) {
    $ModuleDirectory = (& wsl.exe -d $Distribution -- python3 -c 'from pathlib import Path; print(Path.home() / ".cache/wrpl-replay-parser-221be5c/build/python/main")' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $ModuleDirectory) { throw 'Cannot locate the default WSL backend directory' }
}
if (-not $OutputDirectory) {
    $runName = [IO.Path]::GetFileNameWithoutExtension($resolvedReplay) -replace '[^a-zA-Z0-9._-]', '_'
    $OutputDirectory = Join-Path $PSScriptRoot ('outputs\' + $runName + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Output directory already exists: $resolvedOutput"
}
function Convert-ToLinuxPath([string]$InputPath) {
    $converted = & wsl.exe -d $Distribution -- wslpath -a -u $InputPath
    if ($LASTEXITCODE -ne 0) { throw "Cannot convert path: $InputPath" }
    return ($converted | Out-String).Trim()
}
$scriptPath = Convert-ToLinuxPath (Join-Path $PSScriptRoot 'extract_missiles.py')
$arguments = @('-d', $Distribution, '--', 'env', "PYTHONPATH=$ModuleDirectory", 'python3', $scriptPath,
    '--game', (Convert-ToLinuxPath $resolvedGame),
    '--replay', (Convert-ToLinuxPath $resolvedReplay),
    '--output', (Convert-ToLinuxPath $resolvedOutput))
if ($Weapon) { $arguments += @('--weapon', $Weapon) }
if ($Entity) { $arguments += @('--entity', $Entity) }
if ($LocalPackage) { $arguments += @('--local-package', (Convert-ToLinuxPath (Resolve-Path -LiteralPath $LocalPackage).Path)) }
& wsl.exe @arguments
if ($LASTEXITCODE -ne 0) { throw "Replay export failed; inspect the output logs: $resolvedOutput" }
$finalizeScript = Convert-ToLinuxPath (Join-Path $PSScriptRoot 'finalize_export.py')
& wsl.exe -d $Distribution -- python3 $finalizeScript (Convert-ToLinuxPath $resolvedOutput)
if ($LASTEXITCODE -ne 0) { throw "Final export log audit failed: $resolvedOutput" }
Write-Output "Exported to: $resolvedOutput"
