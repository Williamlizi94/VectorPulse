$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
$ncuPath = (Get-ChildItem -LiteralPath (Join-Path $projectRoot 'out/nsight') -Filter ncu.exe -Recurse | Select-Object -First 1).FullName
$profileExe = Join-Path $projectRoot 'out/build-cuda/vectorpulse_cuda_profile.exe'
$outputRoot = Join-Path $projectRoot 'docs/benchmarks/nsight'
$sections = @('LaunchStats', 'Occupancy', 'SpeedOfLight', 'ComputeWorkloadAnalysis', 'MemoryWorkloadAnalysis_Tables', 'SchedulerStats', 'WarpStateStats', 'SourceCounters')
foreach ($count in @(10000,100000,1000000)) {
    foreach ($kernel in @('naive','block')) {
        $stem = Join-Path $outputRoot "$count-$kernel"
        $ncuArguments = @('--launch-skip','3','--launch-count','3','--clock-control','none','--cache-control','all','--force-overwrite','--export',$stem)
        foreach ($section in $sections) { $ncuArguments += @('--section',$section) }
        $ncuArguments += @($profileExe,"$count",$kernel)
        & $ncuPath @ncuArguments *> "$stem-collection.txt"
        if ($LASTEXITCODE -ne 0) { throw "Nsight collection failed: $stem; inspect its collection log" }
        & $ncuPath --import "$stem.ncu-rep" --page raw --csv --print-units base *> "$stem-metrics.csv"
        if ($LASTEXITCODE -ne 0) { throw "Nsight export failed: $stem" }
        & $ncuPath --import "$stem.ncu-rep" --page details *> "$stem-details.txt"
        if ($LASTEXITCODE -ne 0) { throw "Nsight details export failed: $stem" }
    }
}
'complete' | Set-Content -LiteralPath (Join-Path $outputRoot 'collection-status.txt')
