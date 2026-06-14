# run_pmmd_win.ps1 -- PMMD Koide worker for the WINDOWS workstation (Ryzen 9950X).
# The bash launcher (run_pmmd.sh) cannot drive a native-Windows box, so run this
# directly on the workstation. It activates the Windows venv, runs its slice of
# the L-scan, and writes a JSON-lines result file. Then copy that file into the
# results dir of the orchestrating Linux machine for the final merge.
#
# Run in PowerShell:   .\run_pmmd_win.ps1
# (If blocked: powershell -ExecutionPolicy Bypass -File .\run_pmmd_win.ps1)

# >>> FILL THESE IN <<<
$VENV   = "$env:USERPROFILE\rs\Scripts\Activate.ps1"   # Windows venv activate
$SCRIPT = "$env:USERPROFILE\rs\pmmd_koide_hpc.py"      # script location
$DIM    = 2                       # 2 = continuum test ; 3 = Hopfion
$Q      = 3
$SIGN   = "zolotarev"; $POLES = 24; $NDEFL = 24
$LS     = "256,288,320"           # this workstation's slice of L values
$THREADS = 16                     # Ryzen 9950X = 16 physical cores
$OUT    = "$env:USERPROFILE\pmmd_res_workstation.jsonl"
# ----------------------------------------------------------------------

# cap BLAS threads (single NUMA domain on the 9950X, no pinning needed)
$env:OMP_NUM_THREADS      = "$THREADS"
$env:OPENBLAS_NUM_THREADS = "$THREADS"
$env:MKL_NUM_THREADS      = "$THREADS"
# CPU only here (the GPU is on the notebook, not this box)
$env:PMMD_BACKEND = "numpy"

Write-Host "[run_pmmd_win] activating venv: $VENV"
& $VENV

Write-Host "[run_pmmd_win] dim=$DIM L=($LS) -> $OUT"
if (Test-Path $OUT) { Remove-Item $OUT }

python $SCRIPT --dim $DIM --Lscan $LS --Q $Q `
    --sign $SIGN --poles $POLES --ndefl $NDEFL --out $OUT

Write-Host ""
Write-Host "[run_pmmd_win] done. Result file: $OUT"
Write-Host "Copy it to the Linux orchestrator's results dir, e.g.:"
Write-Host "  scp `"$OUT`" user@<linux-ip>:~/pmmd_results/"
Write-Host "Then re-run the merge there:"
Write-Host "  python3 ~/rs/pmmd_koide_hpc.py --merge `$(ls ~/pmmd_results/*.jsonl | paste -sd, -)"
