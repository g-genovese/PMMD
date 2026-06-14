#!/usr/bin/env bash
# run_pmmd.sh -- PMMD Koide continuum scan with DYNAMIC LOAD BALANCING.
# A shared flock-protected queue holds the L values; each SLOT (machine/socket)
# pulls the next L when it becomes free. Self-balancing across heterogeneous
# Linux + Windows machines, fast or slow -- no manual L-splitting. Largest L are
# queued first (longest-processing-time-first) to minimise tail stragglers.
#
# SLOT entry:  "TARGET:NUMANODE:BACKEND:SHELL"
#   TARGET   = "local" | "user@IP"
#   NUMANODE = numactl socket number | "x"
#   BACKEND  = "cpu" | "gpu"
#   SHELL    = "sh" (Linux) | "cmd" (Windows OpenSSH)
# A dual-socket box = TWO sh slots (nodes 0 and 1). A GPU notebook can be a "gpu"
# slot, but see the GPU caveat (host LOBPCG limits the win).
#
# >>> Linux paths <<<
VENV="$HOME/rs/bin/activate"
SCRIPT="$HOME/rs/pmmd_koide_hpc.py"
# >>> Windows paths (cmd slots) <<<
VENV_WIN='%USERPROFILE%\rs\Scripts\activate.bat'
SCRIPT_WIN='%USERPROFILE%\rs\pmmd_koide_hpc.py'
WIN_THREADS=16
# >>> common <<<
DIM=2; Q=3
SIGN="zolotarev"; POLES=24; NDEFL=24
THREADS_PER_SOCKET=14

# Full set of L values to scan (the queue). Edit freely.
TASKS=( 64 96 128 160 192 224 256 288 320 384 )

SLOTS=(
  "local:x:cpu:sh"                                  # notebook 1 (HX 370), orchestrator
  "root@192.168.51.73:0:cpu:sh"                     # numa socket 0
  "root@192.168.51.73:1:cpu:sh"                     # numa socket 1
  "aziz@192.168.7.138:x:cpu:sh"                     # linux server
  "g.genovese@newfdm.local@192.168.51.149:x:cpu:cmd" # Windows workstation (domain user, NO quotes)
  # "user@192.168.X.Y:x:cpu:sh"                     # notebook 2 (7945HX) -- add its IP
)
# NOTE on domain usernames: write them plainly as user@host, e.g.
#   g.genovese@newfdm.local@192.168.51.149
# The launcher splits on the LAST '@' (user="g.genovese@newfdm.local",
# host="192.168.51.149") and uses ssh -l / scp -o User=, which tolerate the '@'.
# Do NOT wrap the username in quotes inside the slot string.
# ----------------------------------------------------------------------

set -u
RESDIR="$HOME/pmmd_results"; mkdir -p "$RESDIR"
QF="$RESDIR/queue.txt"; LOCKF="$RESDIR/queue.lock"
TENV="OMP_NUM_THREADS=$THREADS_PER_SOCKET OPENBLAS_NUM_THREADS=$THREADS_PER_SOCKET MKL_NUM_THREADS=$THREADS_PER_SOCKET"
source "$VENV"

# queue: largest L first
printf "%s\n" "${TASKS[@]}" | sort -rn > "$QF"
echo "[run_pmmd] dim=$DIM ; ${#SLOTS[@]} slots ; $(wc -l < "$QF") tasks (largest-first)"

atomic_pop() {                     # prints next L and removes it, atomically
    exec 9>"$LOCKF"; flock 9
    local line; line=$(head -n1 "$QF")
    [ -n "$line" ] && { tail -n +2 "$QF" > "$QF.tmp"; mv "$QF.tmp" "$QF"; }
    flock -u 9; exec 9>&-
    printf "%s" "$line"
}

run_one() {                        # $1=slot fields  $2=L
    local slot="$1" L="$2"
    local target="${slot%%:*}"; local r1="${slot#*:}"
    local node="${r1%%:*}"; local r2="${r1#*:}"
    local backend="${r2%%:*}"; local shell="${r2#*:}"
    local lf="$RESDIR/pmmd_res_L${L}.jsonl"
    # split user/host on the LAST '@' so domain usernames (user@domain) survive
    local host user
    if [ "$target" = "local" ]; then host="local"; user=""
    else host="${target##*@}"; user="${target%@*}"; fi

    if [ "$shell" = "cmd" ]; then
        # ---- Windows remote via OpenSSH -> cmd.exe (paths use %USERPROFILE%) ----
        local outw="%USERPROFILE%\\pmmd_res_L${L}.jsonl"
        local setenv="set OMP_NUM_THREADS=${WIN_THREADS}&& set OPENBLAS_NUM_THREADS=${WIN_THREADS}&& set MKL_NUM_THREADS=${WIN_THREADS}&& set PMMD_BACKEND=numpy"
        ssh -l "$user" "$host" "call \"${VENV_WIN}\" && ${setenv}&& del /q \"${outw}\" 2>nul & python \"${SCRIPT_WIN}\" --dim $DIM --L $L --Q $Q --sign $SIGN --poles $POLES --ndefl $NDEFL --out \"${outw}\"" \
            && scp -q -o "User=$user" "$host:pmmd_res_L${L}.jsonl" "$lf"
    elif [ "$target" = "local" ]; then
        # ---- local Linux: use the orchestrator's own venv/script paths ----
        local numa=""; [ "$node" != "x" ] && numa="numactl --cpunodebind=${node} --membind=${node}"
        local gpu=""; [ "$backend" = "gpu" ] && gpu="PMMD_BACKEND=cupy"
        eval "$TENV $gpu $numa python3 $SCRIPT --dim $DIM --L $L --Q $Q --sign $SIGN --poles $POLES --ndefl $NDEFL --out $lf"
    else
        # ---- remote Linux: paths are relative to the REMOTE home ($HOME there) ----
        local numa=""; [ "$node" != "x" ] && numa="numactl --cpunodebind=${node} --membind=${node}"
        local gpu=""; [ "$backend" = "gpu" ] && gpu="PMMD_BACKEND=cupy"
        ssh -l "$user" "$host" "source \$HOME/rs/bin/activate && $TENV $gpu $numa python3 \$HOME/rs/pmmd_koide_hpc.py --dim $DIM --L $L --Q $Q --sign $SIGN --poles $POLES --ndefl $NDEFL --out \$HOME/pmmd_res_L${L}.jsonl" \
            && scp -q -o "User=$user" "$host:pmmd_res_L${L}.jsonl" "$lf"
    fi
}

slot_loop() {                      # one background worker per slot
    local slot="$1" sid="$2"
    while true; do
        local L; L=$(atomic_pop)
        [ -z "$L" ] && break
        local t0=$SECONDS
        echo "[slot $sid] $slot  <- L=$L"
        if run_one "$slot" "$L"; then
            echo "[slot $sid] L=$L done in $((SECONDS-t0))s"
        else
            echo "[slot $sid][warn] L=$L FAILED (requeue)"; 
            exec 9>"$LOCKF"; flock 9; echo "$L" >> "$QF"; flock -u 9; exec 9>&-
            sleep 3
        fi
    done
}

# launch all slot loops in parallel
pids=(); s=0
for slot in "${SLOTS[@]}"; do
    slot_loop "$slot" "$s" & pids+=($!); s=$((s+1))
done
echo "[run_pmmd] ${#pids[@]} slots draining the queue..."
for p in "${pids[@]}"; do wait "$p"; done

FILES=$(ls "$RESDIR"/pmmd_res_L*.jsonl 2>/dev/null | paste -sd, -)
echo "[run_pmmd] merging: $FILES"
python3 "$SCRIPT" --merge "$FILES"
echo "[run_pmmd] done. Report the 'Q_K(L->inf)' line back to consolidate."

# ---- NOTES ----
# * DYNAMIC BALANCING: fast machines (9950X, Zen5/Zen4 notebooks) auto-take more
#   and bigger L; slow Broadwell sockets take fewer. No tuning. A failed task is
#   requeued so another slot retries it.
# * Validate each machine first (want SELF-TEST PASSED):
#     Linux:   source ~/rs/bin/activate && python3 ~/rs/pmmd_koide_hpc.py --selftest
#     Windows: ssh user@IP "call \"%USERPROFILE%\rs\Scripts\activate.bat\" && python \"%USERPROFILE%\rs\pmmd_koide_hpc.py\" --selftest"
# * GPU caveat: matvec on device, LOBPCG on host -> limited win, esp small 2D.
#   Add a "gpu" slot only after measuring one L cpu-vs-gpu on that notebook.
# * Copy pmmd_koide_hpc.py into ~/rs/ (Linux) / %USERPROFILE%\rs\ (Windows) first.
# * flock is used on the orchestrator (Linux) only -- fine.
