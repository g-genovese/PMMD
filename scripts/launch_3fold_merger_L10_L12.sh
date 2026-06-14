#!/usr/bin/env bash
# launch_3fold_merger_L10_L12.sh
# ================================
# Distributed launcher for the 3-fold merger extension at L=10, L=12.
# 
# Existing v5.1 data (Stratum 1 verified):
#   - L=6: 32 trials, ~25,000 3-fold events
#   - L=8: 32 trials, ~28,000 3-fold events  
#   - Combined: 53,611 events, z = -0.39 sigma match to framework's
#     (1/4 unanimous, 3/4 split) prediction
#
# This extension adds:
#   - L=10: 32 trials, expected ~80,000-100,000 3-fold events
#   - L=12: 16 trials, expected ~140,000-180,000 3-fold events
#   - Combined post-extension: ~300,000+ events
#
# Memory requirements (numba-optimized v2 kernel):
#   - L=10: ~3.3 GB per worker
#   - L=12: ~14 GB per worker
#
# Time estimates (per trial, single core):
#   - L=10: ~1.5-3 hours
#   - L=12: ~8-15 hours
#
# Server allocations (after Sim B completes):
#   - Server A (EPYC 7282, 16c/32t, 192 GB): 32 L=10 trials with 16 workers
#                                              -> 8 batches of 4 hours = 32h wall
#                                              OR launch separately on each server
#   - Server B (Xeon 4210R, 10c/20t, 192 GB): 16 L=12 trials with 10 workers
#                                              -> takes ~24-40h wall
#   - Server C (2x E5-2690v4, 28c/56t, 256 GB): the other half of L=12 (NUMA split)
#                                                -> takes ~16-30h wall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/e8_3fold_merger_stats_v2.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "ERROR: $PY_SCRIPT not found"
    exit 1
fi

echo "================================================================"
echo "3-fold merger extension launcher"
echo "Targets: L=10 (32 trials), L=12 (16 trials)"
echo "================================================================"
echo

# Server A: L=10, 32 trials, 16 workers
launch_serverA_L10() {
    nohup python3 "$PY_SCRIPT" \
        --L 10 --trials 32 --workers 16 \
        --seed 42 \
        --output e8_3fold_L10_serverA.json \
        > log_L10_serverA.out 2>&1 &
    echo "Launched on Server A: L=10, 32 trials, 16 workers"
    echo "  PID: $!"
    echo "  Estimated wall time: 6-8 hours"
}

# Server B: L=12, 16 trials, 10 workers
launch_serverB_L12_half() {
    nohup python3 "$PY_SCRIPT" \
        --L 12 --trials 16 --workers 10 \
        --seed 100 \
        --output e8_3fold_L12_serverB.json \
        > log_L12_serverB.out 2>&1 &
    echo "Launched on Server B: L=12, 16 trials, 10 workers"
    echo "  PID: $!"
    echo "  Estimated wall time: 20-30 hours"
}

# Server C (NUMA-split for 2-socket Broadwell):
launch_serverC_L12_NUMA() {
    # Socket 0: 16 trials
    nohup numactl --cpunodebind=0 --membind=0 \
        python3 "$PY_SCRIPT" \
        --L 12 --trials 16 --workers 14 \
        --seed 200 \
        --output e8_3fold_L12_serverC_sock0.json \
        > log_L12_serverC_sock0.out 2>&1 &
    echo "Launched on Server C socket 0: L=12, 16 trials, 14 workers"
    echo "  PID: $!"
    # Socket 1: 16 trials
    nohup numactl --cpunodebind=1 --membind=1 \
        python3 "$PY_SCRIPT" \
        --L 12 --trials 16 --workers 14 \
        --seed 300 \
        --output e8_3fold_L12_serverC_sock1.json \
        > log_L12_serverC_sock1.out 2>&1 &
    echo "Launched on Server C socket 1: L=12, 16 trials, 14 workers"
    echo "  PID: $!"
    echo "  Total: L=12 32 trials across Server C, estimated 16-25 hours wall"
}

# Detect hostname and launch accordingly (USER CUSTOMIZES THIS)
case "$(hostname)" in
    *serverA*) launch_serverA_L10 ;;
    *serverB*) launch_serverB_L12_half ;;
    *serverC*) launch_serverC_L12_NUMA ;;
    *)
        echo "USAGE:"
        echo "  On serverA: ssh serverA \"cd $SCRIPT_DIR && ./$(basename $0)\""
        echo "  On serverB: ssh serverB \"cd $SCRIPT_DIR && ./$(basename $0)\""
        echo "  On serverC: ssh serverC \"cd $SCRIPT_DIR && ./$(basename $0)\""
        echo
        echo "Or call functions directly:"
        echo "  launch_serverA_L10"
        echo "  launch_serverB_L12_half"
        echo "  launch_serverC_L12_NUMA"
        echo
        echo "Monitor: tail -f log_*.out"
        ;;
esac
