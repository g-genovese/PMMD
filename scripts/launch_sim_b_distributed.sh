#!/usr/bin/env bash
# =============================================================================
# launch_sim_b_distributed.sh
# =============================================================================
# Distributed Sim B run for v4.3 partial-order tracking across 3 servers.
#
# Workload partition (proportional to compute power):
#   Server A (AMD EPYC 7282,    16c/32t, 192 GB):  40 trials, 11 workers
#   Server B (Intel Xeon 4210R, 10c/20t, 192 GB):  22 trials, 11 workers
#   Server C (2x Xeon E5-2690v4, 28c/56t, 256 GB): 66 trials (33+33, NUMA split)
#                                                  7 workers per socket
# Total: 128 trials, ~15 hours wall time (vs ~33 hours single-server).
#
# PREREQUISITES:
#   1. SSH key-based access from launch host to server-A, server-B, server-C
#   2. Hostname aliases in ~/.ssh/config:
#        Host server-A
#            HostName ...
#        Host server-B
#            HostName ...
#        Host server-C
#            HostName ...
#   3. Script e8_percolation_v43_partial_order.py present at REMOTE_PATH on all servers
#   4. Python 3.10+ with numpy + numba on all servers
#
# USAGE:
#   chmod +x launch_sim_b_distributed.sh
#   ./launch_sim_b_distributed.sh
#
# After completion:
#   1. Verify all server outputs are in $LOCAL_OUTPUT_DIR
#   2. Run merge_sim_b_results.py to aggregate (TBD: companion script)
#   3. Run compute_dBS.py for d_BS extraction
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------

REMOTE_PATH="${REMOTE_PATH:-$HOME/sim_b}"               # Path on each server
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./sim_b_outputs}" # Local collection dir
SCRIPT="e8_percolation_v43_partial_order.py"

SEED=20260518         # Base seed for all servers
L=12                  # Lattice size
TARGET_P=0.005800     # p_c from L=12 v4.2 run, 128 trials

# Per-server allocation
# Format: (server_id  trials  trial_start  workers  numa_split)
SERVER_A_TRIALS=40
SERVER_A_START=0
SERVER_A_WORKERS=11

SERVER_B_TRIALS=22
SERVER_B_START=40
SERVER_B_WORKERS=11

# Server C is dual-socket; split into 2 jobs with NUMA pinning
SERVER_C_TRIALS_S0=33
SERVER_C_START_S0=62
SERVER_C_WORKERS_S0=7

SERVER_C_TRIALS_S1=33
SERVER_C_START_S1=95
SERVER_C_WORKERS_S1=7

# Sanity check: total trials = 128
TOTAL=$((SERVER_A_TRIALS + SERVER_B_TRIALS + SERVER_C_TRIALS_S0 + SERVER_C_TRIALS_S1))
if [ "$TOTAL" -ne 128 ]; then
    echo "ERROR: trials don't sum to 128 (got $TOTAL)"
    exit 1
fi

# Sanity check: no overlapping trial ranges
echo "Trial ranges:"
echo "  Server A:   [$SERVER_A_START, $((SERVER_A_START + SERVER_A_TRIALS - 1))]"
echo "  Server B:   [$SERVER_B_START, $((SERVER_B_START + SERVER_B_TRIALS - 1))]"
echo "  Server C S0: [$SERVER_C_START_S0, $((SERVER_C_START_S0 + SERVER_C_TRIALS_S0 - 1))]"
echo "  Server C S1: [$SERVER_C_START_S1, $((SERVER_C_START_S1 + SERVER_C_TRIALS_S1 - 1))]"
echo ""

# --- Preparation -------------------------------------------------------------

mkdir -p "$LOCAL_OUTPUT_DIR"

echo "Distributing script to all servers..."
for SRV in server-A server-B server-C; do
    ssh "$SRV" "mkdir -p $REMOTE_PATH"
    scp -q "$SCRIPT" "$SRV:$REMOTE_PATH/"
    echo "  $SRV: script uploaded"
done
echo ""

# --- Launch ------------------------------------------------------------------

echo "Launching distributed Sim B at $(date)..."
echo ""

# Common arguments
ARGS_BASE="--L $L --seed $SEED --save-curves \
           --track-partial-order --target-p $TARGET_P \
           --save-partial-order-data"

# Server A
echo "Server A: $SERVER_A_TRIALS trials, $SERVER_A_WORKERS workers"
ssh server-A "cd $REMOTE_PATH && \
    nohup python3 $SCRIPT $ARGS_BASE \
        --trials $SERVER_A_TRIALS \
        --trial-start $SERVER_A_START \
        --workers $SERVER_A_WORKERS \
        --output sim_b_L${L}_A.json \
    > sim_b_A.log 2>&1 &" &
SSHPID_A=$!

# Server B
echo "Server B: $SERVER_B_TRIALS trials, $SERVER_B_WORKERS workers"
ssh server-B "cd $REMOTE_PATH && \
    nohup python3 $SCRIPT $ARGS_BASE \
        --trials $SERVER_B_TRIALS \
        --trial-start $SERVER_B_START \
        --workers $SERVER_B_WORKERS \
        --output sim_b_L${L}_B.json \
    > sim_b_B.log 2>&1 &" &
SSHPID_B=$!

# Server C: dual-socket NUMA-pinned
echo "Server C (socket 0): $SERVER_C_TRIALS_S0 trials, $SERVER_C_WORKERS_S0 workers (NUMA node 0)"
echo "Server C (socket 1): $SERVER_C_TRIALS_S1 trials, $SERVER_C_WORKERS_S1 workers (NUMA node 1)"
ssh server-C "cd $REMOTE_PATH && \
    nohup numactl --cpunodebind=0 --membind=0 \
        python3 $SCRIPT $ARGS_BASE \
            --trials $SERVER_C_TRIALS_S0 \
            --trial-start $SERVER_C_START_S0 \
            --workers $SERVER_C_WORKERS_S0 \
            --output sim_b_L${L}_C0.json \
        > sim_b_C0.log 2>&1 & \
    nohup numactl --cpunodebind=1 --membind=1 \
        python3 $SCRIPT $ARGS_BASE \
            --trials $SERVER_C_TRIALS_S1 \
            --trial-start $SERVER_C_START_S1 \
            --workers $SERVER_C_WORKERS_S1 \
            --output sim_b_L${L}_C1.json \
        > sim_b_C1.log 2>&1 &" &
SSHPID_C=$!

# Wait for all SSH launches to complete
wait $SSHPID_A $SSHPID_B $SSHPID_C
echo ""
echo "All jobs launched at $(date). They are running in background on the servers."
echo ""
echo "Monitor progress:"
echo "  ssh server-A 'tail -f $REMOTE_PATH/sim_b_A.log'"
echo "  ssh server-B 'tail -f $REMOTE_PATH/sim_b_B.log'"
echo "  ssh server-C 'tail -f $REMOTE_PATH/sim_b_C0.log'  (socket 0)"
echo "  ssh server-C 'tail -f $REMOTE_PATH/sim_b_C1.log'  (socket 1)"
echo ""
echo "Wait for completion, then collect results:"
echo "  rsync -avz server-A:$REMOTE_PATH/sim_b_L${L}_A* $LOCAL_OUTPUT_DIR/"
echo "  rsync -avz server-B:$REMOTE_PATH/sim_b_L${L}_B* $LOCAL_OUTPUT_DIR/"
echo "  rsync -avz server-C:$REMOTE_PATH/sim_b_L${L}_C* $LOCAL_OUTPUT_DIR/"
echo ""
echo "Expected wall time: ~12-15 hours (slowest server determines)"
echo ""
echo "Output files per server:"
echo "  sim_b_L${L}_X.json                       (summary)"
echo "  sim_b_L${L}_X_curves.npz                 (binned curves)"
echo "  sim_b_L${L}_X_po_trial####.npz           (per-trial partial order, ~30 MB each)"
