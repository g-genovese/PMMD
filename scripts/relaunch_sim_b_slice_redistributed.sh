#!/usr/bin/env bash
# =============================================================================
# relaunch_sim_b_slice_redistributed.sh
# =============================================================================
# Rilancia SOLO la slice del Server B (trials 40-61, 22 trials) della Sim B
# v4.3 partial-order tracking, redistribuita sui 3 server.
#
# CONTEXT:
#   - Run originale (launch_sim_b_distributed.sh): A [0-39], B [40-61],
#     C-S0 [62-94], C-S1 [95-127]
#   - A e C hanno completato con successo
#   - B è stato OOM-killato (anche a 10 workers: 171 GB > soglia 165 GB)
#   - Lo slice [40-61] (22 trials) deve essere rilanciato
#
# NEW WORKLOAD PARTITION (proporzionale a workers disponibili, 33 totali):
#   Server A (192 GB):           7 trials @ 11 workers → [40-46], ~4h
#   Server B (192 GB):           5 trials @  8 workers → [47-51], ~4h (137 GB)
#   Server C socket 0 (256 GB):  5 trials @  7 workers → [52-56], ~4h
#   Server C socket 1 (256 GB):  5 trials @  7 workers → [57-61], ~4h
#   Total: 22 trials, ~4h wall (vs ~8h della slice originale di B su B solo)
#
# SEED STRATEGY:
#   Stesso base SEED=20260518 e trial-start originali → riproducibile, niente
#   sovrapposizione con i trial già completati da A [0-39] e C [62-127].
#
# OUTPUT FILES:
#   Naming con range trial per evitare collisioni con i file originali di A/C:
#     sim_b_L12_A_t40-46.json   (+ _curves.npz, + per-trial PO files)
#     sim_b_L12_B_t47-51.json
#     sim_b_L12_C0_t52-56.json
#     sim_b_L12_C1_t57-61.json
#
# PREREQUISITES:
#   1. SSH aliases server-A, server-B, server-C in ~/.ssh/config
#   2. e8_percolation_v43_partial_order.py già presente su tutti i server
#      (rispedito comunque per sicurezza)
#   3. Python 3.10+ con numpy + numba su tutti i server
#   4. Verifica preliminare che NESSUN processo Python residuo della run
#      precedente sia ancora attivo su B (potrebbe trattenere memoria)
#
# USAGE:
#   chmod +x relaunch_sim_b_slice_redistributed.sh
#   ./relaunch_sim_b_slice_redistributed.sh
#
# DOPO IL COMPLETAMENTO:
#   1. rsync dei 4 file di output (vedi comandi a fine script)
#   2. Aggregare i 22 nuovi trial ai 106 già esistenti → 128 totali
#   3. Eseguire compute_dBS.py sull'insieme completo
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------

REMOTE_PATH="${REMOTE_PATH:-$HOME/sim_b}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./sim_b_outputs}"
SCRIPT="e8_percolation_v43_partial_order.py"

SEED=20260518         # IDENTICO al run originale (riproducibilità trial-by-trial)
L=12
TARGET_P=0.005800     # IDENTICO al run originale

# Per-server allocation (slice [40-61], 22 trials totali)
# A — 7 trials @ 11 workers (memoria 188 GB, già provata OK)
SERVER_A_TRIALS=7
SERVER_A_START=40
SERVER_A_WORKERS=11

# B — 5 trials @ 8 workers (memoria 137 GB, ampio margine vs 165 GB cap)
SERVER_B_TRIALS=5
SERVER_B_START=47
SERVER_B_WORKERS=8

# C socket 0 — 5 trials @ 7 workers
SERVER_C_TRIALS_S0=5
SERVER_C_START_S0=52
SERVER_C_WORKERS_S0=7

# C socket 1 — 5 trials @ 7 workers
SERVER_C_TRIALS_S1=5
SERVER_C_START_S1=57
SERVER_C_WORKERS_S1=7

# Sanity check
TOTAL=$((SERVER_A_TRIALS + SERVER_B_TRIALS + SERVER_C_TRIALS_S0 + SERVER_C_TRIALS_S1))
if [ "$TOTAL" -ne 22 ]; then
    echo "ERROR: trials don't sum to 22 (got $TOTAL)"
    exit 1
fi

# Verify trial-range continuity [40, 61]
A_END=$((SERVER_A_START + SERVER_A_TRIALS - 1))
B_END=$((SERVER_B_START + SERVER_B_TRIALS - 1))
S0_END=$((SERVER_C_START_S0 + SERVER_C_TRIALS_S0 - 1))
S1_END=$((SERVER_C_START_S1 + SERVER_C_TRIALS_S1 - 1))

if [ "$SERVER_B_START" -ne "$((A_END + 1))" ] \
   || [ "$SERVER_C_START_S0" -ne "$((B_END + 1))" ] \
   || [ "$SERVER_C_START_S1" -ne "$((S0_END + 1))" ] \
   || [ "$S1_END" -ne 61 ]; then
    echo "ERROR: trial ranges non-contigui o non terminano a 61"
    echo "  A:    [$SERVER_A_START, $A_END]"
    echo "  B:    [$SERVER_B_START, $B_END]"
    echo "  C-S0: [$SERVER_C_START_S0, $S0_END]"
    echo "  C-S1: [$SERVER_C_START_S1, $S1_END]"
    exit 1
fi

echo "================================================================"
echo "Sim B slice rerun — redistribuzione [40-61]"
echo "================================================================"
echo "Trial ranges (contigui, nessuna sovrapposizione, totale 22):"
echo "  Server A   :  [$SERVER_A_START, $A_END]      ($SERVER_A_TRIALS trials @ $SERVER_A_WORKERS workers, ~188 GB)"
echo "  Server B   :  [$SERVER_B_START, $B_END]      ($SERVER_B_TRIALS trials @ $SERVER_B_WORKERS workers, ~137 GB *SAFE*)"
echo "  Server C-S0:  [$SERVER_C_START_S0, $S0_END]      ($SERVER_C_TRIALS_S0 trials @ $SERVER_C_WORKERS_S0 workers)"
echo "  Server C-S1:  [$SERVER_C_START_S1, $S1_END]      ($SERVER_C_TRIALS_S1 trials @ $SERVER_C_WORKERS_S1 workers)"
echo "----------------------------------------------------------------"
echo "Seed:    $SEED  (identico al run originale → trials identici)"
echo "Target p: $TARGET_P"
echo "Wall time atteso: ~4h (tutti in 1 batch)"
echo "================================================================"
echo ""

# --- Preflight check: verifica che B non abbia processi orfani ---------------

echo "[Preflight] Controllo processi Python residui su server-B..."
ORPHAN_CHECK=$(ssh server-B "pgrep -f e8_percolation_v43_partial_order || true" || echo "")
if [ -n "$ORPHAN_CHECK" ]; then
    echo "  ⚠️  Trovati processi residui su server-B:"
    echo "$ORPHAN_CHECK" | sed 's/^/    PID /'
    echo "  Rimuoverli prima di procedere:"
    echo "    ssh server-B 'pkill -f e8_percolation_v43_partial_order'"
    echo "  Quindi rilanciare questo script."
    exit 1
fi
echo "  ✓ Nessun processo orfano su B"
echo ""

# --- Preparation -------------------------------------------------------------

mkdir -p "$LOCAL_OUTPUT_DIR"

echo "[Setup] Ridistribuzione script sui 3 server (overwrite di sicurezza)..."
for SRV in server-A server-B server-C; do
    ssh "$SRV" "mkdir -p $REMOTE_PATH"
    scp -q "$SCRIPT" "$SRV:$REMOTE_PATH/"
    echo "  ✓ $SRV"
done
echo ""

# --- Launch ------------------------------------------------------------------

ARGS_BASE="--L $L --seed $SEED --save-curves \
           --track-partial-order --target-p $TARGET_P \
           --save-partial-order-data"

echo "[Launch] Avvio dei job in background sui 3 server a $(date)..."
echo ""

# Server A
echo "  → Server A: $SERVER_A_TRIALS trials [$SERVER_A_START-$A_END] @ $SERVER_A_WORKERS workers"
ssh server-A "cd $REMOTE_PATH && \
    nohup python3 $SCRIPT $ARGS_BASE \
        --trials $SERVER_A_TRIALS \
        --trial-start $SERVER_A_START \
        --workers $SERVER_A_WORKERS \
        --output sim_b_L${L}_A_t${SERVER_A_START}-${A_END}.json \
    > sim_b_A_rerun.log 2>&1 &" &
SSHPID_A=$!

# Server B — workers ridotti per evitare OOM
echo "  → Server B: $SERVER_B_TRIALS trials [$SERVER_B_START-$B_END] @ $SERVER_B_WORKERS workers (memoria sicura 137 GB)"
ssh server-B "cd $REMOTE_PATH && \
    nohup python3 $SCRIPT $ARGS_BASE \
        --trials $SERVER_B_TRIALS \
        --trial-start $SERVER_B_START \
        --workers $SERVER_B_WORKERS \
        --output sim_b_L${L}_B_t${SERVER_B_START}-${B_END}.json \
    > sim_b_B_rerun.log 2>&1 &" &
SSHPID_B=$!

# Server C — dual-socket NUMA
echo "  → Server C-S0: $SERVER_C_TRIALS_S0 trials [$SERVER_C_START_S0-$S0_END] @ $SERVER_C_WORKERS_S0 workers (NUMA 0)"
echo "  → Server C-S1: $SERVER_C_TRIALS_S1 trials [$SERVER_C_START_S1-$S1_END] @ $SERVER_C_WORKERS_S1 workers (NUMA 1)"
ssh server-C "cd $REMOTE_PATH && \
    nohup numactl --cpunodebind=0 --membind=0 \
        python3 $SCRIPT $ARGS_BASE \
            --trials $SERVER_C_TRIALS_S0 \
            --trial-start $SERVER_C_START_S0 \
            --workers $SERVER_C_WORKERS_S0 \
            --output sim_b_L${L}_C0_t${SERVER_C_START_S0}-${S0_END}.json \
        > sim_b_C0_rerun.log 2>&1 & \
    nohup numactl --cpunodebind=1 --membind=1 \
        python3 $SCRIPT $ARGS_BASE \
            --trials $SERVER_C_TRIALS_S1 \
            --trial-start $SERVER_C_START_S1 \
            --workers $SERVER_C_WORKERS_S1 \
            --output sim_b_L${L}_C1_t${SERVER_C_START_S1}-${S1_END}.json \
        > sim_b_C1_rerun.log 2>&1 &" &
SSHPID_C=$!

wait $SSHPID_A $SSHPID_B $SSHPID_C
echo ""
echo "[Status] Tutti i job lanciati a $(date)."
echo ""

# --- Monitoring instructions -------------------------------------------------

cat <<EOF
================================================================
MONITORAGGIO
================================================================
Per seguire l'avanzamento (4 log indipendenti):
  ssh server-A 'tail -f $REMOTE_PATH/sim_b_A_rerun.log'
  ssh server-B 'tail -f $REMOTE_PATH/sim_b_B_rerun.log'
  ssh server-C 'tail -f $REMOTE_PATH/sim_b_C0_rerun.log'
  ssh server-C 'tail -f $REMOTE_PATH/sim_b_C1_rerun.log'

CONTROLLO MEMORIA SU B (suggerito ogni 30 min nella prima ora):
  ssh server-B 'free -g; ps -eo pid,rss,cmd | grep e8_percolation | grep -v grep'
  → RSS totale atteso ≤ 137 GB; se cresce oltre 160 GB intervenire.

================================================================
RACCOLTA RISULTATI (dopo completamento, ~4h)
================================================================
  rsync -avz server-A:$REMOTE_PATH/sim_b_L${L}_A_t40-46* $LOCAL_OUTPUT_DIR/
  rsync -avz server-B:$REMOTE_PATH/sim_b_L${L}_B_t47-51* $LOCAL_OUTPUT_DIR/
  rsync -avz server-C:$REMOTE_PATH/sim_b_L${L}_C0_t52-56* $LOCAL_OUTPUT_DIR/
  rsync -avz server-C:$REMOTE_PATH/sim_b_L${L}_C1_t57-61* $LOCAL_OUTPUT_DIR/

A questo punto in $LOCAL_OUTPUT_DIR avrai:
  - I file originali di A (trials [0-39])
  - I 4 file nuovi (trials [40-61], slice ex-B redistribuita)
  - I file originali di C (trials [62-127])
  → 128 trials totali pronti per merge_sim_b_results.py + compute_dBS.py.
================================================================
EOF
