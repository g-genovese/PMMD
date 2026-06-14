# Sim C — p-Scan Execution Plan (d_BS on the disordered foam, multi-snapshot)

**Goal.** Map d_BS(p) on the *disordered* percolating-foam ensemble across the
foam-connected p-range, and extrapolate L -> infinity, to close the Stratum-2
promotion of d_BS = 4 (Remark rem:dBS-cutproject-v6). A p-scan (not a single p)
protects against the objection that the result depends on an ad-hoc choice of p.

**Two independent estimators must converge on d_BS = 4 across the foam-connected range:**
1. Intrinsic       -- compute_dBS.py on the 8D activation-time partial order.
2. Cut-and-project -- sim_c_cutproject_dBS.py: 4D golden-ratio projection + longest-chain.

---

## What is reused from Sim B (and what is not)

- REUSED -- calibration curves. Sim B's S_max(p), chi(p) at L=12 (128 trials) are
  already in sim_b_aggregate.npz. They fix the foam-connected p-range without guessing:
  p=0.0058 -> giant 45k (0.01% of N, critical-fractal incipient); p=0.008 -> 2.2M;
  p=0.012 -> 4.7M; p=0.020 -> 8.5M (foam-connected). chi-peak (effective p_c, L=12) ~ 0.0058.
- NOT REUSED -- d_BS. Sim B never extracted d_BS (d_BS_pertrial is all-NaN, "stub").
  Per-trial partial-order snapshots were saved only at p=0.0058 in separate
  *_po_trial*.npz files (on your servers, not in the aggregate).
- IF you still have Sim B's *_po_trial*.npz (L=12, p=0.0058): analyze them
  immediately for the "L=12 at p_c" point -- no regeneration needed:
    python sim_c_cutproject_dBS.py --po-files "<simB>_po_trial*.npz" --output cp_simB_L12_pc.json
    python compute_dBS.py          --po-files "<simB>_po_trial*.npz" --output dBS_simB_L12_pc.json

---

## Key efficiency: multi-snapshot (one sweep -> all p)

e8_percolation_v44_multisnapshot.py takes snapshots at SEVERAL p in a single
Newman-Ziff sweep (the 14h cost is the sweep; extra snapshots are nearly free).
Pass several values to --target-p. Validated: multi-snapshot cluster sizes match
single-p runs exactly.

Memory per worker (multi-snapshot adds n_snaps * N_idx * 4 bytes):
  ~3 GB (L=8, 3 snaps), ~12 GB (L=10, 3 snaps), ~22 GB (L=12, 3 snaps).
Set --workers so that workers * per-worker-RAM < 0.85 * server RAM.

Scan p-values: --target-p 0.008 0.012 0.020 (foam-connected). Add 0.0058 to also
include the critical point in the same sweep (4 snaps -> ~24 GB/worker at L=12).

---

## Phase 1 -- Generation (BALANCED across servers, one sweep per L)

Load balancing: at L=12 every server is RAM-limited (~22 GB/worker with 3 snapshots),
NOT core-limited; at L=8 it is core-limited. So each server takes a SHARE OF EVERY L
(not whole-L blocks), sized to its effective worker count, and runs L=12 -> L=10 -> L=8
in sequence. All three finish in ~3.1h (one L=12 wave dominates).

Effective workers / trial split:
  L=12 (RAM-cap ~22GB/w): C=9w/9 trials, A=7w/7, B=7w/7   (23 trials, 1 wave, ~2.5h)
  L=10 (~12GB/w):         C=18w/18,      A=14w/14, B=14w/14 (46 trials, 1 wave, ~0.55h)
  L=8  (~3GB/w):          C=32w/32,      A=18w/18, B=18w/10 (60 trials, 1 wave, ~0.1h)

Per-server wall-time estimate: ~3.1h each (2.5 + 0.55 + 0.1). Distinct --trial-start
per L avoids seed collisions ACROSS servers within the same L.

NOTE: time/trial is an estimate from L^8 scaling + the paper's ~14h L=12 figure; the
older E5-2690v4 cores (Server C) are slower per-core than EPYC 7282 / Xeon 4210R, so
this split slightly over-loads C. Calibrate after the FIRST L=12 trial on each server
(printed per-trial time) and shift a few trials if needed. If one server finishes early,
give it extra L=12 trials: rerun with higher --trials and a --trial-start past the used range.

# Server C -- 2x E5-2690v4, 56t, 256 GB
python e8_percolation_v44_multisnapshot.py --L 12 --trials 9  --trial-start 0  --workers 9 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L12_C.json
python e8_percolation_v44_multisnapshot.py --L 10 --trials 18 --trial-start 0  --workers 18 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L10_C.json
python e8_percolation_v44_multisnapshot.py --L 8  --trials 32 --trial-start 0  --workers 32 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L8_C.json

# Server A -- EPYC 7282, 32t, 192 GB
python e8_percolation_v44_multisnapshot.py --L 12 --trials 7  --trial-start 9  --workers 7 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L12_A.json
python e8_percolation_v44_multisnapshot.py --L 10 --trials 14 --trial-start 18 --workers 14 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L10_A.json
python e8_percolation_v44_multisnapshot.py --L 8  --trials 18 --trial-start 32 --workers 18 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L8_A.json

# Server B -- Xeon 4210R, 20t, 192 GB
python e8_percolation_v44_multisnapshot.py --L 12 --trials 7  --trial-start 16 --workers 7 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L12_B.json
python e8_percolation_v44_multisnapshot.py --L 10 --trials 14 --trial-start 32 --workers 14 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L10_B.json
python e8_percolation_v44_multisnapshot.py --L 8  --trials 10 --trial-start 50 --workers 10 \
  --seed 20260520 --track-partial-order --target-p 0.008 0.012 0.020 \
  --save-partial-order-data --output simc_L8_B.json

Ensemble totals: L=12 -> 23, L=10 -> 46, L=8 -> 60 trials (x3 p each via multi-snapshot).



---

## Phase 2 -- Analysis (gather all NPZ on one machine)

# Estimator 2 (cut-and-project): one call per L; reads ALL p in the glob and
# tags each record with its target_p for the scan.
for L in 8 10 12; do
  python sim_c_cutproject_dBS.py --po-files "simc_L${L}_*_po_p*_trial*.npz" \
    --output cp_L${L}.json
done

# Estimator 1 (intrinsic): per (L, p)
for L in 8 10 12; do for p in 0.008000 0.012000 0.020000; do
  python compute_dBS.py --po-files "simc_L${L}_*_po_p${p}_trial*.npz" \
    --n-pair-samples 20000 --output dBS_intr_L${L}_p${p}.json
done; done

At L=12 increase the chain subsample upper end:
  --chain-sizes 2000 4000 8000 16000 32000 64000 128000 256000

---

## Phase 3 -- Merge -> d_BS(p) curve + FSS

python merge_sim_c.py --inputs "cp_L*.json" --output simc_pscan_FINAL.json

Prints d_inf(p) for each p (longest-chain, finite-size-extrapolated in 1/L).
PROMOTION CRITERION: d_inf ~ 4 (within error) stable across the foam-connected
p-range, with the intrinsic estimator agreeing. A drift toward the critical
fractal dimension as p -> p_c is itself an honest, informative outcome to report.

---

## Sandbox validation already done (L<=6, few trials)
- multi-snapshot cluster sizes = single-p sizes (exact match): mechanism correct.
- cut-and-project longest-chain FSS -> 4.07 +/- 0.42 (foam-connected): consistent with d_BS=4.
- box-counting unreliable below ~1e5 points: rely on longest-chain.

## Files
- e8_percolation_v44_multisnapshot.py -- generation, multi-snapshot (new)
- compute_dBS.py                       -- intrinsic estimator (existing)
- sim_c_cutproject_dBS.py              -- cut-and-project estimator, p-tagged (updated)
- merge_sim_c.py                       -- d_BS(p) scan + FSS-per-p (updated)
