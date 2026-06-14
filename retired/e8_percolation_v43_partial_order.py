"""
e8_percolation_v43_partial_order.py
====================================
v4.3 extension of v4.2: adds causal-set partial-order tracking for d_BS extraction
(Bombelli-Sorkin combinatorial dimension), and --trial-start for distributed runs.

DROP-IN REPLACEMENT for v4.2. All v4.2 behaviour preserved when --track-partial-order
is NOT given. With the flag enabled, the kernel additionally:

  - Tracks activation_time[v]: the bond-index i at which vertex v was first activated
  - Takes a snapshot of parent[] at bond index corresponding to --target-p
  - Returns these arrays for off-line d_BS analysis (compute_dBS.py)

NEW CLI options (relative to v4.2):
  --trial-start N           (int)  Offset for trial seed indexing. Required for distributed
                                   runs to avoid seed collisions across servers.
                                   Per-trial seed = seed + trial-start + i, for i in [0, trials).
  --track-partial-order     (flag) Enable activation-time tracking + parent snapshot at target-p
  --target-p P              (float) p value at which to take the parent[] snapshot;
                                   default = 0.005800 (chi peak from L=12 v4.2 run, 128 trials).
  --save-partial-order-data (flag) Save activation_time + parent_snapshot to a per-trial NPZ
                                   (only meaningful with --track-partial-order)

MEMORY (per worker):
  Without --track-partial-order: ~10.3 GB (same as v4.2 at L=12)
  With    --track-partial-order: ~17.1 GB at L=12 (adds two N_idx int32 arrays)

USAGE for distributed Sim B run at L=12, 128 trials across 3 servers:

  # Server A (EPYC 7282, 192 GB, ~11 workers cap with PO tracking):
  python e8_percolation_v43_partial_order.py \\
    --L 12 --trials 40 --trial-start 0 --workers 11 \\
    --seed 20260518 --track-partial-order --target-p 0.005800 \\
    --save-curves --save-partial-order-data \\
    --output sim_b_L12_A.json

  # Server B (Xeon 4210R, 192 GB, ~11 workers cap):
  python e8_percolation_v43_partial_order.py \\
    --L 12 --trials 22 --trial-start 40 --workers 11 \\
    --seed 20260518 --track-partial-order --target-p 0.005800 \\
    --save-curves --save-partial-order-data \\
    --output sim_b_L12_B.json

  # Server C (2x E5-2690v4, 256 GB, NUMA pinning split 33+33):
  numactl --cpunodebind=0 --membind=0 \\
    python e8_percolation_v43_partial_order.py \\
      --L 12 --trials 33 --trial-start 62 --workers 7 \\
      --seed 20260518 --track-partial-order --target-p 0.005800 \\
      --save-curves --save-partial-order-data \\
      --output sim_b_L12_C0.json &
  numactl --cpunodebind=1 --membind=1 \\
    python e8_percolation_v43_partial_order.py \\
      --L 12 --trials 33 --trial-start 95 --workers 7 \\
      --seed 20260518 --track-partial-order --target-p 0.005800 \\
      --save-curves --save-partial-order-data \\
      --output sim_b_L12_C1.json &
  wait

  After all servers finish: aggregate with merge_sim_b_results.py (companion script).
  Then run compute_dBS.py on aggregated NPZ files for d_BS extraction.
"""

import argparse
import json
import os
import sys
import time
import platform
import numpy as np
from multiprocessing import cpu_count, get_context

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator


# =====================================================================
# Precompute E_8 root shifts (identical to v4.2)
# =====================================================================

def make_e8_shifts():
    typeA = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    d = np.zeros(8, dtype=np.int8)
                    d[i] = s1
                    d[j] = s2
                    typeA.append(d)
    typeA = np.stack(typeA)
    assert typeA.shape == (112, 8)

    typeB_h0 = []
    typeB_h1 = []
    for mask in range(256):
        if bin(mask).count("1") % 2 != 0:
            continue
        s = np.array([-1 if (mask >> k) & 1 else 1 for k in range(8)],
                     dtype=np.int8)
        d0 = ((s - 1) // 2).astype(np.int8)
        d1 = ((s + 1) // 2).astype(np.int8)
        typeB_h0.append(d0)
        typeB_h1.append(d1)
    typeB_h0 = np.stack(typeB_h0)
    typeB_h1 = np.stack(typeB_h1)
    assert typeB_h0.shape == (128, 8)
    assert typeB_h1.shape == (128, 8)
    return typeA, typeB_h0, typeB_h1


def _build_order(L, seed):
    L8 = L ** 8
    order = np.empty(L8 + 2, dtype=np.int64)
    pos = 0
    for h in range(2):
        for lex_idx in range(L8):
            x = lex_idx
            s = 0
            for k in range(8):
                s += x % L
                x //= L
            if s % 2 == 0:
                order[pos] = h * L8 + lex_idx
                pos += 1
    order = order[:pos]
    np.random.seed(seed)
    np.random.shuffle(order)
    return order


# =====================================================================
# v4.2 kernel (UNCHANGED, retained for non-PO runs)
# =====================================================================

@njit(cache=True, fastmath=True)
def _newman_ziff_kernel_v42(L, order, typeA, typeB_h0, typeB_h1, n_bins):
    """Original v4.2 kernel: tracks (S_max, sum_sq, n_clusters, sum_s_log_s)."""
    L8 = L ** 8
    N_idx = 2 * L8
    N_valid = order.shape[0]

    parent = np.full(N_idx, -1, dtype=np.int32)
    size = np.zeros(N_idx, dtype=np.int32)

    largest = np.int32(0)
    sum_sq = np.int64(0)
    n_clusters = np.int64(0)
    sum_s_log_s = np.float64(0.0)

    bin_size = max(N_valid // n_bins, 1)
    S_max_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_bin = np.zeros(n_bins, dtype=np.float64)
    n_clusters_bin = np.zeros(n_bins, dtype=np.float64)
    sum_s_log_s_bin = np.zeros(n_bins, dtype=np.float64)

    c = np.empty(8, dtype=np.int64)

    for i in range(N_valid):
        v = order[i]
        if v >= L8:
            h = 1
            lex_v = v - L8
        else:
            h = 0
            lex_v = v

        x = lex_v
        for k in range(8):
            c[k] = x % L
            x //= L

        parent[v] = v
        size[v] = 1
        sum_sq += 1
        n_clusters += 1
        if largest < 1:
            largest = 1
        current = np.int64(v)

        for ri in range(112):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeA[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h * L8 + lex_u

            if parent[u] == -1:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt

            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt

            if ra == rb:
                continue

            sa = size[ra]
            sb = size[rb]
            sum_sq += 2 * np.int64(sa) * np.int64(sb)

            new_size = sa + sb
            fsa = np.float64(sa)
            fsb = np.float64(sb)
            fnew = np.float64(new_size)
            sum_s_log_s -= fsa * np.log(fsa)
            sum_s_log_s -= fsb * np.log(fsb)
            sum_s_log_s += fnew * np.log(fnew)
            n_clusters -= 1

            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                if sa + sb > largest:
                    largest = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                if sa + sb > largest:
                    largest = sa + sb

        if h == 0:
            typeB = typeB_h0
            h_new = 1
        else:
            typeB = typeB_h1
            h_new = 0

        for ri in range(128):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeB[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h_new * L8 + lex_u

            if parent[u] == -1:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt

            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt

            if ra == rb:
                continue

            sa = size[ra]
            sb = size[rb]
            sum_sq += 2 * np.int64(sa) * np.int64(sb)

            new_size = sa + sb
            fsa = np.float64(sa)
            fsb = np.float64(sb)
            fnew = np.float64(new_size)
            sum_s_log_s -= fsa * np.log(fsa)
            sum_s_log_s -= fsb * np.log(fsb)
            sum_s_log_s += fnew * np.log(fnew)
            n_clusters -= 1

            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                if sa + sb > largest:
                    largest = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                if sa + sb > largest:
                    largest = sa + sb

        if (i + 1) % bin_size == 0:
            idx = (i + 1) // bin_size - 1
            if 0 <= idx < n_bins:
                S_max_bin[idx] = float(largest)
                sum_sq_bin[idx] = float(sum_sq)
                n_clusters_bin[idx] = float(n_clusters)
                sum_s_log_s_bin[idx] = sum_s_log_s

    return S_max_bin, sum_sq_bin, n_clusters_bin, sum_s_log_s_bin


# =====================================================================
# v4.3 kernel with partial-order tracking (NEW)
# =====================================================================

@njit(cache=True, fastmath=True)
def _newman_ziff_kernel_v43_po(L, order, typeA, typeB_h0, typeB_h1, n_bins,
                                snapshot_bond_idx):
    """v4.3 kernel: same as v4.2 + tracks:
       - activation_time[v]: bond-index i when vertex v was first activated (-1 if never)
       - parent_snapshot[v]: copy of parent[v] at bond index snapshot_bond_idx
       The snapshot allows reconstruction of cluster membership at target p
       via union-find on the frozen parent array.
       Returns (S_max, sum_sq, n_clusters, sum_s_log_s, activation_time, parent_snapshot).
    """
    L8 = L ** 8
    N_idx = 2 * L8
    N_valid = order.shape[0]

    parent = np.full(N_idx, -1, dtype=np.int32)
    size = np.zeros(N_idx, dtype=np.int32)

    # NEW: partial-order tracking
    activation_time = np.full(N_idx, -1, dtype=np.int32)
    parent_snapshot = np.full(N_idx, -1, dtype=np.int32)
    snapshot_taken = False

    largest = np.int32(0)
    sum_sq = np.int64(0)
    n_clusters = np.int64(0)
    sum_s_log_s = np.float64(0.0)

    bin_size = max(N_valid // n_bins, 1)
    S_max_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_bin = np.zeros(n_bins, dtype=np.float64)
    n_clusters_bin = np.zeros(n_bins, dtype=np.float64)
    sum_s_log_s_bin = np.zeros(n_bins, dtype=np.float64)

    c = np.empty(8, dtype=np.int64)

    for i in range(N_valid):
        v = order[i]
        # NEW: record activation time
        activation_time[v] = np.int32(i)

        if v >= L8:
            h = 1
            lex_v = v - L8
        else:
            h = 0
            lex_v = v

        x = lex_v
        for k in range(8):
            c[k] = x % L
            x //= L

        parent[v] = v
        size[v] = 1
        sum_sq += 1
        n_clusters += 1
        if largest < 1:
            largest = 1
        current = np.int64(v)

        # ---- Type-A neighbours ----
        for ri in range(112):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeA[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h * L8 + lex_u

            if parent[u] == -1:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt

            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt

            if ra == rb:
                continue

            sa = size[ra]
            sb = size[rb]
            sum_sq += 2 * np.int64(sa) * np.int64(sb)

            new_size = sa + sb
            fsa = np.float64(sa)
            fsb = np.float64(sb)
            fnew = np.float64(new_size)
            sum_s_log_s -= fsa * np.log(fsa)
            sum_s_log_s -= fsb * np.log(fsb)
            sum_s_log_s += fnew * np.log(fnew)
            n_clusters -= 1

            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                if sa + sb > largest:
                    largest = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                if sa + sb > largest:
                    largest = sa + sb

        # ---- Type-B neighbours ----
        if h == 0:
            typeB = typeB_h0
            h_new = 1
        else:
            typeB = typeB_h1
            h_new = 0

        for ri in range(128):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeB[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h_new * L8 + lex_u

            if parent[u] == -1:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt

            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt

            if ra == rb:
                continue

            sa = size[ra]
            sb = size[rb]
            sum_sq += 2 * np.int64(sa) * np.int64(sb)

            new_size = sa + sb
            fsa = np.float64(sa)
            fsb = np.float64(sb)
            fnew = np.float64(new_size)
            sum_s_log_s -= fsa * np.log(fsa)
            sum_s_log_s -= fsb * np.log(fsb)
            sum_s_log_s += fnew * np.log(fnew)
            n_clusters -= 1

            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                if sa + sb > largest:
                    largest = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                if sa + sb > largest:
                    largest = sa + sb

        # NEW: take parent snapshot after bond index `snapshot_bond_idx` has been processed
        if (not snapshot_taken) and i >= snapshot_bond_idx:
            for k in range(N_idx):
                parent_snapshot[k] = parent[k]
            snapshot_taken = True

        # Sample at bin boundary
        if (i + 1) % bin_size == 0:
            idx = (i + 1) // bin_size - 1
            if 0 <= idx < n_bins:
                S_max_bin[idx] = float(largest)
                sum_sq_bin[idx] = float(sum_sq)
                n_clusters_bin[idx] = float(n_clusters)
                sum_s_log_s_bin[idx] = sum_s_log_s

    # Edge case: if snapshot_bond_idx >= N_valid, snapshot is the final state
    if not snapshot_taken:
        for k in range(N_idx):
            parent_snapshot[k] = parent[k]

    return (S_max_bin, sum_sq_bin, n_clusters_bin, sum_s_log_s_bin,
            activation_time, parent_snapshot)


# =====================================================================
# Worker dispatch (chooses kernel based on flag)
# =====================================================================

def _worker(args):
    (L, seed, typeA, typeB_h0, typeB_h1, n_bins,
     track_po, snapshot_bond_idx, save_po_path, trial_idx) = args
    t0 = time.time()
    order = _build_order(L, seed)

    if track_po:
        (S_max, sum_sq, n_clusters, sum_s_log_s,
         activation_time, parent_snapshot) = _newman_ziff_kernel_v43_po(
            L, order, typeA, typeB_h0, typeB_h1, n_bins, snapshot_bond_idx)

        # Save partial-order data for this trial.
        # CRITICAL: filter to vertices activated BY snapshot_bond_idx (not later),
        # since d_BS extraction at p_c only needs the causal set up to snapshot p.
        # This reduces L=12 per-trial size from ~5 GB to ~30 MB (snapshot-only).
        if save_po_path is not None:
            mask_at_snap = (activation_time >= 0) & (activation_time <= snapshot_bond_idx)
            activated_at_snap = np.where(mask_at_snap)[0].astype(np.int64)
            at_compact = activation_time[activated_at_snap].astype(np.int32)
            ps_compact = parent_snapshot[activated_at_snap].astype(np.int32)

            trial_path = save_po_path.replace('.npz', f'_trial{trial_idx:04d}.npz')
            np.savez_compressed(
                trial_path,
                seed=seed,
                snapshot_bond_idx=snapshot_bond_idx,
                activated_vertices=activated_at_snap,
                activation_time=at_compact,
                parent_snapshot=ps_compact,
                n_total_activated_at_snap=np.int64(len(activated_at_snap)),
                L=np.int32(L),
            )
        return (S_max, sum_sq, n_clusters, sum_s_log_s, time.time() - t0,
                True)
    else:
        S_max, sum_sq, n_clusters, sum_s_log_s = _newman_ziff_kernel_v42(
            L, order, typeA, typeB_h0, typeB_h1, n_bins)
        return (S_max, sum_sq, n_clusters, sum_s_log_s, time.time() - t0,
                False)


# =====================================================================
# Analysis (5 estimators, unchanged from v4.2)
# =====================================================================

def analyze_v42(S_max_arr, sum_sq_arr, n_clusters_arr, sum_s_log_s_arr,
                L, n_bins):
    """Compute all 5 p_c estimators (same as v4.2)."""
    N = L ** 8
    bin_size = N // n_bins
    ps = np.arange(1, n_bins + 1) * (bin_size / N)

    S_max_mean = S_max_arr.mean(axis=0)
    sum_sq_mean = sum_sq_arr.mean(axis=0)
    n_clusters_mean = n_clusters_arr.mean(axis=0)
    sum_s_log_s_mean = sum_s_log_s_arr.mean(axis=0)

    n_active = (np.arange(1, n_bins + 1) * bin_size).astype(np.float64)

    denom = n_active - S_max_mean
    chi = np.where(denom > 0, (sum_sq_mean - S_max_mean ** 2) / denom, 0.0)

    dS_max = np.diff(S_max_mean)

    with np.errstate(divide='ignore', invalid='ignore'):
        S1 = np.where(n_active > 0,
                      np.log(n_active) - sum_s_log_s_mean / n_active,
                      0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        S2 = np.where(n_clusters_mean > 0, np.log(n_clusters_mean), 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        S3 = np.where(sum_sq_mean > 0,
                      2 * np.log(n_active) - np.log(sum_sq_mean),
                      0.0)

    pc_chi = float(ps[int(np.argmax(chi))])
    pc_dS_max = float(ps[int(np.argmax(dS_max))])
    pc_S1 = float(ps[int(np.argmax(S1))])
    pc_S2 = float(ps[int(np.argmax(S2))])
    pc_S3 = float(ps[int(np.argmax(S3))])

    return {
        "pc_chi": pc_chi,
        "pc_dS_max": pc_dS_max,
        "pc_S1": pc_S1,
        "pc_S2": pc_S2,
        "pc_S3": pc_S3,
        "chi": chi,
        "S_max_mean": S_max_mean,
        "n_clusters_mean": n_clusters_mean,
        "sum_s_log_s_mean": sum_s_log_s_mean,
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "ps": ps,
    }


# =====================================================================
# Driver (extended for v4.3)
# =====================================================================

def run_single_L(L, trials, trial_start, workers, seed_base, n_bins, output,
                 save_curves, track_po, target_p, save_po_data):
    print("=" * 72)
    print(f"  L = {L}, trials = {trials} (offset {trial_start}), "
          f"workers = {workers}, bins = {n_bins}")
    print(f"  v4.3 kernel: {'PARTIAL-ORDER + 5 estimators' if track_po else '5 estimators only'}")
    if track_po:
        print(f"  Target p for parent snapshot: {target_p:.6f}")
        if save_po_data:
            print(f"  Partial-order data: will save per-trial NPZs alongside output")
    print("=" * 72)

    typeA, typeB_h0, typeB_h1 = make_e8_shifts()
    print(f"  Roots: 112 type-A + 128 type-B = {112 + 128} ok")

    N = L ** 8
    # Estimate memory per worker
    base_mem = (2 * N * 4 * 2 + N * 8) / 1e9  # v4.2 baseline
    po_extra = (2 * (2 * N) * 4) / 1e9 if track_po else 0  # activation_time + parent_snapshot
    mem_per_worker_gb = base_mem + po_extra
    print(f"  N = {N:,} valid vertices (ID space 2N = {2*N:,})")
    print(f"  Memory per worker: ~{mem_per_worker_gb:.2f} GB"
          + (f" (+{po_extra:.1f} GB for partial-order tracking)" if track_po else ""))
    print(f"  Memory total:      ~{workers * mem_per_worker_gb:.2f} GB ({workers} workers)")
    if workers * mem_per_worker_gb > 165:  # 190 GB system, ~25 GB headroom
        rec = max(1, int(160 / mem_per_worker_gb))
        print(f"  WARNING: total memory may exceed 165 GB. "
              f"Consider --workers {rec}")

    # Per-trial seed: seed_base + trial_start + i, for i in [0, trials)
    seeds = [seed_base + trial_start + i for i in range(trials)]

    # Determine snapshot_bond_idx from target_p
    # Order has N elements (one per valid vertex); target_bond_idx = target_p * N_valid
    # Actually N_valid ~ N. Let's use ceil to be safe.
    N_valid_est = N  # approximation; exact value computed in _build_order
    snapshot_bond_idx = int(round(target_p * N_valid_est))
    if track_po:
        print(f"  snapshot_bond_idx = {snapshot_bond_idx:,} "
              f"(target_p={target_p:.6f} * N={N_valid_est:,})")

    # Path for partial-order data
    if track_po and save_po_data:
        save_po_path = output.replace(".json", "_po.npz")  # template
    else:
        save_po_path = None

    # Build work items
    work = [(L, seeds[i], typeA, typeB_h0, typeB_h1, n_bins,
             track_po, snapshot_bond_idx, save_po_path, trial_start + i)
            for i in range(trials)]

    t0 = time.time()
    print(f"\n  Launching {trials} trials across {workers} workers...")
    if HAS_NUMBA:
        print(f"  (First trial per worker incurs ~5-10 s numba JIT cost)")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    ctx = get_context("spawn")
    results = []
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, work)):
            results.append(res)
            elapsed = time.time() - t0
            done = i + 1
            ETA = elapsed * (trials - done) / done if done > 0 else 0
            print(f"    trial {done:>3}/{trials} in {res[4]:6.1f}s  "
                  f"(elapsed {elapsed:.0f}s, ETA {ETA:.0f}s)", flush=True)
    dt = time.time() - t0

    S_max_arr = np.stack([r[0] for r in results])
    sum_sq_arr = np.stack([r[1] for r in results])
    n_clusters_arr = np.stack([r[2] for r in results])
    sum_s_log_s_arr = np.stack([r[3] for r in results])
    times = [r[4] for r in results]

    print(f"\n  Total wall time: {dt:.1f}s ({dt/60:.1f} min, {dt/3600:.2f} h)")
    print(f"  Per-trial: avg {np.mean(times):.1f}s "
          f"(min {min(times):.1f}, max {max(times):.1f})")

    ana = analyze_v42(S_max_arr, sum_sq_arr, n_clusters_arr, sum_s_log_s_arr,
                      L, n_bins)

    bethe = 1.0 / 239
    fw = 1.0 / 183
    pc_list = [ana['pc_chi'], ana['pc_dS_max'], ana['pc_S1'], ana['pc_S2'], ana['pc_S3']]
    spread_pct = (max(pc_list) - min(pc_list)) / np.mean(pc_list) * 100

    print()
    print(f"  Results L={L}, N={N}:")
    print(f"    p_c estimators (and 1/p_c):")
    print(f"      chi peak              = {ana['pc_chi']:.6f}   1/p_c = {1/ana['pc_chi']:.2f}")
    print(f"      dS_max peak           = {ana['pc_dS_max']:.6f}   1/p_c = {1/ana['pc_dS_max']:.2f}")
    print(f"      S1 (Shannon-site)     = {ana['pc_S1']:.6f}   1/p_c = {1/ana['pc_S1']:.2f}")
    print(f"      S2 (Shannon-count)    = {ana['pc_S2']:.6f}   1/p_c = {1/ana['pc_S2']:.2f}")
    print(f"      S3 (Renyi-2)          = {ana['pc_S3']:.6f}   1/p_c = {1/ana['pc_S3']:.2f}")
    print(f"    estimator spread      = {spread_pct:.2f}%")
    print(f"    bethe baseline        = {bethe:.6f}   (1/(z-1) = 1/239)")
    print(f"    target (1/183)        = {fw:.6f}")
    if track_po:
        print(f"    PARTIAL ORDER: snapshot taken at bond {snapshot_bond_idx:,}")
        if save_po_data:
            n_saved = sum(1 for r in results if r[5])
            print(f"    Per-trial NPZ files saved: {n_saved}")

    # Build output JSON
    output_data = {
        "version": "v4.3_partial_order",
        "L": L,
        "N": int(N),
        "avg_degree": 240.0,
        "trials": trials,
        "trial_start": trial_start,
        "seed_base": seed_base,
        "seeds_used": [int(s) for s in seeds],
        "workers": workers,
        "track_partial_order": track_po,
        "target_p": float(target_p) if track_po else None,
        "snapshot_bond_idx": int(snapshot_bond_idx) if track_po else None,
        "pc_chi": float(ana['pc_chi']),
        "pc_dS_max": float(ana['pc_dS_max']),
        "pc_S1": float(ana['pc_S1']),
        "pc_S2": float(ana['pc_S2']),
        "pc_S3": float(ana['pc_S3']),
        "estimator_spread_pct": float(spread_pct),
        "bethe": float(bethe),
        "target": float(fw),
        "n_bins": n_bins,
        "total_time_s": float(dt),
        "per_trial_times_s": [float(t) for t in times],
        "method": "implicit_adjacency_newman_ziff_v43",
        "partial_order_data_saved": (save_po_path is not None),
        "partial_order_npz_template": save_po_path,
    }

    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved summary JSON to {output}")

    if save_curves:
        curves_path = output.replace('.json', '_curves.npz')
        np.savez_compressed(
            curves_path,
            ps=ana['ps'],
            chi=ana['chi'],
            S_max_mean=ana['S_max_mean'],
            n_clusters_mean=ana['n_clusters_mean'],
            sum_s_log_s_mean=ana['sum_s_log_s_mean'],
            S1=ana['S1'],
            S2=ana['S2'],
            S3=ana['S3'],
            S_max_all=S_max_arr,
            sum_sq_all=sum_sq_arr,
            n_clusters_all=n_clusters_arr,
            sum_s_log_s_all=sum_s_log_s_arr,
        )
        print(f"  Saved curves to {curves_path}")

    return output_data


# =====================================================================
# Main entry point
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="E_8 percolation with implicit adjacency v4.3: "
                    "5 estimators + optional partial-order tracking for d_BS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--L", type=int, nargs="+", default=[4])
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--trial-start", type=int, default=0,
                    help="Offset for trial seed indexing (for distributed runs)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed; per-trial seed = seed + trial-start + i")
    ap.add_argument("--n-bins", type=int, default=10000)
    ap.add_argument("--output", type=str, default="e8_percolation_v43_results.json")
    ap.add_argument("--save-curves", action="store_true")
    ap.add_argument("--track-partial-order", action="store_true",
                    help="Enable activation-time tracking + parent snapshot at target-p")
    ap.add_argument("--target-p", type=float, default=0.005800,
                    help="p value where parent snapshot is taken "
                         "(default = chi peak from L=12 v4.2 run)")
    ap.add_argument("--save-partial-order-data", action="store_true",
                    help="Save per-trial NPZ with activation_time + parent_snapshot")
    args = ap.parse_args()

    workers = args.workers if args.workers else min(args.trials, cpu_count())

    print(f"\nPlatform: {platform.system()} ({platform.platform()})")
    print(f"CPUs detected: {cpu_count()}  |  Workers used: {workers}")
    print(f"Numba: {'available' if HAS_NUMBA else 'NOT AVAILABLE'}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"v4.3 partial-order support: "
          f"{'ENABLED' if args.track_partial_order else 'disabled (v4.2 behaviour)'}")

    all_results = []
    for L in args.L:
        out = args.output.replace(".json", f"_L{L}.json") if len(args.L) > 1 else args.output
        r = run_single_L(L, args.trials, args.trial_start, workers, args.seed,
                         args.n_bins, out, args.save_curves,
                         args.track_partial_order, args.target_p,
                         args.save_partial_order_data)
        all_results.append(r)

    # FSS summary unchanged from v4.2 if multiple L given
    if len(all_results) >= 2:
        print("\n" + "=" * 72)
        print("FINITE-SIZE SCALING SUMMARY (5 estimators)")
        print("=" * 72)
        print(f"\n  {'L':>3} {'N':>14} {'1/pc_chi':>10} {'1/pc_dS':>10} "
              f"{'1/pc_S1':>10} {'1/pc_S2':>10} {'1/pc_S3':>10}")
        for r in all_results:
            print(f"  {r['L']:>3} {r['N']:>14,} "
                  f"{1/r['pc_chi']:>10.2f} {1/r['pc_dS_max']:>10.2f} "
                  f"{1/r['pc_S1']:>10.2f} {1/r['pc_S2']:>10.2f} {1/r['pc_S3']:>10.2f}")

        Ns = np.array([r["N"] for r in all_results], dtype=float)
        if len(Ns) >= 2:
            x = Ns ** (-1/3)
            A = np.vstack([x, np.ones(len(x))]).T
            fss_results = {}
            print(f"\n  FSS extrapolation p_c(N) = p_inf + a/N^(1/3):")
            for key, label in [("pc_chi", "chi peak"),
                               ("pc_dS_max", "dS_max"),
                               ("pc_S1", "S1 (Shannon-site)"),
                               ("pc_S2", "S2 (Shannon-count)"),
                               ("pc_S3", "S3 (Renyi-2)")]:
                pcs = np.array([r[key] for r in all_results])
                slope, intercept = np.linalg.lstsq(A, pcs, rcond=None)[0]
                fss_results[f"fss_{key}_infinity"] = float(intercept)
                fss_results[f"fss_{key}_slope"] = float(slope)
                print(f"    {label:>22}: pc(inf) = {intercept:.6f}   "
                      f"1/pc(inf) = {1/intercept:.2f}")
            fname = args.output.replace(".json", "_FSS.json")
            with open(fname, "w") as f:
                json.dump({"all_results": all_results, **fss_results}, f, indent=2)
            print(f"\n  Saved FSS to {fname}")


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()
    main()
