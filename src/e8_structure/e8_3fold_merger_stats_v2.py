"""
e8_3fold_merger_stats_v2.py
============================
FAST k-fold cluster merger statistics during E_8 site percolation.

Implements the FRAMEWORK-CORRECT picture (Giuseppe's clarification, May 2026):

  * ONE foam, bi-temporal. The Σ+/Σ- labels are PROJECTIONS of one substrate
    field, not two competing foams. There is NO annihilation at the substrate
    level. Phases enter into RELATION (averaging), not destruction.
  * Each cluster has both temporal orientations internally. A vertex's
    instantaneous phase decides which projection it contributes to, but the
    cluster as a whole is gravitationally integral.
  * The matter/antimatter asymmetry η_B is the cumulative outcome of phase-
    distribution biases at 3-fold cluster mergers, observed via the
    cluster's signed orientation balance — NOT via two foams annihilating.

This script therefore performs STANDARD single-foam Newman-Ziff percolation
on the E_8 torus, with each vertex decorated by a random ±1 phase. At every
k-fold cluster merger (k ≥ 2 clusters joining via a new vertex), the
orientation pattern of the merging clusters is recorded.

Tests the v4.0 paper's fractal 3-fold recursion prediction (Section 17):
  - At percolation criticality, 3-fold cluster mergers occur with measurable
    frequency
  - Their orientation-pattern distribution (3:0 vs 2:1) should match random
    binomial (1/4 vs 3/4 under exact Z/2 symmetry)
  - The cumulative orientation asymmetry of the percolating cluster traces
    a random walk → final ratio σ ∼ 1/√N_3fold

DROP-IN ALGORITHMIC PARALLEL to e8_percolation_implicit.py: same vertex IDs,
same E_8 root translations, same modular torus, numba-accelerated.

Memory per worker (L=10):
  parent[2*L^8] int32     = 800 MB
  size[2*L^8]   int32     = 800 MB
  orient_sum[2*L^8] int32 = 800 MB
  order[L^8]   int64      = 800 MB
  colors[L^8]  int8       = 100 MB
  Total: ~3.3 GB per worker.

USAGE:
  python e8_3fold_merger_stats_v2.py --L 4 --trials 16 --workers 4 --output stats.json
  python e8_3fold_merger_stats_v2.py --L 6 --trials 32 --workers 16
  python e8_3fold_merger_stats_v2.py --L 8 --trials 32 --workers 16
"""

import argparse
import json
import sys
import time
import platform
import numpy as np
from collections import Counter
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
# E_8 root shifts (same as e8_percolation_implicit.py)
# =====================================================================

def make_e8_shifts():
    """typeA (112, 8), typeB_h0 (128, 8), typeB_h1 (128, 8) in int8."""
    typeA = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = np.zeros(8, dtype=np.int8)
                    v[i] = s1
                    v[j] = s2
                    typeA.append(v)
    typeA = np.stack(typeA)
    assert typeA.shape == (112, 8)

    typeB_h0 = []
    typeB_h1 = []
    for mask in range(256):
        if bin(mask).count("1") % 2 != 0:
            continue
        s = np.array([-1 if (mask >> k) & 1 else 1 for k in range(8)],
                     dtype=np.int8)
        d0 = ((s - 1) // 2).astype(np.int8)  # {-1, 0}
        d1 = ((s + 1) // 2).astype(np.int8)  # {0, 1}
        typeB_h0.append(d0)
        typeB_h1.append(d1)
    typeB_h0 = np.stack(typeB_h0)
    typeB_h1 = np.stack(typeB_h1)
    return typeA, typeB_h0, typeB_h1


@njit(cache=True)
def make_valid_order(L, seed):
    """Random permutation of all valid vertex IDs (h, c) with sum(c) even."""
    L8 = L ** 8
    np.random.seed(seed)
    valid_count = 0
    valid = np.empty(2 * L8, dtype=np.int64)
    c = np.empty(8, dtype=np.int64)
    for v in range(2 * L8):
        if v >= L8:
            h = 1
            lex_v = v - L8
        else:
            h = 0
            lex_v = v
        x = lex_v
        s = 0
        for k in range(8):
            c[k] = x % L
            x //= L
            s += c[k]
        if s % 2 == 0:
            valid[valid_count] = v
            valid_count += 1
    valid = valid[:valid_count]
    perm = np.random.permutation(valid_count)
    order = valid[perm]
    return order


# =====================================================================
# Main trial: single-foam percolation + orientation decoration
# =====================================================================

@njit(cache=True)
def trial_3fold(L, order, typeA, typeB_h0, typeB_h1, n_bins, orient_seed,
                max_kfold, m_bins, size_bins_log):
    """
    Newman-Ziff percolation with phase orientation tracking.

    For every vertex activation, count the number of distinct neighbour
    cluster roots (k). If k >= 2, this is a k-fold cluster merger event.
    Record the orientation pattern (sign of orient_sum per merging cluster)
    for events with k in [2, max_kfold].

    Outputs (per bin, n_bins bins):
      S_max_bin[b]        : largest cluster size at end of bin b
      sum_sq_bin[b]       : sum of |C|^2 over all clusters at end of bin b
      orient_max_bin[b]   : signed orient_sum of the largest cluster
      n_kfold_bin[b, k]   : cumulative count of k-fold mergers (k=0..max_kfold)
      n_3fold_pattern_bin[b, p] : count of 3-fold mergers by pattern p,
                                  where p in {0=(3,0), 1=(2,1), 2=(1,2), 3=(0,3)}
                                  (n_plus, n_minus) of the three merging
                                  clusters' majority signs.
                                  Mergers with a zero-balance cluster
                                  (orient_sum==0) are NOT counted in any pattern.
      m_size_hist_3fold_bin[b, mm, ss] : 2D histogram of per-cluster
                                  imbalance m = orient_sum/size (binned to
                                  m_bins on [-1, +1]) and cluster size (binned
                                  log2 to size_bins_log bins, bin ss covers
                                  size in [2^ss, 2^(ss+1)) ). Cumulative.
                                  Each 3-fold event increments 3 entries.
    """
    L8 = L ** 8
    N_total = 2 * L8
    N_valid = order.shape[0]
    bin_size = max(N_valid // n_bins, 1)

    parent = np.full(N_total, -1, dtype=np.int32)
    size = np.zeros(N_total, dtype=np.int32)
    orient_sum = np.zeros(N_total, dtype=np.int32)

    np.random.seed(orient_seed)
    orientations = np.where(np.random.random(N_valid) < 0.5, 1, -1).astype(np.int8)

    S_max_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_bin = np.zeros(n_bins, dtype=np.float64)
    orient_max_bin = np.zeros(n_bins, dtype=np.float64)
    n_kfold_bin = np.zeros((n_bins, max_kfold + 1), dtype=np.float64)
    n_3fold_pattern_bin = np.zeros((n_bins, 4), dtype=np.float64)
    # patterns: 0=(3,0), 1=(2,1), 2=(1,2), 3=(0,3)
    n_3fold_with_zero_bin = np.zeros(n_bins, dtype=np.float64)
    # count of 3-fold mergers where at least one cluster has orient_sum==0

    # 2D histogram of (cluster m = orient_sum/size, log2 cluster size)
    # for each cluster participating in a 3-fold event (3 increments per event)
    m_size_hist_3fold_bin = np.zeros((n_bins, m_bins, size_bins_log),
                                     dtype=np.float64)

    # Cumulative event counters (will be sampled at bin boundaries)
    n_kfold_cum = np.zeros(max_kfold + 1, dtype=np.int64)
    n_3fold_pattern_cum = np.zeros(4, dtype=np.int64)
    n_3fold_with_zero_cum = np.int64(0)
    m_size_hist_3fold_cum = np.zeros((m_bins, size_bins_log), dtype=np.int64)

    # Running stats
    largest = np.int32(0)
    sum_sq = np.int64(0)
    largest_root = np.int32(-1)

    c = np.empty(8, dtype=np.int64)
    # Buffers for unique neighbour cluster roots (at most 240 different)
    unique_roots = np.empty(240, dtype=np.int64)
    unique_orient_signs = np.empty(240, dtype=np.int8)  # sign of orient_sum
    unique_orient_sums = np.empty(240, dtype=np.int32)  # raw signed sum
    unique_sizes_local = np.empty(240, dtype=np.int32)  # cluster size at scan time

    for i in range(N_valid):
        v = order[i]
        o_v = orientations[i]

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

        # ===== PASS 1: collect unique active neighbour cluster roots =====
        n_unique = 0
        # Type-A
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
            # find root with path compression
            r = u
            while parent[r] != r:
                r = parent[r]
            x2 = u
            while parent[x2] != r:
                nxt = parent[x2]
                parent[x2] = r
                x2 = nxt
            # deduplicate
            already = False
            for j in range(n_unique):
                if unique_roots[j] == r:
                    already = True
                    break
            if not already:
                unique_roots[n_unique] = r
                osr = orient_sum[r]
                unique_orient_signs[n_unique] = (
                    1 if osr > 0 else (-1 if osr < 0 else 0)
                )
                unique_orient_sums[n_unique] = osr
                unique_sizes_local[n_unique] = size[r]
                n_unique += 1

        # Type-B
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
            r = u
            while parent[r] != r:
                r = parent[r]
            x2 = u
            while parent[x2] != r:
                nxt = parent[x2]
                parent[x2] = r
                x2 = nxt
            already = False
            for j in range(n_unique):
                if unique_roots[j] == r:
                    already = True
                    break
            if not already:
                unique_roots[n_unique] = r
                osr = orient_sum[r]
                unique_orient_signs[n_unique] = (
                    1 if osr > 0 else (-1 if osr < 0 else 0)
                )
                unique_orient_sums[n_unique] = osr
                unique_sizes_local[n_unique] = size[r]
                n_unique += 1

        # ===== RECORD k-fold event =====
        k = n_unique
        if k <= max_kfold:
            n_kfold_cum[k] += 1
        if k == 3:
            # Count (n_plus, n_minus) pattern over the 3 cluster majority signs
            n_p = 0
            n_m = 0
            n_z = 0
            for j in range(3):
                s = unique_orient_signs[j]
                if s > 0:
                    n_p += 1
                elif s < 0:
                    n_m += 1
                else:
                    n_z += 1
            if n_z == 0:
                # patterns: (3,0)→0, (2,1)→1, (1,2)→2, (0,3)→3
                if n_p == 3:
                    n_3fold_pattern_cum[0] += 1
                elif n_p == 2:
                    n_3fold_pattern_cum[1] += 1
                elif n_p == 1:
                    n_3fold_pattern_cum[2] += 1
                else:
                    n_3fold_pattern_cum[3] += 1
            else:
                n_3fold_with_zero_cum += 1

            # ===== Fine-grained per-cluster (m, log2 size) histogram =====
            # m = orient_sum / size in [-1, +1]; bin index m_bin in [0, m_bins-1].
            # size bin ss covers cluster sizes in [2^ss, 2^(ss+1)); the very
            # last bin (size_bins_log - 1) absorbs all larger sizes.
            for j in range(3):
                os_j = unique_orient_sums[j]
                sz_j = unique_sizes_local[j]
                if sz_j <= 0:
                    continue
                m_val = float(os_j) / float(sz_j)  # in [-1, +1]
                # discretize m
                m_idx = int((m_val + 1.0) * m_bins / 2.0)
                if m_idx < 0:
                    m_idx = 0
                elif m_idx >= m_bins:
                    m_idx = m_bins - 1
                # discretize size logarithmically
                sz_tmp = sz_j
                s_idx = 0
                while sz_tmp >= 2 and s_idx < size_bins_log - 1:
                    sz_tmp >>= 1
                    s_idx += 1
                m_size_hist_3fold_cum[m_idx, s_idx] += 1

        # ===== Activate v and merge with all neighbour roots =====
        parent[v] = v
        size[v] = 1
        orient_sum[v] = o_v
        sum_sq += 1
        # union v with each unique root
        cur_root = np.int64(v)
        if cur_root > largest:
            pass  # not used; we'll track largest after the merge
        for j in range(k):
            r = unique_roots[j]
            # find current root of cur_root (could have been redirected)
            r1 = cur_root
            while parent[r1] != r1:
                r1 = parent[r1]
            r2 = r
            while parent[r2] != r2:
                r2 = parent[r2]
            if r1 == r2:
                continue
            sa = size[r1]
            sb = size[r2]
            delta = 2 * np.int64(sa) * np.int64(sb)
            sum_sq += delta
            if sa < sb:
                parent[r1] = r2
                size[r2] = sa + sb
                orient_sum[r2] += orient_sum[r1]
                cur_root = r2
                new_size = sa + sb
                new_root = r2
            else:
                parent[r2] = r1
                size[r1] = sa + sb
                orient_sum[r1] += orient_sum[r2]
                cur_root = r1
                new_size = sa + sb
                new_root = r1
            if new_size > largest:
                largest = new_size
                largest_root = new_root

        # If no merges happened (k==0), v is its own cluster of size 1
        if k == 0 and largest < 1:
            largest = 1
            largest_root = np.int32(v)

        # ===== Bin update at end-of-bin =====
        if (i + 1) % bin_size == 0:
            bin_idx = (i + 1) // bin_size - 1
            if bin_idx < n_bins:
                S_max_bin[bin_idx] = np.float64(largest)
                sum_sq_bin[bin_idx] = np.float64(sum_sq)
                # need to find the current root of largest_root
                # (in case it's been merged into something bigger; tracked above)
                lr = largest_root
                while lr >= 0 and parent[lr] != lr:
                    lr = parent[lr]
                if lr >= 0:
                    orient_max_bin[bin_idx] = np.float64(orient_sum[lr])
                else:
                    orient_max_bin[bin_idx] = 0.0
                for kk in range(max_kfold + 1):
                    n_kfold_bin[bin_idx, kk] = np.float64(n_kfold_cum[kk])
                for pp in range(4):
                    n_3fold_pattern_bin[bin_idx, pp] = np.float64(n_3fold_pattern_cum[pp])
                n_3fold_with_zero_bin[bin_idx] = np.float64(n_3fold_with_zero_cum)
                # snapshot the 2D (m, log2 size) histogram
                for mm in range(m_bins):
                    for ss in range(size_bins_log):
                        m_size_hist_3fold_bin[bin_idx, mm, ss] = (
                            np.float64(m_size_hist_3fold_cum[mm, ss])
                        )

    return (S_max_bin, sum_sq_bin, orient_max_bin,
            n_kfold_bin, n_3fold_pattern_bin, n_3fold_with_zero_bin,
            m_size_hist_3fold_bin)


# =====================================================================
# Worker
# =====================================================================

def worker(args):
    (L, n_bins, order_seed, orient_seed, max_kfold, m_bins, size_bins_log,
     trial_idx, total) = args
    typeA, typeB_h0, typeB_h1 = make_e8_shifts()
    order = make_valid_order(L, order_seed)
    t0 = time.time()
    result = trial_3fold(L, order, typeA, typeB_h0, typeB_h1,
                          n_bins, orient_seed, max_kfold,
                          m_bins, size_bins_log)
    dt = time.time() - t0
    print(f"[trial {trial_idx + 1}/{total}] L={L} done in {dt:.1f}s", flush=True)
    return [np.asarray(x) for x in result]


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--n-bins", type=int, default=1000)
    ap.add_argument("--max-kfold", type=int, default=8,
                    help="Max k for k-fold merger histogram (default 8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-p", type=float, default=1.0 / 175.0,
                    help="Target p_c for reporting bin (default 1/175)")
    ap.add_argument("--m-bins", type=int, default=11,
                    help="Discretization bins for cluster imbalance m=orient/size in [-1,+1] (default 11)")
    ap.add_argument("--size-bins-log", type=int, default=24,
                    help="Log2 bins for cluster size: bin ss covers [2^ss, 2^(ss+1)). Default 24 (covers up to 2^24=16M).")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    L = args.L
    L8 = L ** 8
    N_valid = L8
    print(f"L={L}, N_valid={N_valid:,}, trials={args.trials}, workers={args.workers}")
    print(f"n_bins={args.n_bins}, max_kfold={args.max_kfold}, target_p={args.target_p:.6f}")
    print(f"m_bins={args.m_bins}, size_bins_log={args.size_bins_log}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Numba: {HAS_NUMBA}")

    rng = np.random.RandomState(args.seed)
    seeds = rng.randint(0, 2 ** 31 - 1, size=(args.trials, 2))
    work = [(L, args.n_bins, int(seeds[k, 0]), int(seeds[k, 1]),
             args.max_kfold, args.m_bins, args.size_bins_log,
             k, args.trials) for k in range(args.trials)]

    ctx = get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        results = pool.map(worker, work)
    dt = time.time() - t0
    print(f"Total time: {dt:.1f}s ({dt / args.trials:.1f}s/trial avg)")

    # Stack per-trial results
    S_max_arr = np.stack([r[0] for r in results])  # (trials, n_bins)
    sum_sq_arr = np.stack([r[1] for r in results])
    orient_max_arr = np.stack([r[2] for r in results])
    n_kfold_arr = np.stack([r[3] for r in results])  # (trials, n_bins, max_kfold+1)
    n_3fold_pattern_arr = np.stack([r[4] for r in results])  # (trials, n_bins, 4)
    n_3fold_with_zero_arr = np.stack([r[5] for r in results])
    m_size_hist_arr = np.stack([r[6] for r in results])  # (trials, n_bins, m_bins, size_bins_log)

    bin_centers = (np.arange(args.n_bins) + 0.5) / args.n_bins

    # ===== Find p_c per trial via chi peak =====
    chi_arr = (sum_sq_arr - S_max_arr ** 2) / N_valid
    chi_mean = chi_arr.mean(axis=0)

    # First-local-max search in p < 0.1
    def first_local_max(chi, n_bins, p_max=0.1):
        idx_max = min(int(p_max * n_bins), n_bins - 1)
        for i in range(1, idx_max):
            if chi[i] > chi[i - 1] and chi[i] >= chi[i + 1]:
                return i, float(chi[i])
        i_arg = int(np.argmax(chi[:idx_max + 1]))
        return i_arg, float(chi[i_arg])

    ipeak_mean, chi_peak_mean = first_local_max(chi_mean, args.n_bins)
    p_c_mean = (ipeak_mean + 0.5) / args.n_bins
    print(f"\np_c (trial-avg chi peak): {p_c_mean:.6f} = 1/{1/p_c_mean:.2f}")
    print(f"  (1-color reference: 1/175 ≈ {1/175:.6f})")

    # Bin index corresponding to target_p
    i_target = int(args.target_p * args.n_bins)
    i_target = max(0, min(args.n_bins - 1, i_target))
    print(f"Reporting at target p={args.target_p:.6f} → bin {i_target}")

    # ===== Aggregate 3-fold pattern counts at target_p =====
    pattern_names = ["3+0-", "2+1-", "1+2-", "0+3-"]
    pat_at_target = n_3fold_pattern_arr[:, i_target, :].sum(axis=0)  # sum over trials
    pat_at_target_mean = n_3fold_pattern_arr[:, i_target, :].mean(axis=0)
    total_3fold_at_target = pat_at_target.sum()
    with_zero_at_target = n_3fold_with_zero_arr[:, i_target].sum()

    print(f"\n===== 3-fold merger orientation patterns at p ≈ {args.target_p:.4f} =====")
    print(f"Total 3-fold events (summed across {args.trials} trials): "
          f"{int(total_3fold_at_target)}  ({total_3fold_at_target/args.trials:.1f} per trial)")
    print(f"Mergers with a zero-balance cluster (excluded): {int(with_zero_at_target)}")
    print(f"\n{'Pattern':>10} {'Observed':>12} {'Frac':>10} {'Predicted':>12}")
    # Framework prediction (random binomial of 3 fair coins, only the 4 nonzero patterns):
    # P(3+0-) = 1/8, P(2+1-) = 3/8, P(1+2-) = 3/8, P(0+3-) = 1/8
    predicted_frac = [1 / 8, 3 / 8, 3 / 8, 1 / 8]
    for j, name in enumerate(pattern_names):
        obs = int(pat_at_target[j])
        frac = obs / total_3fold_at_target if total_3fold_at_target > 0 else 0.0
        print(f"{name:>10} {obs:>12} {frac:>10.4f} {predicted_frac[j]:>12.4f}")

    # Collapsed: unanimous (3+0- or 0+3-) vs split (2+1- or 1+2-)
    unanimous = int(pat_at_target[0] + pat_at_target[3])
    split = int(pat_at_target[1] + pat_at_target[2])
    if total_3fold_at_target > 0:
        unanimous_frac = unanimous / total_3fold_at_target
        split_frac = split / total_3fold_at_target
        print(f"\n  Unanimous (3:0 or 0:3): {unanimous} ({unanimous_frac:.4f}, predicted 0.25)")
        print(f"  Split (2:1 or 1:2):     {split} ({split_frac:.4f}, predicted 0.75)")
        delta = abs(unanimous_frac - 0.25)
        sigma_theoretical = np.sqrt(0.25 * 0.75 / total_3fold_at_target)
        print(f"  Deviation from prediction: {delta:.4f} ({delta/sigma_theoretical:.2f}σ at this stats)")

    # ===== Fine-grained m distribution (at target_p and integrated) =====
    # m_size_hist_arr shape: (trials, n_bins, m_bins, size_bins_log)
    # Sum over trials → (n_bins, m_bins, size_bins_log)
    hist_total = m_size_hist_arr.sum(axis=0)
    # Differential per bin: hist_total is cumulative, so subtract to get per-bin
    # increments. Use the cumulative AT target_p (events that have happened by p).
    hist_at_target = hist_total[i_target]  # (m_bins, size_bins_log)
    hist_final = hist_total[-1]            # cumulative over whole percolation

    print(f"\n===== Fine-grained cluster-imbalance histogram at p ≈ {args.target_p:.4f} =====")
    print(f"(Each 3-fold event contributes 3 cluster-counts.)")
    # Marginal over size
    m_marginal_at_target = hist_at_target.sum(axis=1)
    total_counts = m_marginal_at_target.sum()
    print(f"Total cluster-counts: {int(total_counts)} (≈ 3 × {int(total_counts/3)} 3-fold events)")
    if total_counts > 0:
        # Bin centers for m
        m_centers = -1.0 + 2.0 * (np.arange(args.m_bins) + 0.5) / args.m_bins
        print(f"\n  {'m_center':>10} {'count':>10} {'frac':>10}  (cluster imbalance m = orient_sum/size)")
        for mm in range(args.m_bins):
            cnt = int(m_marginal_at_target[mm])
            frac = cnt / total_counts
            bar = '█' * int(40 * frac)
            print(f"  {m_centers[mm]:>+10.3f} {cnt:>10} {frac:>10.4f}  {bar}")
        # Symmetry check
        # The framework predicts P(m) symmetric around 0 (Z/2 symmetry)
        # Compute mean and std of m from histogram
        mean_m = float(np.sum(m_centers * m_marginal_at_target) / total_counts)
        var_m = float(np.sum((m_centers - mean_m)**2 * m_marginal_at_target) / total_counts)
        print(f"\n  <m> = {mean_m:+.4f}  (predicted 0; if nonzero, Z/2 breaking signal)")
        print(f"  std(m) = {np.sqrt(var_m):.4f}")

    # Imbalance distribution split by cluster size class
    # Pool the size bins into 4 classes for readability:
    #   0: singleton (size=1)             → ss=0
    #   1: tiny       (2≤size<8)         → ss in [1, 2]
    #   2: small      (8≤size<128)       → ss in [3, 6]
    #   3: medium+    (size≥128)         → ss ≥ 7
    print(f"\n===== Imbalance distribution by cluster-size class (at target_p) =====")
    class_ranges = [(0, 0), (1, 2), (3, 6), (7, args.size_bins_log - 1)]
    class_labels = ["size=1", "size 2-7", "size 8-127", "size≥128"]
    for cl_idx, (s_lo, s_hi) in enumerate(class_ranges):
        if s_lo > args.size_bins_log - 1:
            continue
        s_hi = min(s_hi, args.size_bins_log - 1)
        m_dist = hist_at_target[:, s_lo:s_hi + 1].sum(axis=1)
        tot = m_dist.sum()
        if tot == 0:
            continue
        m_centers = -1.0 + 2.0 * (np.arange(args.m_bins) + 0.5) / args.m_bins
        mean_m_class = float(np.sum(m_centers * m_dist) / tot)
        var_m_class = float(np.sum((m_centers - mean_m_class)**2 * m_dist) / tot)
        # Predicted std for ideal CLT: 1/sqrt(<size in class>)
        # Approx <size> = geometric center of class range
        if s_lo == 0:
            pred_std = 1.0
        else:
            size_geom = np.sqrt(2.0**s_lo * 2.0**(s_hi + 1))
            pred_std = 1.0 / np.sqrt(size_geom)
        print(f"  {class_labels[cl_idx]:>14}: n={int(tot)}  <m>={mean_m_class:+.4f}  "
              f"std(m)={np.sqrt(var_m_class):.4f}  (CLT predicts ~{pred_std:.4f})")

    # ===== Final cluster orientation balance =====
    final_S = S_max_arr[:, -1]
    final_orient = orient_max_arr[:, -1]
    final_n_plus = (final_S + final_orient) / 2
    final_n_minus = (final_S - final_orient) / 2
    print(f"\n===== Final percolating cluster orientation balance =====")
    print(f"  S_max:        mean={final_S.mean():,.0f}, std={final_S.std():,.0f}")
    print(f"  orient_sum:   mean={final_orient.mean():.1f}, std={final_orient.std():.1f}")
    print(f"  |orient_sum|/sqrt(S):  mean={np.abs(final_orient).mean() / np.sqrt(final_S.mean()):.4f}")
    print(f"  (expected ~ 1.0 for unbiased random walk in cluster decoration)")
    n_3fold_per_trial = n_3fold_pattern_arr[:, -1, :].sum(axis=1) + n_3fold_with_zero_arr[:, -1]
    if n_3fold_per_trial.mean() > 0:
        print(f"  Total 3-fold events per trial (mean): {n_3fold_per_trial.mean():.0f}")
        print(f"  |orient_sum|/sqrt(N_3fold): mean={np.abs(final_orient).mean() / np.sqrt(n_3fold_per_trial.mean()):.4f}")

    # ===== Full k-fold distribution at end =====
    print(f"\n===== Full k-fold merger distribution (cumulative over percolation) =====")
    print(f"{'k':>4} {'mean count':>14} {'std':>14}")
    n_kfold_final = n_kfold_arr[:, -1, :]
    for kk in range(args.max_kfold + 1):
        m = n_kfold_final[:, kk].mean()
        s = n_kfold_final[:, kk].std()
        print(f"{kk:>4} {m:>14,.1f} {s:>14,.1f}")

    # ===== Save output =====
    output = {
        "L": L,
        "N_valid": N_valid,
        "trials": args.trials,
        "n_bins": args.n_bins,
        "max_kfold": args.max_kfold,
        "target_p": args.target_p,
        "platform": platform.system(),
        "time_s": dt,
        "bin_centers": bin_centers.tolist(),
        "p_c_estimated": float(p_c_mean),
        "chi_peak_estimated": float(chi_peak_mean),
        "S_max_mean": S_max_arr.mean(axis=0).tolist(),
        "S_max_sem": (S_max_arr.std(axis=0, ddof=1) / np.sqrt(args.trials)).tolist() if args.trials > 1 else None,
        "sum_sq_mean": sum_sq_arr.mean(axis=0).tolist(),
        "orient_max_mean": orient_max_arr.mean(axis=0).tolist(),
        "orient_max_std": orient_max_arr.std(axis=0).tolist(),
        "n_kfold_mean": n_kfold_arr.mean(axis=0).tolist(),  # (n_bins, max_kfold+1)
        "n_3fold_pattern_mean": n_3fold_pattern_arr.mean(axis=0).tolist(),
        "n_3fold_with_zero_mean": n_3fold_with_zero_arr.mean(axis=0).tolist(),
        "framework_test_at_target_p": {
            "total_3fold": int(total_3fold_at_target),
            "with_zero_balance_cluster": int(with_zero_at_target),
            "patterns": {pattern_names[j]: int(pat_at_target[j]) for j in range(4)},
            "unanimous_frac": float(unanimous / total_3fold_at_target) if total_3fold_at_target > 0 else None,
            "split_frac": float(split / total_3fold_at_target) if total_3fold_at_target > 0 else None,
        },
        "final_state": {
            "S_max_per_trial": final_S.tolist(),
            "orient_sum_per_trial": final_orient.tolist(),
            "n_plus_per_trial": final_n_plus.tolist(),
            "n_minus_per_trial": final_n_minus.tolist(),
            "n_3fold_per_trial": n_3fold_per_trial.tolist(),
        },
        "m_bins": args.m_bins,
        "size_bins_log": args.size_bins_log,
        "m_size_hist_3fold_at_target_p": hist_at_target.tolist(),
        "m_size_hist_3fold_final": hist_final.tolist(),
        "m_bin_centers": (-1.0 + 2.0 * (np.arange(args.m_bins) + 0.5)
                          / args.m_bins).tolist(),
    }

    out_path = args.output or f"e8_3fold_stats_L{L}.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved to {out_path}")

    npz_path = out_path.replace(".json", "_pertrial.npz")
    np.savez_compressed(
        npz_path,
        bin_centers=bin_centers,
        S_max=S_max_arr,
        sum_sq=sum_sq_arr,
        orient_max=orient_max_arr,
        n_kfold=n_kfold_arr,
        n_3fold_pattern=n_3fold_pattern_arr,
        n_3fold_with_zero=n_3fold_with_zero_arr,
        m_size_hist_3fold=m_size_hist_arr,  # (trials, n_bins, m_bins, size_bins_log)
    )
    print(f"Saved per-trial curves to {npz_path}")


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()
    main()
