"""
compute_dBS.py  (v2 - intrinsic monotone-path partial order)
============================================================
Post-processing analysis for Sim B: extracts d_BS from per-trial partial-order
data produced by v4.3 simulation.

KEY DESIGN DECISION:
The partial order is constructed INTRINSICALLY from the activation-time
ordering and the activated bond graph. No external "time direction" is
chosen; no global Phi alignment is assumed. This is consistent with the
framework's per-vertex tau-bit structure (visible/dark matter share the same
percolating cluster with locally-varying time direction).

PARTIAL ORDER DEFINITION:
    v < w   iff   exists path v = v_0, v_1, ..., v_k = w in the activated
                  bond graph (restricted to vertices in same cluster)
                  with t(v_i) < t(v_{i+1}) for all i

This makes the partial order an intrinsic causal-set structure on the cluster,
emerging from the foam connectivity itself rather than from an externally
chosen time direction. Both visible (tau = +1) and dark matter (tau = -1)
sectors live on the same partial order; tau is a per-vertex decoration
that does not enter the partial order.

TWO d_BS ESTIMATORS:
1. SCALING: d_BS from log<N> vs log(Delta_t) linear fit
   - For each pair (v, w), N = |I(v,w)| (interval cardinality)
   - In d-dim Minkowski Poisson: <N> ~ Delta_t^d
2. CHAIN RATIO (paper formula): d_BS = log2(N_{k+1}/N_k) + 1
   - N_k = number of k-element chains in interval
   - Asymptotic ratio approaches 2^(d-1)

USAGE:
    python compute_dBS.py --po-files 'sim_b_L12_*_po_trial*.npz' \\
        --partial-order framework \\
        --n-pair-samples 5000 \\
        --top-n-clusters 3 \\
        --output dBS_results.json
"""

import argparse
import json
import os
import sys
import time
import glob
from collections import defaultdict, deque
import numpy as np

try:
    from scipy.stats import linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
# E_8 shift tables (identical to v4.3 simulation kernel)
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
    return typeA, typeB_h0, typeB_h1


@njit(cache=True)
def compute_e8_neighbors(v, L, typeA, typeB_h0, typeB_h1, out):
    """Fill out (length 240) with neighbor vertex IDs of v."""
    L8 = L ** 8
    if v >= L8:
        h = 1
        lex_v = v - L8
    else:
        h = 0
        lex_v = v

    c = np.empty(8, dtype=np.int64)
    x = lex_v
    for k in range(8):
        c[k] = x % L
        x //= L

    idx = 0
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
        out[idx] = h * L8 + lex_u
        idx += 1

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
        out[idx] = h_new * L8 + lex_u
        idx += 1


# =====================================================================
# Cluster reconstruction
# =====================================================================

def reconstruct_clusters(trial_data):
    """Identify cluster membership from parent_snapshot."""
    activated = trial_data['activated_vertices']
    ps = trial_data['parent_snapshot']
    parent_map = {int(activated[i]): int(ps[i]) for i in range(len(activated))}

    def find_root(v):
        path = []
        while parent_map.get(v) != v:
            nxt = parent_map.get(v)
            if nxt is None or nxt == v:
                break
            path.append(v)
            v = nxt
        for p in path:
            parent_map[p] = v
        return v

    clusters = defaultdict(list)
    for v in activated:
        r = find_root(int(v))
        clusters[r].append(int(v))
    return dict(clusters)


def build_cluster_adjacency(cluster_members, L, typeA, typeB_h0, typeB_h1,
                             progress=False):
    """For each vertex in cluster, find in-cluster activated neighbors."""
    member_set = set(cluster_members)
    adj = {}
    deg_sum = 0
    out_buf = np.empty(240, dtype=np.int64)
    n = len(cluster_members)
    for i, v in enumerate(cluster_members):
        compute_e8_neighbors(np.int64(v), L, typeA, typeB_h0, typeB_h1, out_buf)
        in_cluster_neighbors = [int(u) for u in out_buf if int(u) in member_set]
        adj[v] = np.array(in_cluster_neighbors, dtype=np.int64)
        deg_sum += len(in_cluster_neighbors)
        if progress and (i + 1) % 5000 == 0:
            print(f"      adjacency: {i+1}/{n} ({(i+1)/n*100:.1f}%)",
                  flush=True)
    return adj, deg_sum / max(1, len(cluster_members))


def monotone_reach_set(start, t_upper, adj, t_dict):
    """BFS from start: vertices reachable via monotone-increasing t path."""
    visited = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        t_u = t_dict[u]
        for u_n in adj.get(u, []):
            u_n = int(u_n)
            if u_n in visited:
                continue
            t_n = t_dict.get(u_n, -1)
            if t_n <= t_u:
                continue
            if t_n > t_upper:
                continue
            visited.add(u_n)
            queue.append(u_n)
    return visited


def monotone_reach_set_reverse(start, t_lower, adj, t_dict):
    """BFS from start going BACKWARDS in t: vertices from which start is reachable."""
    visited = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        t_u = t_dict[u]
        for u_n in adj.get(u, []):
            u_n = int(u_n)
            if u_n in visited:
                continue
            t_n = t_dict.get(u_n, -1)
            if t_n >= t_u:
                continue
            if t_n < t_lower:
                continue
            visited.add(u_n)
            queue.append(u_n)
    return visited


def alexandrov_interval(v, w, adj, t_dict):
    """Compute I(v, w) = {z : v < z < w} via 2-sided BFS."""
    t_v = t_dict[v]
    t_w = t_dict[w]
    if t_v >= t_w:
        return set()
    forward = monotone_reach_set(v, t_w, adj, t_dict)
    if w not in forward:
        return set()
    backward = monotone_reach_set_reverse(w, t_v, adj, t_dict)
    return forward & backward


def estimator_scaling(intervals_data, min_pairs_per_bin=10):
    """Fit log<N> vs log(dt) to extract d_BS."""
    if not intervals_data:
        return None, None, 0
    Ns = np.array([x[0] for x in intervals_data], dtype=np.float64)
    dts = np.array([x[1] for x in intervals_data], dtype=np.float64)
    mask = (Ns > 1) & (dts > 0)
    if mask.sum() < 10:
        return None, None, 0
    Ns, dts = Ns[mask], dts[mask]
    log_dts = np.log(dts)
    log_Ns = np.log(Ns)
    n_bins = max(4, int(np.ceil(np.log2(dts.max() / max(dts.min(), 1)))))
    bin_edges = np.linspace(log_dts.min(), log_dts.max() * 1.001, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        in_bin = (log_dts >= bin_edges[i]) & (log_dts < bin_edges[i+1])
        if in_bin.sum() < min_pairs_per_bin:
            continue
        xs.append(log_dts[in_bin].mean())
        ys.append(log_Ns[in_bin].mean())
    if len(xs) < 3:
        return None, None, 0
    xs, ys = np.array(xs), np.array(ys)
    if HAS_SCIPY:
        slope, _, _, _, stderr = linregress(xs, ys)
        return float(slope), float(stderr), len(xs)
    n = len(xs)
    slope = (n*(xs*ys).sum() - xs.sum()*ys.sum()) / (n*(xs**2).sum() - xs.sum()**2)
    return float(slope), None, n


def count_chains_in_interval(interval_set, adj, t_dict, max_chain_length=6):
    """Count k-chains in interval via DP on reachability matrix."""
    if len(interval_set) < 2:
        return np.zeros(max_chain_length, dtype=np.int64)
    ordered = sorted(interval_set, key=lambda x: t_dict[x])
    N = len(ordered)
    if N > 200:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(N, size=200, replace=False))
        ordered = [ordered[i] for i in idx]
        N = 200
    interval_set_local = set(ordered)

    # Reachability matrix M[i,j] = 1 if ordered[i] precedes ordered[j]
    M = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        u = ordered[i]
        visited = {u}
        queue = deque([u])
        while queue:
            x = queue.popleft()
            for x_n in adj.get(x, []):
                x_n = int(x_n)
                if x_n not in interval_set_local or x_n in visited:
                    continue
                if t_dict[x_n] <= t_dict[x]:
                    continue
                visited.add(x_n)
                queue.append(x_n)
        for j in range(i + 1, N):
            if ordered[j] in visited:
                M[i, j] = True

    # DP: dp[i,k] = #(k-chains ending at ordered[i])
    dp = np.zeros((N, max_chain_length + 1), dtype=np.int64)
    dp[:, 1] = 1
    for k in range(2, max_chain_length + 1):
        for i in range(N):
            s = 0
            for j in range(i):
                if M[j, i]:
                    s += dp[j, k - 1]
            dp[i, k] = s

    return np.array([dp[:, k].sum() for k in range(1, max_chain_length + 1)],
                    dtype=np.int64)


def estimator_chain_ratio(all_chain_counts):
    """d_BS via log2(N_{k+1}/N_k) + 1 from paper."""
    if not all_chain_counts:
        return {}
    arr = np.array(all_chain_counts, dtype=np.float64)
    sums = arr.sum(axis=0)
    estimates = {}
    for k in range(1, sums.shape[0] - 1):
        if sums[k - 1] > 0 and sums[k] > 0:
            ratio = sums[k] / sums[k - 1]
            if ratio > 0:
                estimates[f"d_BS_k{k+1}_over_k{k}"] = float(np.log2(ratio) + 1)
    return estimates


def analyze_cluster_framework(cluster_members, trial_data, L,
                              typeA, typeB_h0, typeB_h1,
                              n_pair_samples=5000, top_pairs_chain=200,
                              max_chain_length=6, rng=None, verbose=False):
    """Compute d_BS estimates for one cluster using monotone-path partial order."""
    if rng is None:
        rng = np.random.default_rng()
    if len(cluster_members) < 50:
        return None
    activated = trial_data['activated_vertices']
    at = trial_data['activation_time']
    member_set = set(cluster_members)
    t_dict = {int(activated[i]): int(at[i]) for i in range(len(activated))
              if int(activated[i]) in member_set}

    t0 = time.time()
    adj, avg_deg = build_cluster_adjacency(cluster_members, L,
                                            typeA, typeB_h0, typeB_h1,
                                            progress=verbose)
    if verbose:
        print(f"    Adjacency built in {time.time()-t0:.1f}s "
              f"(avg in-cluster degree {avg_deg:.1f})")

    sorted_members = sorted(cluster_members, key=lambda x: t_dict[x])
    M = len(sorted_members)
    intervals_for_scaling = []
    chain_counts_per_interval = []
    n_reachable = 0
    pairs_sampled = 0

    t0 = time.time()
    while pairs_sampled < n_pair_samples:
        i = rng.integers(0, M - 1)
        j = rng.integers(i + 1, M)
        v, w = sorted_members[i], sorted_members[j]
        pairs_sampled += 1
        interval = alexandrov_interval(v, w, adj, t_dict)
        if not interval:
            continue
        n_reachable += 1
        N_interval = len(interval)
        dt = t_dict[w] - t_dict[v]
        intervals_for_scaling.append((N_interval, dt))
        if len(chain_counts_per_interval) < top_pairs_chain and N_interval >= 4:
            chain_counts_per_interval.append(
                count_chains_in_interval(interval, adj, t_dict, max_chain_length))
    if verbose:
        print(f"    Sampled {pairs_sampled} pairs: {n_reachable} reachable "
              f"({n_reachable/pairs_sampled*100:.1f}%), "
              f"{len(chain_counts_per_interval)} chain-analyzed "
              f"in {time.time()-t0:.1f}s")

    d_BS_scaling, d_BS_err, n_bins = estimator_scaling(intervals_for_scaling)
    d_BS_chain = estimator_chain_ratio(chain_counts_per_interval)
    return {
        'cluster_size': len(cluster_members),
        'avg_in_cluster_degree': float(avg_deg),
        'n_pairs_sampled': pairs_sampled,
        'n_pairs_reachable': n_reachable,
        'reachability_fraction': n_reachable / max(1, pairs_sampled),
        'd_BS_scaling': d_BS_scaling,
        'd_BS_scaling_stderr': d_BS_err,
        'n_scaling_bins': n_bins,
        'd_BS_chain_estimates': d_BS_chain,
    }


def analyze_cluster_activation_total(cluster_members, trial_data, n_pair_samples=2000,
                                      rng=None):
    """Baseline activation-total partial order: v < w iff t(v) < t(w) (in same cluster).
    Gives degenerate d_BS ~ 1 (totally ordered)."""
    if rng is None:
        rng = np.random.default_rng()
    if len(cluster_members) < 50:
        return None
    activated = trial_data['activated_vertices']
    at = trial_data['activation_time']
    member_set = set(cluster_members)
    t_dict = {int(activated[i]): int(at[i]) for i in range(len(activated))
              if int(activated[i]) in member_set}
    sorted_members = sorted(cluster_members, key=lambda x: t_dict[x])
    M = len(sorted_members)
    intervals_for_scaling = []
    pairs_sampled = 0
    while pairs_sampled < n_pair_samples:
        i = rng.integers(0, M - 1)
        j = rng.integers(i + 1, M)
        v, w = sorted_members[i], sorted_members[j]
        pairs_sampled += 1
        tv, tw = t_dict[v], t_dict[w]
        N_interval = sum(1 for x in sorted_members if tv <= t_dict[x] <= tw)
        dt = tw - tv
        intervals_for_scaling.append((N_interval, dt))
    d_BS_scaling, d_BS_err, n_bins = estimator_scaling(intervals_for_scaling)
    return {
        'cluster_size': len(cluster_members),
        'd_BS_scaling': d_BS_scaling,
        'd_BS_scaling_stderr': d_BS_err,
        'n_scaling_bins': n_bins,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--po-files", type=str, required=True)
    ap.add_argument("--partial-order", type=str, default="framework",
                    choices=["activation-total", "framework", "graph-causal"])
    ap.add_argument("--n-pair-samples", type=int, default=5000)
    ap.add_argument("--top-n-clusters", type=int, default=3)
    ap.add_argument("--top-pairs-chain", type=int, default=200)
    ap.add_argument("--max-chain-length", type=int, default=6)
    ap.add_argument("--output", type=str, default="dBS_results.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(args.po_files))
    if not files:
        print(f"ERROR: no files match '{args.po_files}'")
        sys.exit(1)

    print(f"Found {len(files)} per-trial PO files")
    print(f"Partial-order model: {args.partial_order}")
    if args.partial_order == "activation-total":
        print("  (sanity check: expected d_BS ~ 1)")
    print("=" * 72)

    typeA, typeB_h0, typeB_h1 = make_e8_shifts()
    rng = np.random.default_rng(42)
    all_d_scaling = []
    all_d_chain = defaultdict(list)
    per_trial = []

    for tpath in files:
        t0 = time.time()
        trial = dict(np.load(tpath))
        L = int(trial.get('L', 4))
        clusters = reconstruct_clusters(trial)
        clu_sizes = sorted([(r, len(m)) for r, m in clusters.items()],
                            key=lambda x: -x[1])

        tr = {
            'file': os.path.basename(tpath),
            'L': L,
            'n_clusters': len(clusters),
            'top10_sizes': [s for _, s in clu_sizes[:10]],
            'per_cluster': [],
        }
        for root, _ in clu_sizes[:args.top_n_clusters]:
            members = clusters[root]
            if args.partial_order == "framework" or args.partial_order == "graph-causal":
                r = analyze_cluster_framework(
                    members, trial, L, typeA, typeB_h0, typeB_h1,
                    n_pair_samples=args.n_pair_samples,
                    top_pairs_chain=args.top_pairs_chain,
                    max_chain_length=args.max_chain_length,
                    rng=rng, verbose=args.verbose)
            else:
                r = analyze_cluster_activation_total(
                    members, trial, n_pair_samples=args.n_pair_samples, rng=rng)
            if r:
                tr['per_cluster'].append(r)
                if r.get('d_BS_scaling') is not None:
                    all_d_scaling.append(r['d_BS_scaling'])
                for k, v in r.get('d_BS_chain_estimates', {}).items():
                    all_d_chain[k].append(v)
        per_trial.append(tr)
        d_list = [c.get('d_BS_scaling') for c in tr['per_cluster']]
        print(f"  {tr['file']}: top sizes {tr['top10_sizes'][:3]}, "
              f"d_scaling={d_list}, t={time.time()-t0:.1f}s")

    output = {
        'partial_order_model': args.partial_order,
        'n_trials': len(per_trial),
        'per_trial': per_trial,
    }
    if all_d_scaling:
        arr = np.array(all_d_scaling)
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        output['d_BS_scaling_aggregate'] = {
            'n': len(arr), 'mean': float(arr.mean()),
            'std': float(std), 'sem': float(std/np.sqrt(len(arr))),
            'target': 4.0,
        }
        print(f"\n=== d_BS via SCALING ===")
        print(f"  n={len(arr)} samples")
        print(f"  d_BS = {arr.mean():.3f} ± {std/np.sqrt(len(arr)):.3f}")
        print(f"  Target Bridge Claim 6: 4.0")
    if all_d_chain:
        output['d_BS_chain_aggregate'] = {}
        print(f"\n=== d_BS via CHAIN RATIO (paper formula) ===")
        for k, vals in all_d_chain.items():
            arr = np.array(vals)
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            output['d_BS_chain_aggregate'][k] = {
                'n': len(arr), 'mean': float(arr.mean()),
                'std': float(std), 'sem': float(std/np.sqrt(len(arr))) if len(arr) else None,
            }
            print(f"  {k}: {arr.mean():.3f} ± {std/np.sqrt(len(arr)):.3f}  (n={len(arr)})")

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
