"""
E_8 Foam Percolation - High-Performance Edition
================================================

Optimised for multi-core workstations (32+ cores, 64+ GB RAM).
Uses:
  - numba JIT compilation for inner loops (~100x speedup vs pure Python)
  - multiprocessing with fork-based memory sharing (one adjacency copy)
  - flat CSR-like arrays for cache efficiency

USAGE:
    pip install numpy numba
    python e8_percolation_hpc.py --L 5 --trials 32 --workers 32

ESTIMATED RUNTIMES on 32-core workstation with 64 GB RAM:
    L=3,  N ≈   6.6K  : ~1 second total
    L=4,  N ≈   65K   : ~10 seconds
    L=5,  N ≈  390K   : ~2 minutes        (~3 GB RAM peak)
    L=6,  N ≈  1.68M  : ~10-20 minutes    (~15 GB RAM peak)
    L=7,  N ≈  5.76M  : ~1-2 hours        (~50 GB RAM peak)
    L=8,  N ≈ 16.8M   : 6+ hours          (~140 GB RAM peak; needs more RAM)

For the paper's target precision (p_c ≈ 1/183), recommended:
    L=4 with 32 trials  →  ~10 seconds, modest precision
    L=5 with 32 trials  →  ~2 minutes,  good precision
    L=6 with 32 trials  →  ~15 minutes, publication precision

Then run finite-size scaling fit across L=3,4,5,6.

REQUIREMENTS:
    Python 3.8+
    numpy
    numba (optional but strongly recommended; without numba, 50-100x slower)
"""

import argparse
import time
import sys
import os
import json
import multiprocessing as mp
from itertools import product
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: numba not installed. Performance will be 50-100x slower.")
    print("Install with: pip install numba")
    # Define no-op decorator
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator
    def prange(*args):
        return range(*args)


# ===========================================================================
# E_8 LATTICE GENERATION (NUMPY)
# ===========================================================================

def generate_E8_roots():
    """Return (240, 8) array of E_8 root vectors."""
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for s1, s2 in product([+1, -1], repeat=2):
                r = np.zeros(8)
                r[i] = s1
                r[j] = s2
                roots.append(r)
    for signs in product([+1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            roots.append(0.5 * np.array(signs))
    return np.array(roots, dtype=np.float64)


def generate_E8_torus_points(L):
    """Generate E_8 torus vertices in doubled-integer coordinates.
    
    Returns (N, 8) int32 array where each row is a vertex's coords (doubled,
    so multiply by 2 to get the actual lattice coord; values in [0, 2L)).
    """
    pts = []
    for coords in product(range(L), repeat=8):
        if sum(coords) % 2 == 0:
            # D_8 part: integer coords with even sum
            pts.append([2*c for c in coords])
    for coords in product(range(L), repeat=8):
        if sum(coords) % 2 == 0:
            # (D_8 + half) part: half-integer coords, doubled = odd
            pts.append([2*c + 1 for c in coords])
    return np.array(pts, dtype=np.int32)


def build_adjacency_csr(points, roots, L):
    """Build adjacency in CSR-like format.
    
    Returns:
        indptr : (N+1,) int64; adjacency for vertex i is indices[indptr[i]:indptr[i+1]]
        indices: (E,)   int32; flat list of neighbor indices
        avg_deg: float
    """
    N = len(points)
    mod = 2 * L
    
    # Build hash from doubled coords to vertex index
    # Use tuple keys (fast in CPython)
    point_to_idx = {}
    for i in range(N):
        key = tuple(int(c) for c in points[i])
        point_to_idx[key] = i
    
    # Doubled roots (E_8 roots have entries in {-1, -1/2, 0, 1/2, 1}, so doubled in {-2,-1,0,1,2})
    roots_doubled = (2 * roots).astype(np.int32)
    
    # First pass: count neighbors per vertex
    degrees = np.zeros(N, dtype=np.int32)
    neighbor_buf = [[] for _ in range(N)]
    
    for i in range(N):
        p = points[i]
        for r in roots_doubled:
            key = tuple(int((p[k] + r[k]) % mod) for k in range(8))
            j = point_to_idx.get(key)
            if j is not None and j != i:
                neighbor_buf[i].append(j)
    
    degrees = np.array([len(nb) for nb in neighbor_buf], dtype=np.int32)
    indptr = np.zeros(N + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(degrees)
    
    indices = np.zeros(indptr[-1], dtype=np.int32)
    for i in range(N):
        indices[indptr[i]:indptr[i+1]] = neighbor_buf[i]
    
    avg_deg = float(indptr[-1] / N)
    return indptr, indices, avg_deg


# ===========================================================================
# NEWMAN-ZIFF (NUMBA-JIT)
# ===========================================================================

@njit(cache=True)
def _newman_ziff_kernel(N, indptr, indices, order):
    """JIT-compiled inner loop for Newman-Ziff.
    
    Returns S_max and chi arrays of length N.
    """
    parent = np.arange(N, dtype=np.int64)
    size   = np.ones(N, dtype=np.int64)
    active = np.zeros(N, dtype=np.uint8)
    
    S_max = np.zeros(N, dtype=np.int64)
    chi   = np.zeros(N, dtype=np.float64)
    
    sum_s2 = np.int64(0)
    max_size = np.int64(0)
    
    for k in range(N):
        v = order[k]
        active[v] = 1
        sum_s2 += 1
        cur_size = np.int64(1)
        
        # Find root of v (initially v itself)
        rv = v
        
        # For each neighbor u of v
        for idx in range(indptr[v], indptr[v+1]):
            u = indices[idx]
            if active[u]:
                # Find root of u with path compression
                ru = u
                while parent[ru] != ru:
                    ru = parent[ru]
                # Find current root of v
                rv2 = v
                while parent[rv2] != rv2:
                    rv2 = parent[rv2]
                
                if ru != rv2:
                    su = size[ru]
                    sv = size[rv2]
                    # Remove old contributions to sum_s2
                    sum_s2 -= su * su + sv * sv
                    # Union by size
                    if sv < su:
                        parent[rv2] = ru
                        size[ru] = su + sv
                        new_size = size[ru]
                    else:
                        parent[ru] = rv2
                        size[rv2] = su + sv
                        new_size = size[rv2]
                    sum_s2 += new_size * new_size
                    cur_size = new_size
        
        if cur_size > max_size:
            max_size = cur_size
        S_max[k] = max_size
        
        denom = (k + 1) - max_size
        if denom > 0:
            chi[k] = (sum_s2 - max_size * max_size) / denom
        else:
            chi[k] = 0.0
    
    return S_max, chi


def newman_ziff_trial(N, indptr, indices, seed):
    """Single Newman-Ziff trial with given seed."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(N).astype(np.int64)
    return _newman_ziff_kernel(N, indptr, indices, order)


# ===========================================================================
# MULTIPROCESSING WORKER
# ===========================================================================

# Globals for worker processes (shared via fork)
_WORKER_N = None
_WORKER_INDPTR = None
_WORKER_INDICES = None


def _worker_init(N, indptr, indices):
    """Initialise worker globals; via fork these are shared read-only."""
    global _WORKER_N, _WORKER_INDPTR, _WORKER_INDICES
    _WORKER_N = N
    _WORKER_INDPTR = indptr
    _WORKER_INDICES = indices


def _worker_run(seed):
    """Run a single trial in worker process."""
    S, chi = newman_ziff_trial(_WORKER_N, _WORKER_INDPTR, _WORKER_INDICES, seed)
    return S, chi


def run_percolation_parallel(N, indptr, indices, n_trials, n_workers, seed=42):
    """Run n_trials Newman-Ziff trials across n_workers processes."""
    if n_workers <= 1 or n_trials == 1:
        # Single-process path
        S_avg = np.zeros(N, dtype=np.float64)
        chi_avg = np.zeros(N, dtype=np.float64)
        for t in range(n_trials):
            t0 = time.time()
            S, chi = newman_ziff_trial(N, indptr, indices, seed + t)
            S_avg += S
            chi_avg += chi
            print(f"  Trial {t+1}/{n_trials} done ({time.time()-t0:.1f}s)")
        S_avg /= n_trials
        chi_avg /= n_trials
        return S_avg, chi_avg
    
    # Multi-process: fork shares the adjacency
    ctx = mp.get_context('fork')
    seeds = [seed + i for i in range(n_trials)]
    
    t0 = time.time()
    with ctx.Pool(processes=n_workers,
                  initializer=_worker_init,
                  initargs=(N, indptr, indices)) as pool:
        results = pool.map(_worker_run, seeds)
    
    print(f"  All {n_trials} trials done in {time.time()-t0:.1f}s (parallel on {n_workers} workers)")
    
    S_avg = np.zeros(N, dtype=np.float64)
    chi_avg = np.zeros(N, dtype=np.float64)
    for S, chi in results:
        S_avg += S
        chi_avg += chi
    S_avg /= n_trials
    chi_avg /= n_trials
    return S_avg, chi_avg


# ===========================================================================
# MAIN
# ===========================================================================

def estimate_pc(S_avg, chi_avg, N):
    """Estimate p_c from averaged curves."""
    pc_chi = (int(np.argmax(chi_avg)) + 1) / N
    pc_dS  = (int(np.argmax(np.gradient(S_avg))) + 1) / N
    return pc_chi, pc_dS


def fss_extrapolation(L_pc_list):
    """Finite-size scaling: fit p_c(L) = p_inf + a / L^(d/nu).
    
    For mean-field above critical dimension (d=8 > d_c=6), expected
    correction exponent: 1/nu*d = 1/(8 * 1/2) = 1/4. So fit p_c vs L^(-1/4).
    However simpler ansatz is p_c vs 1/N^alpha = 1/L^(8 alpha).
    Try alpha = 1/3.
    """
    if len(L_pc_list) < 2:
        return None
    Ls = np.array([L for L, _ in L_pc_list])
    pcs = np.array([pc for _, pc in L_pc_list])
    Ns = Ls.astype(float)**8
    x = Ns**(-1/3)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, pcs, rcond=None)[0]
    return float(intercept), float(slope)


def run_single_L(L, trials, workers, seed, save_curves=False, output=None):
    """Run simulation at single L and return results dict."""
    print("=" * 70)
    print(f"E_8 PERCOLATION  L = {L},  trials = {trials},  workers = {workers}")
    print("=" * 70)
    
    print(f"\n[1] Generating roots and lattice points...")
    t0 = time.time()
    roots = generate_E8_roots()
    points = generate_E8_torus_points(L)
    N = len(points)
    print(f"    240 roots, N = {N} vertices  ({time.time()-t0:.1f}s)")
    
    # Memory estimate
    mem_adj_gb = N * 240 * 4 / 2**30
    mem_uf_gb  = workers * N * 24 / 2**30
    print(f"    Memory estimate: adjacency {mem_adj_gb:.1f} GB, "
          f"{workers} workers' UFs {mem_uf_gb:.1f} GB, "
          f"total ~{mem_adj_gb + mem_uf_gb + 2:.1f} GB")
    
    print(f"\n[2] Building CSR adjacency...")
    t0 = time.time()
    indptr, indices, avg_deg = build_adjacency_csr(points, roots, L)
    print(f"    Done ({time.time()-t0:.1f}s)")
    print(f"    Average degree: {avg_deg:.2f}  (expected 240 with PBC)")
    
    if not HAS_NUMBA:
        print("\n  ! WARNING: numba not available. Single trial will take MUCH longer.")
        print("  ! Strongly recommend: pip install numba && re-run.")
    
    print(f"\n[3] Running Newman-Ziff: {trials} trials on {workers} cores...")
    t0 = time.time()
    if HAS_NUMBA:
        print("    First trial includes numba JIT compilation (~5-15s overhead).")
    S_avg, chi_avg = run_percolation_parallel(N, indptr, indices, trials, workers, seed)
    total_time = time.time() - t0
    
    pc_chi, pc_dS = estimate_pc(S_avg, chi_avg, N)
    bethe = 1.0 / 239
    fw_target = 1.0 / 183
    
    print(f"\n[4] Results for L = {L}, N = {N}:")
    print(f"    chi-peak  p_c = {pc_chi:.6f}  (= 1/{1/pc_chi:.1f})")
    print(f"    dS-peak   p_c = {pc_dS:.6f}  (= 1/{1/pc_dS:.1f})")
    print(f"    Bethe theory:  {bethe:.6f}  (= 1/239)")
    print(f"    Framework FW:  {fw_target:.6f}  (= 1/183)")
    print(f"    Ratio chi/FW:  {pc_chi/fw_target:.3f}")
    print(f"    Total time:    {total_time:.1f}s")
    
    result = {
        'L': L, 'N': int(N),
        'avg_degree': avg_deg,
        'trials': trials, 'workers': workers,
        'pc_chi': float(pc_chi),
        'pc_dS':  float(pc_dS),
        'bethe': bethe,
        'framework_target': fw_target,
        'time_seconds': total_time,
    }
    
    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"    Saved JSON to {output}")
    if save_curves and output:
        np.save(output.replace('.json', '_S_avg.npy'), S_avg)
        np.save(output.replace('.json', '_chi_avg.npy'), chi_avg)
        print(f"    Saved curves to {output.replace('.json', '_*.npy')}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="E_8 lattice percolation simulation (high-performance edition).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--L', type=int, nargs='+', default=[4],
                        help="Torus size(s). Provide multiple for FSS extrapolation: --L 3 4 5 6")
    parser.add_argument('--trials', type=int, default=32,
                        help="Trials per L value (parallel across workers).")
    parser.add_argument('--workers', type=int, default=min(32, mp.cpu_count()),
                        help=f"Worker processes (auto = min(32, ncpu) = {min(32, mp.cpu_count())})")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='e8_percolation_results.json')
    parser.add_argument('--save-curves', action='store_true')
    args = parser.parse_args()
    
    print(f"\nMachine: {mp.cpu_count()} CPU cores detected.")
    print(f"Workers: {args.workers} (set with --workers).")
    print(f"Numba JIT: {'available' if HAS_NUMBA else 'NOT AVAILABLE (install with: pip install numba)'}")
    
    all_results = []
    for L in args.L:
        out_file = args.output.replace('.json', f'_L{L}.json') if len(args.L) > 1 else args.output
        result = run_single_L(L, args.trials, args.workers, args.seed,
                              save_curves=args.save_curves, output=out_file)
        all_results.append(result)
    
    # FSS extrapolation if multiple L
    if len(all_results) >= 2:
        print("\n" + "=" * 70)
        print("FINITE-SIZE SCALING EXTRAPOLATION")
        print("=" * 70)
        print(f"\n  {'L':>3}  {'N':>10}  {'p_c (chi)':>12}  {'1/p_c':>8}  {'ratio to 1/183':>15}")
        for r in all_results:
            print(f"  {r['L']:>3}  {r['N']:>10}  {r['pc_chi']:>12.6f}  "
                  f"{1/r['pc_chi']:>8.1f}  {r['pc_chi']/r['framework_target']:>15.3f}")
        
        L_pc = [(r['L'], r['pc_chi']) for r in all_results]
        fit = fss_extrapolation(L_pc)
        if fit:
            pc_inf, slope = fit
            print(f"\n  FSS fit p_c(N) = p_inf + a / N^(1/3):")
            print(f"    p_c(∞) = {pc_inf:.6f} = 1/{1/pc_inf:.1f}")
            print(f"    slope a = {slope:.4f}")
            print(f"    Framework target (1/183) = {1/183:.6f}")
            print(f"    Bethe (1/239) = {1/239:.6f}")
            print(f"    Ratio p_inf/(1/183) = {pc_inf * 183:.3f}")
        
        with open(args.output.replace('.json', '_FSS.json'), 'w') as f:
            json.dump({
                'all_results': all_results,
                'fss_pc_infinity': pc_inf if fit else None,
                'fss_slope': slope if fit else None,
            }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
