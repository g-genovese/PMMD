"""
E_8 Foam Percolation - Cross-Platform HPC Edition (Windows / Linux / macOS)
============================================================================

Optimised for multi-core workstations (32+ cores, 64+ GB RAM).
Works on Windows, Linux, and macOS thanks to shared_memory (Python 3.8+).

Uses:
  - numba JIT compilation for inner loops (~100x speedup vs pure Python)
  - multiprocessing.shared_memory for cross-platform memory sharing (no fork)
  - flat CSR-like arrays for cache efficiency

INSTALLATION (Windows PowerShell or cmd):
    pip install numpy numba

USAGE:
    python e8_percolation_winhpc.py --L 4 --trials 32 --workers 32
    python e8_percolation_winhpc.py --L 3 4 5 6 --trials 32 --workers 32   (FSS)

ESTIMATED RUNTIMES on 32-core, 64 GB workstation (Windows or Linux):
    L=3,  N=6,562    : ~5 seconds   (mostly numba JIT compile)
    L=4,  N=65,536   : ~20 seconds  (incl. JIT compile)
    L=5,  N=390,625  : ~3 minutes
    L=6,  N=1,679,616: ~20 minutes
    L=7,  N=5.76M    : ~2-3 hours
    L=8,  N=16.8M    : 8+ hours (needs >100 GB)

For paper target (p_c ≈ 1/183), recommended sequence on your machine:
    python e8_percolation_winhpc.py --L 3 4 5 6 --trials 32 --workers 32

This runs full FSS in ~30 minutes total and extrapolates p_c(L→∞).

REQUIREMENTS:
    Python 3.8+ (for shared_memory)
    numpy
    numba (strongly recommended; without it, 50-100x slower)

WINDOWS NOTES:
  - Run from cmd / PowerShell, not from inside an IDE that swallows stdout
  - If you see "Conda environment errors" or import failures, try:
        py -3 -m pip install numpy numba
  - Antivirus may slow down numba compilation cache; consider whitelisting
    the script directory.
"""

import argparse
import time
import sys
import os
import json
import platform
import multiprocessing as mp
from multiprocessing import shared_memory
from itertools import product
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: numba not installed. Will be 50-100x slower.")
    print("         Install with: pip install numba")
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator


# ===========================================================================
# E_8 LATTICE GENERATION
# ===========================================================================

def generate_E8_roots():
    """(240, 8) array of E_8 root vectors."""
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
    """E_8 torus vertices in doubled-integer coords."""
    pts = []
    for coords in product(range(L), repeat=8):
        if sum(coords) % 2 == 0:
            pts.append([2*c for c in coords])
    for coords in product(range(L), repeat=8):
        if sum(coords) % 2 == 0:
            pts.append([2*c + 1 for c in coords])
    return np.array(pts, dtype=np.int32)


def build_adjacency_csr(points, roots, L):
    """Build CSR adjacency. Returns (indptr, indices, avg_deg)."""
    N = len(points)
    mod = 2 * L
    
    point_to_idx = {}
    for i in range(N):
        key = tuple(int(c) for c in points[i])
        point_to_idx[key] = i
    
    roots_doubled = (2 * roots).astype(np.int32)
    neighbor_buf = [[] for _ in range(N)]
    
    for i in range(N):
        p = points[i]
        for r in roots_doubled:
            key = tuple(int((p[k] + r[k]) % mod) for k in range(8))
            j = point_to_idx.get(key)
            if j is not None and j != i:
                neighbor_buf[i].append(j)
    
    degrees = np.array([len(nb) for nb in neighbor_buf], dtype=np.int64)
    indptr = np.zeros(N + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(degrees)
    
    indices = np.zeros(int(indptr[-1]), dtype=np.int32)
    for i in range(N):
        indices[indptr[i]:indptr[i+1]] = neighbor_buf[i]
    
    return indptr, indices, float(indptr[-1] / N)


# ===========================================================================
# NUMBA KERNEL
# ===========================================================================

@njit(cache=True)
def _newman_ziff_kernel(N, indptr, indices, order):
    """JIT-compiled Newman-Ziff core. Returns S_max, chi arrays of length N."""
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
        
        for idx in range(indptr[v], indptr[v+1]):
            u = indices[idx]
            if active[u]:
                ru = u
                while parent[ru] != ru:
                    ru = parent[ru]
                rv = v
                while parent[rv] != rv:
                    rv = parent[rv]
                
                if ru != rv:
                    su = size[ru]
                    sv = size[rv]
                    sum_s2 -= su * su + sv * sv
                    if sv < su:
                        parent[rv] = ru
                        size[ru] = su + sv
                        new_size = size[ru]
                    else:
                        parent[ru] = rv
                        size[rv] = su + sv
                        new_size = size[rv]
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


# ===========================================================================
# WORKER FUNCTION (uses shared memory)
# ===========================================================================
# Args passed via pickle: shared memory names + shape/dtype info + seed.
# Workers attach to shared memory, create numpy views, run trial, return arrays.

def _worker_run_shm(args):
    """Worker: attach to shared memory, run one trial.
    
    args = (N, indptr_shm_name, indptr_shape, indices_shm_name, indices_shape, seed)
    """
    N, indptr_name, indptr_shape, indices_name, indices_shape, seed = args
    
    # Attach to shared memory (does NOT copy the buffer)
    shm_indptr = shared_memory.SharedMemory(name=indptr_name)
    shm_indices = shared_memory.SharedMemory(name=indices_name)
    try:
        indptr  = np.ndarray(indptr_shape,  dtype=np.int64, buffer=shm_indptr.buf)
        indices = np.ndarray(indices_shape, dtype=np.int32, buffer=shm_indices.buf)
        
        rng = np.random.RandomState(seed)
        order = rng.permutation(N).astype(np.int64)
        S_max, chi = _newman_ziff_kernel(N, indptr, indices, order)
        
        # Return copies (the shared memory disappears once we close it)
        return S_max.copy(), chi.copy()
    finally:
        shm_indptr.close()
        shm_indices.close()


def run_percolation_parallel(N, indptr, indices, n_trials, n_workers, seed=42):
    """Run trials in parallel using shared memory (cross-platform)."""
    
    if n_workers <= 1 or n_trials == 1:
        # Serial path
        S_avg = np.zeros(N, dtype=np.float64)
        chi_avg = np.zeros(N, dtype=np.float64)
        for t in range(n_trials):
            t0 = time.time()
            rng = np.random.RandomState(seed + t)
            order = rng.permutation(N).astype(np.int64)
            S_max, chi = _newman_ziff_kernel(N, indptr, indices, order)
            S_avg += S_max
            chi_avg += chi
            print(f"  Trial {t+1}/{n_trials} ({time.time()-t0:.1f}s)")
        return S_avg / n_trials, chi_avg / n_trials
    
    # Parallel path with shared memory
    print(f"  Allocating shared memory...")
    t0 = time.time()
    
    # Create shared memory blocks and copy adjacency in
    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    indptr_view = np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf)
    indptr_view[:] = indptr[:]
    
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)
    indices_view = np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf)
    indices_view[:] = indices[:]
    
    print(f"  Shared memory allocated in {time.time()-t0:.1f}s "
          f"({(indptr.nbytes + indices.nbytes)/2**30:.2f} GB)")
    
    try:
        args_list = [
            (N, shm_indptr.name, indptr.shape, shm_indices.name, indices.shape, seed + t)
            for t in range(n_trials)
        ]
        
        # Use spawn context (default on Windows; explicit for clarity)
        ctx = mp.get_context('spawn')
        
        t0 = time.time()
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_run_shm, args_list)
        print(f"  All {n_trials} trials done in {time.time()-t0:.1f}s "
              f"on {n_workers} workers")
        
        S_avg = np.zeros(N, dtype=np.float64)
        chi_avg = np.zeros(N, dtype=np.float64)
        for S, chi in results:
            S_avg += S
            chi_avg += chi
        return S_avg / n_trials, chi_avg / n_trials
    finally:
        # Clean up shared memory
        shm_indptr.close()
        shm_indptr.unlink()
        shm_indices.close()
        shm_indices.unlink()


# ===========================================================================
# DRIVER
# ===========================================================================

def estimate_pc(S_avg, chi_avg, N):
    pc_chi = (int(np.argmax(chi_avg)) + 1) / N
    pc_dS  = (int(np.argmax(np.gradient(S_avg))) + 1) / N
    return pc_chi, pc_dS


def fss_extrapolation(results):
    """Fit p_c(N) = p_inf + a/N^(1/3). Returns (p_inf, a) or None."""
    if len(results) < 2:
        return None
    Ns  = np.array([r['N'] for r in results], dtype=float)
    pcs = np.array([r['pc_chi'] for r in results])
    x = Ns**(-1/3)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, pcs, rcond=None)[0]
    return float(intercept), float(slope)


def run_single_L(L, trials, workers, seed, save_curves=False, output=None):
    print("=" * 70)
    print(f"  L = {L},  trials = {trials},  workers = {workers}")
    print("=" * 70)
    
    t0 = time.time()
    roots = generate_E8_roots()
    points = generate_E8_torus_points(L)
    N = len(points)
    print(f"  N = {N} vertices, {time.time()-t0:.1f}s")
    
    mem_adj_gb = (N * 240 * 4 + N * 8) / 2**30
    print(f"  Memory: adjacency ~{mem_adj_gb:.2f} GB (shared across workers)")
    
    t0 = time.time()
    indptr, indices, avg_deg = build_adjacency_csr(points, roots, L)
    print(f"  Adjacency built in {time.time()-t0:.1f}s, avg degree {avg_deg:.2f}")
    
    if not HAS_NUMBA:
        print("  ! No numba: each trial will be very slow.")
    
    print(f"  Running {trials} trials on {workers} workers...")
    S_avg, chi_avg = run_percolation_parallel(N, indptr, indices, trials, workers, seed)
    
    pc_chi, pc_dS = estimate_pc(S_avg, chi_avg, N)
    bethe = 1.0 / 239
    fw = 1.0 / 183
    
    print(f"\n  Results L={L}, N={N}:")
    print(f"    chi-peak p_c = {pc_chi:.6f} = 1/{1/pc_chi:.1f}")
    print(f"    dS-peak  p_c = {pc_dS:.6f} = 1/{1/pc_dS:.1f}")
    print(f"    Bethe 1/239: {bethe:.6f}")
    print(f"    Target 1/183: {fw:.6f}")
    print(f"    ratio chi/target: {pc_chi/fw:.3f}")
    
    result = {
        'L': L, 'N': int(N),
        'avg_degree': avg_deg,
        'trials': trials,
        'pc_chi': float(pc_chi),
        'pc_dS': float(pc_dS),
        'bethe': bethe,
        'target': fw,
    }
    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
    if save_curves and output:
        np.save(output.replace('.json', '_S.npy'), S_avg)
        np.save(output.replace('.json', '_chi.npy'), chi_avg)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="E_8 percolation HPC (cross-platform, Windows-compatible)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--L', type=int, nargs='+', default=[4])
    parser.add_argument('--trials', type=int, default=32)
    parser.add_argument('--workers', type=int, default=min(32, mp.cpu_count()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='e8_percolation_results.json')
    parser.add_argument('--save-curves', action='store_true')
    args = parser.parse_args()
    
    print(f"\nPlatform: {platform.system()} ({platform.platform()})")
    print(f"CPUs detected: {mp.cpu_count()}  |  Workers: {args.workers}")
    print(f"Numba: {'available' if HAS_NUMBA else 'NOT AVAILABLE'}")
    if not HAS_NUMBA:
        print("       Install with: pip install numba (or py -3 -m pip install numba)")
    print(f"Python: {sys.version.split()[0]}\n")
    
    all_results = []
    for L in args.L:
        out_file = args.output.replace('.json', f'_L{L}.json') if len(args.L) > 1 else args.output
        result = run_single_L(L, args.trials, args.workers, args.seed,
                              save_curves=args.save_curves, output=out_file)
        all_results.append(result)
    
    if len(all_results) >= 2:
        print("\n" + "=" * 70)
        print("FINITE-SIZE SCALING")
        print("=" * 70)
        print(f"\n  {'L':>3} {'N':>12} {'pc_chi':>12} {'1/pc':>8} {'ratio':>8}")
        for r in all_results:
            print(f"  {r['L']:>3} {r['N']:>12} {r['pc_chi']:>12.6f} "
                  f"{1/r['pc_chi']:>8.1f} {r['pc_chi']/r['target']:>8.3f}")
        fit = fss_extrapolation(all_results)
        if fit:
            pc_inf, slope = fit
            print(f"\n  FSS fit: p_c(N) = p_inf + a/N^(1/3)")
            print(f"    p_c(inf) = {pc_inf:.6f} = 1/{1/pc_inf:.1f}")
            print(f"    Ratio to target 1/183: {pc_inf*183:.3f}")
        
        with open(args.output.replace('.json', '_FSS.json'), 'w') as f:
            json.dump({
                'all_results': all_results,
                'fss_pc_infinity': pc_inf if fit else None,
                'fss_slope': slope if fit else None,
            }, f, indent=2)
        print(f"\n  Saved to {args.output.replace('.json', '_FSS.json')}")


# CRITICAL on Windows: must protect with __main__ guard!
if __name__ == "__main__":
    # On Windows we need freeze_support for frozen executables (no-op otherwise)
    mp.freeze_support()
    main()
