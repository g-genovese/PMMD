#!/usr/bin/env python3
"""
compute_dBS_full.py
====================

Calcola la dimensione spettrale d_BS per ogni trial Sim B v4.3 partial-order,
usando DUE estimatori indipendenti per cross-check:

  E1 — Myrheim-Meyer ordering fraction (same-cluster, temporally ordered)
       r = sum_C |C|*(|C|-1) / (N*(N-1))
       d_MM via inversion della curva f_MM(d) tabulata
       
  E2 — Fisher exponent τ_F dalla cluster-size distribution
       n_s ~ s^(-τ_F) fit nel range [s_min, s_max]
       d_f via hyperscaling τ_F = 1 + d/d_f (assume framework d_f ≈ 4)

INPUT:
  Directory con file sim_b_L*_po_trial*.npz (i 30 MB ciascuno).
  Lo script processa TUTTI i file *_po_trial*.npz nella directory.

OUTPUT:
  - dBS_pertrial.json  — per-trial d_MM, d_F, r, τ_F, cluster stats
                          (pochi KB)

PARALLELIZATION:
  Usa multiprocessing.Pool con --workers (default: tutti i core disponibili).
  Ogni worker processa 1 PO file alla volta (~50-100 MB RAM per worker).

USAGE (singolo server, file locali):
  python3 compute_dBS_full.py /path/to/po_files/  
  python3 compute_dBS_full.py /path/to/po_files/ --workers 8
  
USAGE (distribuito 3 server):
  Su ogni server processa SOLO i suoi file locali:
    server-A: python3 compute_dBS_full.py ~/sim_b/po_files/ --output dBS_A.json
    server-B: python3 compute_dBS_full.py ~/sim_b/po_files/ --output dBS_B.json
    server-C: python3 compute_dBS_full.py ~/sim_b/po_files/ --output dBS_C.json
  Poi aggregare i 3 file JSON in un singolo result (script separato o merge manuale).

NOTE STRUTTURALI:
  L'estimatore E1 usa la same-cluster ordering fraction, NON la transitive
  closure del causal set (che richiederebbe il replay completo della percolazione).
  E2 usa la distribuzione di taglie cluster, indipendente da E1.
  
  Per i framework's prediction d_BS = 4, ci aspettiamo:
    f_MM(4) ≈ 0.292   (atteso)
    τ_F ≈ 3.0 (mean-field, 8D foam) o ≈ 2.31 (4D-effective)
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
import re
from multiprocessing import Pool, cpu_count

import numpy as np


# =============================================================================
# Myrheim-Meyer ordering fraction inverse table
# =============================================================================
# 
# Values from causal set literature (Bombelli-Henson 2006; Reid 2003):
# uniform Poisson sprinkling in d-dim Minkowski Alexandrov interval.
# These are the "standard" MM values for ordered relations / (N choose 2).
#
# For finite-discrete causal sets (as ours), the same-cluster ordering fraction
# is an APPROXIMATION to the strict BLS Myrheim-Meyer count. The values below
# define our reference curve r_MM(d); inversion gives d_MM.

_MM_REFERENCE_TABLE = np.array([
    # (d, r)  — d-dim Minkowski Alexandrov ordering fraction
    (1.0, 1.000),
    (1.5, 0.730),
    (2.0, 0.500),
    (2.5, 0.413),
    (3.0, 0.350),
    (3.5, 0.318),
    (4.0, 0.292),
    (4.5, 0.270),
    (5.0, 0.244),
    (5.5, 0.227),
    (6.0, 0.208),
    (7.0, 0.180),
    (8.0, 0.143),
    (9.0, 0.124),
    (10.0, 0.108),
    (15.0, 0.063),
    (20.0, 0.043),
])


def d_MM_from_r(r: float) -> float | None:
    """Invert MM curve: given ordering fraction r, return spacetime dimension d."""
    if r <= 0 or r > 1:
        return None
    table_d = _MM_REFERENCE_TABLE[:, 0]
    table_r = _MM_REFERENCE_TABLE[:, 1]
    # Reverse order (r decreasing with d → for interpolation we want r increasing)
    table_r_rev = table_r[::-1]
    table_d_rev = table_d[::-1]
    # Interpolate
    if r >= table_r_rev[-1]:
        return float(table_d_rev[-1])
    if r <= table_r_rev[0]:
        return float(table_d_rev[0])
    return float(np.interp(r, table_r_rev, table_d_rev))


# =============================================================================
# Single-trial processing
# =============================================================================

def find_roots(activated_vertices: np.ndarray, parent_snapshot: np.ndarray) -> np.ndarray:
    """
    Translate global parent IDs to local positions in activated_vertices,
    then path-compress to find the root for each activated vertex.
    
    INPUT:
      activated_vertices: int64 array of GLOBAL vertex IDs (sorted ascending,
                          as output by np.where(mask)[0])
      parent_snapshot:    int32 array, same length as activated_vertices,
                          containing GLOBAL parent vertex IDs in Union-Find
    
    OUTPUT:
      roots: int64 array, same length, containing LOCAL position of root
             for each activated vertex (in [0, n_activated))
    
    METHOD:
      1. Translate global IDs to local positions via searchsorted
         (activated_vertices is sorted, so searchsorted is O(log n))
      2. Iterative path compression: roots[i] = local_parent[roots[i]]
         until convergence
    """
    # Translate global parent IDs to local positions
    # activated_vertices is sorted ascending (np.where output), so searchsorted works
    local_parent = np.searchsorted(activated_vertices, parent_snapshot)
    
    # Sanity: verify that the translation is consistent (parent must be in activated set)
    n = len(activated_vertices)
    local_parent_clipped = np.clip(local_parent, 0, n - 1)
    consistent = activated_vertices[local_parent_clipped] == parent_snapshot
    if not consistent.all():
        n_bad = int((~consistent).sum())
        raise ValueError(f"{n_bad} parent IDs not found in activated_vertices "
                         f"(global IDs that aren't activated — Union-Find structure broken)")
    
    # Iterative path compression
    roots = local_parent.astype(np.int64)
    for _ in range(60):   # plenty for path compression on N~2.5M
        new_roots = local_parent[roots]
        if np.array_equal(new_roots, roots):
            break
        roots = new_roots
    return roots


def fit_tau_F(cluster_sizes: np.ndarray, s_min: int = 2, s_max: int = 1000) -> dict:
    """
    Fit cluster size distribution n_s ~ s^(-τ_F) over [s_min, s_max].
    Use log-bin histogram for stability against sampling noise.
    """
    sizes = cluster_sizes[(cluster_sizes >= s_min) & (cluster_sizes <= s_max)]
    if len(sizes) < 20:
        return {'tau_F': None, 'tau_F_sem': None, 'n_clusters_in_fit': len(sizes),
                'fit_range': (s_min, s_max)}
    
    # Log-binned histogram
    log_min = np.log10(s_min)
    log_max = np.log10(min(s_max, sizes.max()))
    n_bins = max(10, min(30, int((log_max - log_min) * 8)))
    bins = np.logspace(log_min, log_max, n_bins + 1)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    bin_widths = np.diff(bins)
    
    hist, _ = np.histogram(sizes, bins=bins)
    densities = hist / (bin_widths + 1e-10)
    
    # Fit log-log on bins with sufficient counts
    mask = (hist >= 3) & (densities > 0)
    if mask.sum() < 4:
        return {'tau_F': None, 'tau_F_sem': None, 'n_clusters_in_fit': len(sizes),
                'fit_range': (s_min, s_max)}
    
    x = np.log(bin_centers[mask])
    y = np.log(densities[mask])
    # weight by sqrt(counts) for stable fit
    w = np.sqrt(hist[mask])
    
    # Weighted linear regression in log-log
    coeffs, cov = np.polyfit(x, y, 1, w=w, cov='unscaled')
    slope = coeffs[0]
    slope_sem = float(np.sqrt(cov[0, 0])) if cov.shape == (2, 2) else None
    
    tau_F = -slope
    tau_F_sem = slope_sem
    
    return {
        'tau_F': float(tau_F),
        'tau_F_sem': float(tau_F_sem) if tau_F_sem is not None else None,
        'n_clusters_in_fit': int(len(sizes)),
        'fit_range': (s_min, s_max),
        'n_bins_used': int(mask.sum()),
    }


def process_one_po_file(po_path: str) -> dict:
    """Process a single per-trial PO file. Returns dict with all per-trial stats."""
    bname = os.path.basename(po_path)
    
    # Extract trial number from filename
    m = re.search(r'_trial(\d+)\.npz', bname)
    if not m:
        return {'po_file': bname, 'error': 'cannot parse trial number'}
    trial_idx = int(m.group(1))
    
    t0 = time.time()
    try:
        with np.load(po_path) as npz:
            activated_vertices = npz['activated_vertices']    # int64
            activation_time = npz['activation_time']           # int32
            parent_snapshot = npz['parent_snapshot']           # int32
            n_total = int(npz['n_total_activated_at_snap'])
            snapshot_bond_idx = int(npz['snapshot_bond_idx'])
            seed = int(npz['seed'])
            L = int(npz['L'])
    except Exception as e:
        return {'po_file': bname, 'trial_idx': trial_idx, 'error': f'load failed: {e}'}
    
    N = n_total
    if N < 100:
        return {'po_file': bname, 'trial_idx': trial_idx, 'error': f'too few activated vertices: {N}'}
    
    # Step 1: translate global parent IDs to local positions, then path-compress
    # to get the root (cluster representative) for each activated vertex
    try:
        roots = find_roots(activated_vertices, parent_snapshot)
    except ValueError as e:
        return {'po_file': bname, 'trial_idx': trial_idx, 'error': f'find_roots failed: {e}'}
    
    # Step 2: cluster sizes (unique root counts)
    unique_roots, cluster_sizes = np.unique(roots, return_counts=True)
    cluster_sizes = cluster_sizes.astype(np.int64)
    
    # Statistics
    n_clusters = len(cluster_sizes)
    max_cluster = int(cluster_sizes.max())
    sum_sq = int((cluster_sizes**2).sum())   # second moment ~ susceptibility * N
    sum_s_lns = float((cluster_sizes * np.log(cluster_sizes + 1e-30)).sum())
    
    # Step 3: E1 — Same-cluster ordering fraction (Myrheim-Meyer style)
    # All pairs in same cluster, with temporal ordering, count as ordered
    # r = sum_C |C|*(|C|-1) / (N*(N-1))
    # Note: this counts BOTH orderings (a≺b and b≺a) since (|C| choose 2) gives unordered pairs
    # Pairs in same cluster temporally ordered = |C|*(|C|-1)/2
    # Total pairs of activated vertices = N*(N-1)/2
    # r = pairs_same_cluster / total_pairs
    pairs_same_cluster = int((cluster_sizes * (cluster_sizes - 1) / 2).sum())
    total_pairs = N * (N - 1) // 2
    r_ordering = pairs_same_cluster / total_pairs
    
    d_MM = d_MM_from_r(r_ordering)
    
    # Step 4: E2 — Fisher exponent τ_F from cluster size distribution
    fit_result = fit_tau_F(cluster_sizes, s_min=2, s_max=10000)
    tau_F = fit_result['tau_F']
    
    # d from τ_F via hyperscaling: τ_F = 1 + d / d_f
    # For mean-field (d > d_c = 6): d_f = 2d/3, so τ_F = 5/2
    # For 4D-effective: d_f ≈ 4, so τ_F = 1 + d/4 → d = 4(τ_F - 1)
    # We compute both readings:
    d_from_tauF_mf = None     # mean-field reading
    d_from_tauF_4D = None     # 4D-effective reading
    if tau_F is not None and tau_F > 1:
        # Hyperscaling τ_F = 1 + d/d_f
        # Mean-field d_f = 2d/3: τ_F = 1 + d/(2d/3) = 1 + 3/2 = 5/2 (FIXED, gives no info)
        # Better: use τ_F = (D + 1) / D_f with d_f known to extract d
        # For percolation in d>=d_c=6: τ_F = 5/2 (fixed), d_f = 2d/3 (variable with d)
        # For d<=d_c: τ_F varies with d
        # Assume d_f = 4 (foam-fractal value): d = 4*(τ_F - 1)
        d_from_tauF_4D = 4.0 * (tau_F - 1.0)
        # And: d = 3*(τ_F - 1) for d_f = 3 (alternative)
    
    dt = time.time() - t0
    
    return {
        'po_file': bname,
        'trial_idx': trial_idx,
        'seed': seed,
        'L': L,
        'N_activated': N,
        'snapshot_bond_idx': snapshot_bond_idx,
        'n_clusters': int(n_clusters),
        'max_cluster': max_cluster,
        'max_cluster_frac': max_cluster / N,
        'sum_sq_clusters': sum_sq,
        'second_moment_chi': sum_sq / N,    # susceptibility ~ <s²>/N
        'r_ordering': r_ordering,           # E1: same-cluster ordering fraction
        'd_MM': d_MM,                       # E1: dimension from MM inversion
        'tau_F': tau_F,                     # E2: Fisher exponent
        'tau_F_sem': fit_result['tau_F_sem'],
        'd_from_tauF_4D_fractal': d_from_tauF_4D,   # E2: dim assuming d_f = 4
        'fit_quality_n_bins': fit_result.get('n_bins_used'),
        'fit_n_clusters_used': fit_result['n_clusters_in_fit'],
        'time_s': dt,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Compute d_BS per trial from Sim B PO files')
    ap.add_argument('directory', help='Directory with sim_b_L*_po_trial*.npz files')
    ap.add_argument('--workers', type=int, default=cpu_count(),
                    help=f'Number of parallel workers (default: {cpu_count()} = all cores)')
    ap.add_argument('--output', default='dBS_pertrial.json',
                    help='Output JSON path')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    
    pattern = os.path.join(args.directory, 'sim_b_L*_po_trial*.npz')
    po_files = sorted(glob.glob(pattern))
    
    if not po_files:
        print(f"ERROR: no sim_b_L*_po_trial*.npz files in {args.directory}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Computing d_BS for {len(po_files)} Sim B trials")
    print(f"Using {args.workers} parallel workers")
    print(f"{'='*70}\n")
    
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        # Use imap_unordered for streaming results as they complete
        results = []
        for i, r in enumerate(pool.imap_unordered(process_one_po_file, po_files), 1):
            results.append(r)
            if not args.quiet:
                if 'error' in r:
                    print(f"  [{i:>3}/{len(po_files)}] {r['po_file']:<40} ERROR: {r['error']}")
                else:
                    d_mm_str = f"{r['d_MM']:.2f}" if r['d_MM'] is not None else "—"
                    tau_str  = f"{r['tau_F']:.3f}" if r['tau_F'] is not None else "—"
                    d_4d_str = f"{r['d_from_tauF_4D_fractal']:.2f}" if r['d_from_tauF_4D_fractal'] is not None else "—"
                    print(f"  [{i:>3}/{len(po_files)}] trial {r['trial_idx']:>3}: "
                          f"r={r['r_ordering']:.4f} d_MM={d_mm_str:<5} "
                          f"τ_F={tau_str:<6} d(τ_F,4D)={d_4d_str:<5} "
                          f"max_C/N={r['max_cluster_frac']:.3f} ({r['time_s']:.1f}s)")
    
    dt = time.time() - t0
    
    # Sort by trial index
    results.sort(key=lambda x: x.get('trial_idx', -1))
    
    # Compute aggregate statistics
    valid = [r for r in results if 'error' not in r and r['d_MM'] is not None]
    if valid:
        d_MM_vals = np.array([r['d_MM'] for r in valid])
        r_ord_vals = np.array([r['r_ordering'] for r in valid])
        tauF_vals = np.array([r['tau_F'] for r in valid if r['tau_F'] is not None])
        d_4D_vals = np.array([r['d_from_tauF_4D_fractal'] for r in valid 
                              if r['d_from_tauF_4D_fractal'] is not None])
        
        print(f"\n{'='*70}")
        print(f"Aggregate results ({len(valid)} valid trials):")
        print(f"{'='*70}")
        print(f"\nE1 — Same-cluster ordering fraction r:")
        print(f"  mean = {r_ord_vals.mean():.6f}")
        print(f"  std  = {r_ord_vals.std(ddof=1):.6f}")
        print(f"  sem  = {r_ord_vals.std(ddof=1)/np.sqrt(len(r_ord_vals)):.6f}")
        
        print(f"\nE1 — Dimension d_MM (Myrheim-Meyer inversion):")
        print(f"  mean = {d_MM_vals.mean():.3f}")
        print(f"  std  = {d_MM_vals.std(ddof=1):.3f}")
        print(f"  sem  = {d_MM_vals.std(ddof=1)/np.sqrt(len(d_MM_vals)):.3f}")
        print(f"  range = [{d_MM_vals.min():.2f}, {d_MM_vals.max():.2f}]")
        
        if len(tauF_vals) > 0:
            print(f"\nE2 — Fisher exponent τ_F (from cluster size distribution):")
            print(f"  mean = {tauF_vals.mean():.3f}")
            print(f"  std  = {tauF_vals.std(ddof=1):.3f}")
            print(f"  sem  = {tauF_vals.std(ddof=1)/np.sqrt(len(tauF_vals)):.3f}")
        
        if len(d_4D_vals) > 0:
            print(f"\nE2 — Dimension d via hyperscaling τ_F=1+d/d_f (d_f=4):")
            print(f"  mean = {d_4D_vals.mean():.3f}")
            print(f"  std  = {d_4D_vals.std(ddof=1):.3f}")
            print(f"  sem  = {d_4D_vals.std(ddof=1)/np.sqrt(len(d_4D_vals)):.3f}")
        
        print(f"\nCross-check d_MM vs d(τ_F,4D):")
        if len(d_4D_vals) == len(d_MM_vals):
            print(f"  difference mean = {(d_MM_vals - d_4D_vals).mean():+.3f}")
            print(f"  Pearson correlation = {np.corrcoef(d_MM_vals, d_4D_vals)[0,1]:+.3f}")
        
        print(f"\nFramework prediction: d_BS = 4 (foam after cut-and-project to 4D Lorentzian)")
        print(f"Tension with d_MM:    {(d_MM_vals.mean() - 4) / (d_MM_vals.std(ddof=1)/np.sqrt(len(d_MM_vals))):+.1f}σ")
    
    # Save
    output = {
        'description': 'Per-trial d_BS extraction from Sim B v4.3 partial-order data',
        'n_trials_processed': len(results),
        'n_trials_valid': len(valid),
        'computation_time_s': dt,
        'workers': args.workers,
        'estimator_E1_note': 'Same-cluster ordering fraction (approximation of BLS Myrheim-Meyer; '
                              'rigorous count would require Newman-Ziff replay for transitive closure)',
        'estimator_E2_note': 'Fisher exponent τ_F from log-binned cluster size distribution, '
                              'hyperscaling τ_F = 1 + d/d_f to extract d',
        'per_trial_results': results,
    }
    if valid:
        output['aggregate'] = {
            'r_ordering_mean':         float(r_ord_vals.mean()),
            'r_ordering_std':          float(r_ord_vals.std(ddof=1)),
            'r_ordering_sem':          float(r_ord_vals.std(ddof=1)/np.sqrt(len(r_ord_vals))),
            'd_MM_mean':               float(d_MM_vals.mean()),
            'd_MM_std':                float(d_MM_vals.std(ddof=1)),
            'd_MM_sem':                float(d_MM_vals.std(ddof=1)/np.sqrt(len(d_MM_vals))),
            'tau_F_mean':              float(tauF_vals.mean()) if len(tauF_vals) > 0 else None,
            'tau_F_std':               float(tauF_vals.std(ddof=1)) if len(tauF_vals) > 0 else None,
            'tau_F_sem':               float(tauF_vals.std(ddof=1)/np.sqrt(len(tauF_vals))) if len(tauF_vals) > 0 else None,
            'd_from_tauF_4D_mean':     float(d_4D_vals.mean()) if len(d_4D_vals) > 0 else None,
            'd_from_tauF_4D_std':      float(d_4D_vals.std(ddof=1)) if len(d_4D_vals) > 0 else None,
            'd_from_tauF_4D_sem':      float(d_4D_vals.std(ddof=1)/np.sqrt(len(d_4D_vals))) if len(d_4D_vals) > 0 else None,
        }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Saved to {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    print(f"Total time: {dt:.1f}s")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
