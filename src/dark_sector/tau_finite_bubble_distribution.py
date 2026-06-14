#!/usr/bin/env python3
"""
tau_finite_bubble_distribution.py

EXTENDED simulation: separates the giant τ-homogeneous clusters from the FINITE
τ-bubbles, and measures the size distribution n_s of finite τ-homogeneous
sub-clusters. This is the structurally relevant quantity for the PBH mass
spectrum, since only finite compact-boundary regions are genuine event horizons.

THEORY:
  In 3D Bernoulli(1/2) supercritical phase, finite-cluster size distribution
  is expected exponentially decaying:
      n_s ~ exp(-s/s*) / s^θ
  where s* is the characteristic mass scale (away from criticality, exponential
  cutoff dominates; near p_c it would be a power law with τ_Fisher exponent).
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import label
from collections import Counter
import time


def simulate_finite_bubbles(L: int, rng: np.random.Generator) -> dict:
    """Simulate, identify giant clusters, isolate finite bubbles."""
    N = L**3
    tau = rng.integers(0, 2, size=(L, L, L), dtype=np.int8) * 2 - 1   # {-1, +1}
    
    structure = np.zeros((3,3,3), dtype=bool)
    structure[1,1,:] = True; structure[1,:,1] = True; structure[:,1,1] = True
    
    pos_regions, n_pos = label(tau == +1, structure=structure)
    neg_regions, n_neg = label(tau == -1, structure=structure)
    
    pos_sizes = np.bincount(pos_regions.ravel())[1:]
    neg_sizes = np.bincount(neg_regions.ravel())[1:]
    
    # Identify giant cluster on each side (largest), exclude
    if len(pos_sizes) > 0:
        giant_pos = pos_sizes.max()
        finite_pos = pos_sizes[pos_sizes < giant_pos]
    else:
        giant_pos = 0; finite_pos = np.array([])
    
    if len(neg_sizes) > 0:
        giant_neg = neg_sizes.max()
        finite_neg = neg_sizes[neg_sizes < giant_neg]
    else:
        giant_neg = 0; finite_neg = np.array([])
    
    all_finite = np.concatenate([finite_pos, finite_neg]).astype(int)
    
    # Use the giant components to assess "macroscopic" τ-balance
    f_pos_giant = giant_pos / N
    f_neg_giant = giant_neg / N
    f_finite = all_finite.sum() / N
    f_unaccounted = 1 - f_pos_giant - f_neg_giant - f_finite  # should be 0
    
    return {
        'L': L, 'N': N,
        'giant_pos': int(giant_pos),
        'giant_neg': int(giant_neg),
        'f_giant_pos': float(f_pos_giant),
        'f_giant_neg': float(f_neg_giant),
        'f_finite_total': float(f_finite),
        'f_unaccounted': float(f_unaccounted),
        'n_finite_clusters': len(all_finite),
        'finite_sizes': all_finite.tolist(),
        'largest_finite': int(all_finite.max()) if len(all_finite) > 0 else 0,
    }


def main():
    print("="*72)
    print("Finite-bubble distribution in supercritical Bernoulli(1/2) on 3D cubic")
    print("="*72)
    
    L_values = [40, 60, 80, 100]
    n_trials = [20, 10, 5, 3]
    
    pooled_sizes = []
    pooled_giant_fractions = []
    pooled_finite_fractions = []
    
    rng = np.random.default_rng(20260519)
    
    for L, nt in zip(L_values, n_trials):
        N = L**3
        t0 = time.time()
        f_finite_list = []
        n_finite_list = []
        largest_finite_list = []
        
        for trial in range(nt):
            r = simulate_finite_bubbles(L, rng)
            pooled_sizes.extend(r['finite_sizes'])
            pooled_giant_fractions.extend([r['f_giant_pos'], r['f_giant_neg']])
            pooled_finite_fractions.append(r['f_finite_total'])
            f_finite_list.append(r['f_finite_total'])
            n_finite_list.append(r['n_finite_clusters'])
            largest_finite_list.append(r['largest_finite'])
        
        elapsed = time.time() - t0
        n_clusters_mean = np.mean(n_finite_list)
        f_finite_mean = np.mean(f_finite_list)
        largest_mean = np.mean(largest_finite_list)
        
        print(f"\nL = {L:3d}  (N = {N:>9,d}, {nt} trials, {elapsed:.1f}s)")
        print(f"  Finite-cluster total fraction: {f_finite_mean:.4f}")
        print(f"  Number finite clusters/trial:  {n_clusters_mean:.0f}")
        print(f"  Density per N:                 {n_clusters_mean/N:.4f}")
        print(f"  Largest finite bubble:         {largest_mean:.0f} vertices")
    
    # ---- Distribution analysis ----
    sizes = np.array(pooled_sizes)
    if len(sizes) > 0:
        print(f"\n{'='*72}")
        print(f"Pooled finite-cluster size distribution ({len(sizes)} clusters)")
        print(f"{'='*72}")
        
        # Histogram with logarithmic binning
        log_min, log_max = 0, np.log10(sizes.max() + 1)
        bins = np.logspace(log_min, log_max, 20)
        hist, bin_edges = np.histogram(sizes, bins=bins)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        densities = hist / (bin_widths + 1e-10)
        
        print(f"\n{'size_range':>20} {'count':>8} {'density':>12}")
        for i in range(len(hist)):
            if hist[i] > 0:
                lo, hi = bin_edges[i], bin_edges[i+1]
                print(f"  [{lo:6.1f}, {hi:7.1f}]   {hist[i]:>8d} {densities[i]:>12.4f}")
        
        # Fit n_s ~ s^(-tau) * exp(-s/s_star) — log-log fit on s, log(n_s)
        # In supercritical phase, exponential cutoff is expected
        mask = (hist > 5)  # only well-populated bins
        if mask.sum() >= 4:
            x = np.log(bin_centers[mask])
            y = np.log(densities[mask])
            # Linear fit: log(n) = log(A) - tau*log(s) - s/s_star (linear in log(s) + s)
            # Try simple power-law first
            try:
                # log(density) = log(A) - tau*log(s)
                tau_fit, log_A = np.polyfit(x, y, 1)
                tau_fit = -tau_fit
                print(f"\nPower-law fit (excluding exponential cutoff):")
                print(f"  n_s ~ s^(-tau),  tau = {tau_fit:.2f}")
                print(f"  Theoretical for 3D supercritical (exp cutoff dominant): tau ~ 2.18")
            except Exception as e:
                print(f"Fit failed: {e}")
        
        # Compute statistics  
        print(f"\nFinite-cluster statistics:")
        print(f"  Mean size:       {sizes.mean():.2f}")
        print(f"  Median size:     {np.median(sizes):.1f}")
        print(f"  Max size:        {sizes.max():.0f}")
        print(f"  Singleton frac:  {(sizes == 1).mean():.4f}")
        print(f"  Size ≤ 5 frac:   {(sizes <= 5).mean():.4f}")
        print(f"  Size ≤ 100 frac: {(sizes <= 100).mean():.4f}")
        
        # Characteristic size s* (from mean of non-singletons)
        non_single = sizes[sizes > 1]
        if len(non_single) > 0:
            print(f"  Mean (excl. singletons): {non_single.mean():.2f}")
    
    print(f"\n{'='*72}")
    print(f"Volume balance check (giant + finite + unaccounted)")
    print(f"{'='*72}")
    print(f"Giant clusters fraction (each):  {np.mean(pooled_giant_fractions):.4f}")
    print(f"  expected from P_inf(0.5,3Dcubic): ~0.489")
    print(f"Finite total fraction:           {np.mean(pooled_finite_fractions):.4f}")
    print(f"  expected: ~0.022 (1 - 2*0.489)")
    
    # Save
    import json
    out = {
        'description': 'Finite tau-bubble size distribution in 3D Bernoulli(1/2) supercritical',
        'theoretical_pinf_05': 0.489,
        'expected_finite_fraction': 0.022,
        'observed_pinf_per_side': float(np.mean(pooled_giant_fractions)),
        'observed_finite_total': float(np.mean(pooled_finite_fractions)),
        'pooled_finite_sizes': sizes.tolist() if len(sizes) > 0 else [],
        'n_finite_clusters_total': int(len(sizes)),
    }
    with open('tau_finite_bubble_distribution.json', 'w') as f:
        json.dump(out, f, default=str)
    print(f"\nResults saved to tau_finite_bubble_distribution.json")


if __name__ == '__main__':
    main()
