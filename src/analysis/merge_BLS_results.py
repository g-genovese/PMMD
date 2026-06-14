#!/usr/bin/env python3
"""
merge_BLS_results.py
=====================

Aggrega gli output di compute_BLS_rigorous.py dai 3 server in un singolo
dBS_aggregate finale 128-trial.

USAGE:
  python3 merge_BLS_results.py BLS_A.json BLS_B.json BLS_C.json
  python3 merge_BLS_results.py BLS_*.json --output BLS_aggregate.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--output', default='BLS_aggregate.json')
    args = ap.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Merging BLS rigorous results from {len(args.inputs)} files")
    print(f"{'='*70}\n")
    
    all_results = []
    K_giant = None
    K_full = None
    for input_path in args.inputs:
        with open(input_path) as f:
            data = json.load(f)
        print(f"  {input_path}: {data['n_trials_valid']}/{data['n_trials_processed']} valid")
        K_giant = data.get('K_giant')
        K_full = data.get('K_full')
        for r in data['per_trial_results']:
            r['source_file'] = os.path.basename(input_path)
            all_results.append(r)
    
    # Sort & validate coverage
    all_results.sort(key=lambda x: x.get('trial_idx', -1))
    trial_ids = [r['trial_idx'] for r in all_results if 'trial_idx' in r]
    seen = set()
    duplicates = []
    for t in trial_ids:
        if t in seen:
            duplicates.append(t)
        seen.add(t)
    missing = sorted(set(range(128)) - seen)
    
    if duplicates:
        print(f"\n⚠️  Duplicate trials: {duplicates}")
    if missing:
        print(f"\n⚠️  Missing trials: {missing}")
    else:
        print(f"\n  ✓ All 128 trials [0, 127] covered")
    
    valid = [r for r in all_results if 'error' not in r and r.get('d_BS_giant') is not None]
    if not valid:
        print("ERROR: no valid trials")
        sys.exit(1)
    
    r_g = np.array([r['r_giant'] for r in valid])
    d_g = np.array([r['d_BS_giant'] for r in valid])
    r_f = np.array([r['r_full'] for r in valid if r['r_full'] is not None])
    d_f = np.array([r['d_BS_full'] for r in valid if r['d_BS_full'] is not None])
    gc_size = np.array([r['giant_cluster_size'] for r in valid])
    gc_frac = np.array([r['giant_cluster_frac'] for r in valid])
    
    n = len(valid)
    
    print(f"\n{'='*70}")
    print(f"Aggregate (128 trials, K_giant={K_giant}, K_full={K_full})")
    print(f"{'='*70}")
    
    print(f"\nGiant cluster stats:")
    print(f"  size: mean = {gc_size.mean():.0f}, std = {gc_size.std(ddof=1):.0f}")
    print(f"  fraction of N: mean = {gc_frac.mean():.4f}, std = {gc_frac.std(ddof=1):.4f}")
    
    print(f"\n=== E1: Giant cluster BLS (the main measurement) ===")
    sem_r_g = r_g.std(ddof=1)/np.sqrt(n)
    sem_d_g = d_g.std(ddof=1)/np.sqrt(n)
    print(f"  r_giant: mean = {r_g.mean():.5f} ± {sem_r_g:.5f}")
    print(f"  d_BS_giant: mean = {d_g.mean():.3f} ± {sem_d_g:.3f}")
    print(f"  std d_BS_giant: {d_g.std(ddof=1):.3f}")
    print(f"  range d_BS_giant: [{d_g.min():.2f}, {d_g.max():.2f}]")
    
    print(f"\n=== Framework prediction: d_BS = 4 ===")
    tension_g = (d_g.mean() - 4) / sem_d_g
    print(f"  Tension d_BS_giant - 4: {d_g.mean() - 4:+.3f} ({tension_g:+.2f}σ)")
    
    promotion_E1 = abs(tension_g) < 2.0
    print(f"\n  Bridge claim promotion (tension < 2σ): {'YES ✓' if promotion_E1 else 'NO (tension > 2σ)'}")
    
    if len(d_f) > 0:
        print(f"\n=== E2: Full activated BLS (cross-check) ===")
        sem_d_f = d_f.std(ddof=1)/np.sqrt(len(d_f))
        print(f"  r_full: mean = {r_f.mean():.6f} ± {r_f.std(ddof=1)/np.sqrt(len(r_f)):.6f}")
        print(f"  d_BS_full: mean = {d_f.mean():.3f} ± {sem_d_f:.3f}")
        tension_f = (d_f.mean() - 4) / sem_d_f
        print(f"  Tension d_BS_full - 4: {d_f.mean() - 4:+.3f} ({tension_f:+.2f}σ)")
    
    output = {
        'description': 'Aggregate BLS dimension extraction across all 128 Sim B trials',
        'n_inputs': len(args.inputs),
        'input_files': args.inputs,
        'n_trials_total': len(all_results),
        'n_trials_valid': n,
        'duplicates': duplicates,
        'missing_trials': missing,
        'K_giant': K_giant,
        'K_full': K_full,
        'aggregate': {
            'r_giant_mean': float(r_g.mean()),
            'r_giant_std': float(r_g.std(ddof=1)),
            'r_giant_sem': float(sem_r_g),
            'd_BS_giant_mean': float(d_g.mean()),
            'd_BS_giant_std': float(d_g.std(ddof=1)),
            'd_BS_giant_sem': float(sem_d_g),
            'd_BS_giant_min': float(d_g.min()),
            'd_BS_giant_max': float(d_g.max()),
            'tension_giant_vs_4_sigma': float(tension_g),
            'bridge_promotion_E1': bool(promotion_E1),
            'r_full_mean': float(r_f.mean()) if len(r_f) > 0 else None,
            'r_full_sem': float(r_f.std(ddof=1)/np.sqrt(len(r_f))) if len(r_f) > 0 else None,
            'd_BS_full_mean': float(d_f.mean()) if len(d_f) > 0 else None,
            'd_BS_full_sem': float(d_f.std(ddof=1)/np.sqrt(len(d_f))) if len(d_f) > 0 else None,
            'tension_full_vs_4_sigma': float((d_f.mean() - 4) / (d_f.std(ddof=1)/np.sqrt(len(d_f)))) if len(d_f) > 0 else None,
            'giant_cluster_size_mean': float(gc_size.mean()),
            'giant_cluster_frac_mean': float(gc_frac.mean()),
        },
        'per_trial_results': all_results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
