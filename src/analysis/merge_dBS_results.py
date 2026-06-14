#!/usr/bin/env python3
"""
merge_dBS_results.py
=====================

Aggrega gli output di compute_dBS_full.py dai 3 server in un singolo file
unificato 128-trial.

USAGE:
  python3 merge_dBS_results.py dBS_A.json dBS_B.json dBS_C.json
  python3 merge_dBS_results.py dBS_*.json
  python3 merge_dBS_results.py dBS_*.json --output dBS_aggregate.json

Verifica copertura [0, 127] senza gap/overlap, ricalcola aggregate statistics,
produce report sintetico.
"""

from __future__ import annotations
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description='Merge per-server d_BS outputs')
    ap.add_argument('inputs', nargs='+', help='Per-server JSON files from compute_dBS_full.py')
    ap.add_argument('--output', default='dBS_aggregate.json', help='Output path')
    args = ap.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Merging d_BS results from {len(args.inputs)} server(s)")
    print(f"{'='*70}\n")
    
    all_results = []
    for input_path in args.inputs:
        with open(input_path) as f:
            data = json.load(f)
        print(f"  {input_path}: {data['n_trials_valid']}/{data['n_trials_processed']} valid trials")
        for r in data['per_trial_results']:
            r['source_file'] = input_path
            all_results.append(r)
    
    # Sort by trial index
    all_results.sort(key=lambda x: x.get('trial_idx', -1))
    
    # Validate coverage
    trial_ids = [r['trial_idx'] for r in all_results if 'trial_idx' in r]
    seen = set()
    duplicates = []
    for t in trial_ids:
        if t in seen:
            duplicates.append(t)
        seen.add(t)
    if duplicates:
        print(f"\n⚠️  WARNING: duplicate trials: {duplicates}")
    
    expected_range = set(range(128))  # Sim B has 128 trials [0, 127]
    missing = sorted(expected_range - seen)
    if missing:
        print(f"\n⚠️  WARNING: missing trials: {missing}")
    else:
        print(f"\n  ✓ All 128 trials [0, 127] covered")
    
    # Compute aggregate stats
    valid = [r for r in all_results if 'error' not in r and r.get('d_MM') is not None]
    if not valid:
        print("ERROR: no valid trials")
        sys.exit(1)
    
    d_MM_vals = np.array([r['d_MM'] for r in valid])
    r_ord_vals = np.array([r['r_ordering'] for r in valid])
    tauF_vals = np.array([r['tau_F'] for r in valid if r.get('tau_F') is not None])
    d_4D_vals = np.array([r['d_from_tauF_4D_fractal'] for r in valid 
                          if r.get('d_from_tauF_4D_fractal') is not None])
    max_frac_vals = np.array([r['max_cluster_frac'] for r in valid])
    chi_vals = np.array([r['second_moment_chi'] for r in valid])
    
    n = len(valid)
    
    print(f"\n{'='*70}")
    print(f"Aggregate statistics — 128 trials at L=12")
    print(f"{'='*70}")
    
    print(f"\nE1 — Same-cluster ordering fraction r:")
    print(f"  mean = {r_ord_vals.mean():.6f}")
    print(f"  std  = {r_ord_vals.std(ddof=1):.6f}")
    print(f"  sem  = {r_ord_vals.std(ddof=1)/np.sqrt(n):.6f}")
    print(f"  range = [{r_ord_vals.min():.4f}, {r_ord_vals.max():.4f}]")
    
    print(f"\nE1 — Dimension d_MM (Myrheim-Meyer):")
    print(f"  mean ± sem = {d_MM_vals.mean():.3f} ± {d_MM_vals.std(ddof=1)/np.sqrt(n):.3f}")
    print(f"  std  = {d_MM_vals.std(ddof=1):.3f}")
    print(f"  range = [{d_MM_vals.min():.2f}, {d_MM_vals.max():.2f}]")
    
    if len(tauF_vals) > 0:
        print(f"\nE2 — Fisher exponent τ_F:")
        print(f"  mean ± sem = {tauF_vals.mean():.3f} ± {tauF_vals.std(ddof=1)/np.sqrt(len(tauF_vals)):.3f}")
        print(f"  std  = {tauF_vals.std(ddof=1):.3f}")
    
    if len(d_4D_vals) > 0:
        print(f"\nE2 — Dimension d (via hyperscaling, d_f=4):")
        print(f"  mean ± sem = {d_4D_vals.mean():.3f} ± {d_4D_vals.std(ddof=1)/np.sqrt(len(d_4D_vals)):.3f}")
        print(f"  std  = {d_4D_vals.std(ddof=1):.3f}")
    
    print(f"\nMax cluster fraction (giant cluster size / N):")
    print(f"  mean = {max_frac_vals.mean():.4f}")
    print(f"  std  = {max_frac_vals.std(ddof=1):.4f}")
    
    print(f"\nSusceptibility χ = <s²>/N:")
    print(f"  mean = {chi_vals.mean():.2f}")
    print(f"  std  = {chi_vals.std(ddof=1):.2f}")
    
    print(f"\n{'='*70}")
    print(f"Framework prediction: d_BS = 4")
    print(f"{'='*70}")
    sem_MM = d_MM_vals.std(ddof=1)/np.sqrt(n)
    tension_MM = (d_MM_vals.mean() - 4) / sem_MM
    print(f"E1 (d_MM):     {d_MM_vals.mean():.3f} ± {sem_MM:.3f}  → tension with 4: {tension_MM:+.1f}σ")
    if len(d_4D_vals) > 0:
        sem_4D = d_4D_vals.std(ddof=1)/np.sqrt(len(d_4D_vals))
        tension_4D = (d_4D_vals.mean() - 4) / sem_4D
        print(f"E2 (d_τF,4D):  {d_4D_vals.mean():.3f} ± {sem_4D:.3f}  → tension with 4: {tension_4D:+.1f}σ")
    
    # Bridge claims status
    print(f"\nBridge claim promotion target: |d_BS - 4| / sem < 2σ for unconditional Stratum 1")
    promoted_MM = abs(tension_MM) < 2.0
    print(f"  E1: {'PROMOTED ✓' if promoted_MM else 'NOT PROMOTED (tension > 2σ)'}")
    if len(d_4D_vals) > 0:
        promoted_4D = abs(tension_4D) < 2.0
        print(f"  E2: {'PROMOTED ✓' if promoted_4D else 'NOT PROMOTED (tension > 2σ)'}")
    
    # Save
    output = {
        'description': 'Aggregate d_BS results across all 128 Sim B trials',
        'n_inputs': len(args.inputs),
        'input_files': args.inputs,
        'n_trials_total': len(all_results),
        'n_trials_valid': n,
        'duplicates': duplicates,
        'missing_trials': missing,
        'aggregate': {
            'r_ordering_mean':         float(r_ord_vals.mean()),
            'r_ordering_std':          float(r_ord_vals.std(ddof=1)),
            'r_ordering_sem':          float(r_ord_vals.std(ddof=1)/np.sqrt(n)),
            'd_MM_mean':               float(d_MM_vals.mean()),
            'd_MM_std':                float(d_MM_vals.std(ddof=1)),
            'd_MM_sem':                float(d_MM_vals.std(ddof=1)/np.sqrt(n)),
            'tau_F_mean':              float(tauF_vals.mean()) if len(tauF_vals) > 0 else None,
            'tau_F_sem':               float(tauF_vals.std(ddof=1)/np.sqrt(len(tauF_vals))) if len(tauF_vals) > 0 else None,
            'd_from_tauF_4D_mean':     float(d_4D_vals.mean()) if len(d_4D_vals) > 0 else None,
            'd_from_tauF_4D_sem':      float(d_4D_vals.std(ddof=1)/np.sqrt(len(d_4D_vals))) if len(d_4D_vals) > 0 else None,
            'max_cluster_frac_mean':   float(max_frac_vals.mean()),
            'susceptibility_chi_mean': float(chi_vals.mean()),
            'tension_d_MM_vs_4':       float(tension_MM),
            'bridge_promotion_E1':     bool(abs(tension_MM) < 2.0),
        },
        'per_trial_results': all_results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    import os
    print(f"\nSaved {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
