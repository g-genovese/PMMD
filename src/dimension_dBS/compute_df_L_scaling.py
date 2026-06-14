#!/usr/bin/env python3
"""
compute_df_L_scaling.py
========================

Estrae la dimensione frattale d_f del giant cluster percolante al p_c via
scaling con L, usando i dati FSS esistenti L ∈ {8, 10, 12}.

PRINCIPIO STRUTTURALE
---------------------
In percolazione critica a p_c, il giant cluster ha dimensione frattale d_f:
    |C_max|(L) ~ L^{d_f}

Per 8D Bernoulli mean-field percolation (d > d_c = 6):
    d_f^{MF} = 4 (universale)

Questo è il candidato naturale per il "d_BS = 4" del framework PMMD, perché:
- È misurabile direttamente dai dati esistenti
- È coerente con 8D embedding + mean-field universality class
- Coincide numericamente col valore predetto dal framework

INPUT
-----
Una o più directory contenenti file sim_b_L*_*.json + _curves.npz.

OUTPUT
------
JSON con:
- Per ogni L: |C_max|(p_c) statistics (mean, std, sem)
- Linear fit: log|C_max| = d_f * log(L) + c
- Tension vs predizione d_f = 4

USAGE
-----
  python3 compute_df_L_scaling.py /path/to/sim_b_L8/ /path/to/sim_b_L10/ /path/to/sim_b_L12/
  python3 compute_df_L_scaling.py /path/to/all_sim_b/    # tutti L mescolati
  python3 compute_df_L_scaling.py /path/to/dir/ --output df_results.json
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np


def discover_files(directories: list[str]) -> dict[int, list[dict]]:
    """Find all sim_b_L*_*.json files in input dirs, group by L."""
    by_L = defaultdict(list)
    for d in directories:
        pattern = os.path.join(d, 'sim_b_L*_*.json')
        for jp in sorted(glob.glob(pattern)):
            if 'aggregate' in os.path.basename(jp):
                continue
            cp = jp.replace('.json', '_curves.npz')
            if not os.path.exists(cp):
                print(f"  WARNING: no curves.npz for {os.path.basename(jp)}")
                continue
            try:
                with open(jp) as f:
                    meta = json.load(f)
                L = int(meta['L'])
                by_L[L].append({
                    'json_path': jp,
                    'curves_path': cp,
                    'L': L,
                    'N': meta.get('N', L ** 8),
                    'n_bins': meta['n_bins'],
                    'trial_start': meta['trial_start'],
                    'trials': meta['trials'],
                    'pc_chi': meta['pc_chi'],
                    'target_p': meta.get('target_p'),
                    'meta': meta,
                })
            except Exception as e:
                print(f"  WARNING: skipping {os.path.basename(jp)}: {e}")
    return dict(by_L)


def aggregate_curves_for_L(files: list[dict], quiet: bool = False) -> dict:
    """For one L, load all curves and stack per-trial arrays."""
    L = files[0]['L']
    n_bins = files[0]['n_bins']
    N = files[0]['N']
    
    # Sort by trial_start to ensure proper ordering
    files = sorted(files, key=lambda f: f['trial_start'])
    
    # Validate coverage
    expected_start = files[0]['trial_start']
    cursor = expected_start
    total_trials = 0
    for f in files:
        if f['trial_start'] != cursor:
            print(f"  WARNING (L={L}): gap or overlap at trial {cursor}/{f['trial_start']}")
        cursor = f['trial_start'] + f['trials']
        total_trials += f['trials']
    
    if not quiet:
        print(f"  L={L}: {len(files)} files, {total_trials} trials total")
    
    # Allocate pooled arrays
    S_max_all = np.zeros((total_trials, n_bins), dtype=np.float64)
    sum_sq_all = np.zeros((total_trials, n_bins), dtype=np.float64)
    
    cursor = 0
    ps = None
    for f in files:
        with np.load(f['curves_path']) as npz:
            if ps is None:
                ps = npz['ps'].astype(np.float64)
            else:
                # Sanity check: ps must match across files
                ps_this = npz['ps'].astype(np.float64)
                if not np.allclose(ps, ps_this):
                    print(f"  WARNING (L={L}): ps mismatch in {os.path.basename(f['json_path'])}")
            n_t = f['trials']
            S_max_all[cursor:cursor+n_t] = npz['S_max_all']
            sum_sq_all[cursor:cursor+n_t] = npz['sum_sq_all']
            cursor += n_t
    
    return {
        'L': L,
        'N': N,
        'n_bins': n_bins,
        'total_trials': total_trials,
        'ps': ps,
        'S_max_all': S_max_all,
        'sum_sq_all': sum_sq_all,
    }


def extract_Cmax_at_pc(agg: dict, quiet: bool = False) -> dict:
    """For each trial, find p_c via chi peak, then read S_max at that bin."""
    L = agg['L']
    N = agg['N']
    n_bins = agg['n_bins']
    n_trials = agg['total_trials']
    ps = agg['ps']
    S_max_all = agg['S_max_all']
    sum_sq_all = agg['sum_sq_all']
    
    # n_active grows linearly with bin: n_active[b] = (b+1) * (N // n_bins)
    bin_size = N // n_bins
    n_active = (np.arange(1, n_bins + 1) * bin_size).astype(np.float64)
    
    Cmax_at_pc = np.zeros(n_trials)
    pc_per_trial = np.zeros(n_trials)
    
    for t in range(n_trials):
        S = S_max_all[t]
        SS = sum_sq_all[t]
        denom = n_active - S
        chi = np.where(denom > 0, (SS - S**2) / denom, 0.0)
        pc_bin = int(chi.argmax())
        pc_per_trial[t] = ps[pc_bin]
        Cmax_at_pc[t] = S[pc_bin]
    
    if not quiet:
        print(f"    p_c per-trial:  {pc_per_trial.mean():.6f} ± {pc_per_trial.std(ddof=1)/np.sqrt(n_trials):.6f}")
        print(f"    |C_max|@p_c:    {Cmax_at_pc.mean():.1f} ± {Cmax_at_pc.std(ddof=1)/np.sqrt(n_trials):.1f}  "
              f"(std = {Cmax_at_pc.std(ddof=1):.1f})")
    
    return {
        'L': L,
        'N': N,
        'n_trials': n_trials,
        'pc_per_trial_mean': float(pc_per_trial.mean()),
        'pc_per_trial_sem': float(pc_per_trial.std(ddof=1) / np.sqrt(n_trials)),
        'Cmax_at_pc_per_trial': Cmax_at_pc,
        'Cmax_mean': float(Cmax_at_pc.mean()),
        'Cmax_std': float(Cmax_at_pc.std(ddof=1)),
        'Cmax_sem': float(Cmax_at_pc.std(ddof=1) / np.sqrt(n_trials)),
        'Cmax_frac_N_mean': float((Cmax_at_pc / N).mean()),
    }


def fit_df_log_log(results_by_L: dict[int, dict]) -> dict:
    """
    Linear regression: log|C_max|(L) = d_f * log(L) + c
    
    With error bars: y_err = sem/Cmax in log space (relative error).
    Weighted least squares using log-domain errors.
    """
    Ls = sorted(results_by_L.keys())
    if len(Ls) < 2:
        return {'error': 'need at least 2 L values for fit'}
    
    log_L = np.log(np.array(Ls, dtype=np.float64))
    log_Cmax = np.array([np.log(results_by_L[L]['Cmax_mean']) for L in Ls])
    log_Cmax_err = np.array([
        results_by_L[L]['Cmax_sem'] / results_by_L[L]['Cmax_mean']
        for L in Ls
    ])
    
    # Weighted linear regression
    weights = 1.0 / (log_Cmax_err ** 2)
    coeffs, cov = np.polyfit(log_L, log_Cmax, 1, w=np.sqrt(weights), cov='unscaled')
    d_f = float(coeffs[0])
    intercept = float(coeffs[1])
    d_f_sem = float(np.sqrt(cov[0, 0])) if cov.shape == (2, 2) else None
    
    # R² (goodness of fit)
    log_Cmax_pred = d_f * log_L + intercept
    ss_res = float(np.sum(((log_Cmax - log_Cmax_pred) * np.sqrt(weights)) ** 2))
    ss_tot = float(np.sum(((log_Cmax - log_Cmax.mean()) * np.sqrt(weights)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    
    # Residuals (in log space)
    residuals = (log_Cmax - log_Cmax_pred).tolist()
    
    # Predicted at L: a * L^{d_f}
    fit_table = []
    a = np.exp(intercept)
    for L in Ls:
        L_v = float(L)
        Cmax_pred = a * L_v ** d_f
        Cmax_obs = results_by_L[L]['Cmax_mean']
        fit_table.append({
            'L': L,
            'Cmax_observed': float(Cmax_obs),
            'Cmax_observed_sem': results_by_L[L]['Cmax_sem'],
            'Cmax_predicted': float(Cmax_pred),
            'ratio_obs_pred': float(Cmax_obs / Cmax_pred),
            'residual_log': float(np.log(Cmax_obs) - np.log(Cmax_pred)),
        })
    
    return {
        'd_f': d_f,
        'd_f_sem': d_f_sem,
        'intercept': intercept,
        'a': float(a),
        'r_squared': r_squared,
        'residuals': residuals,
        'fit_table': fit_table,
        'log_L_used': log_L.tolist(),
        'log_Cmax_used': log_Cmax.tolist(),
        'log_Cmax_err_used': log_Cmax_err.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(
        description='Extract d_f via L-scaling from FSS data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('directories', nargs='+',
                    help='One or more directories containing sim_b_L*_*.json + _curves.npz')
    ap.add_argument('--output', default='df_L_scaling_results.json')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    
    t0 = time.time()
    
    print(f"\n{'='*70}")
    print(f"Fractal dimension d_f via L-scaling")
    print(f"{'='*70}\n")
    
    print("Step 1/4: Discovering files")
    by_L = discover_files(args.directories)
    if not by_L:
        print("ERROR: no files found")
        sys.exit(1)
    
    print(f"  Found data for L = {sorted(by_L.keys())}")
    
    print("\nStep 2/4: Aggregating curves per L")
    aggregates = {}
    for L in sorted(by_L.keys()):
        aggregates[L] = aggregate_curves_for_L(by_L[L], quiet=args.quiet)
    
    print("\nStep 3/4: Extracting |C_max| at p_c per trial")
    results_by_L = {}
    for L in sorted(by_L.keys()):
        if not args.quiet:
            print(f"  L = {L}:")
        results_by_L[L] = extract_Cmax_at_pc(aggregates[L], quiet=args.quiet)
    
    print("\nStep 4/4: Fitting log|C_max| = d_f log L + c")
    fit = fit_df_log_log(results_by_L)
    if 'error' in fit:
        print(f"  ERROR: {fit['error']}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Result")
    print(f"{'='*70}")
    print(f"\n  d_f = {fit['d_f']:.4f} ± {fit['d_f_sem']:.4f}")
    print(f"  R² = {fit['r_squared']:.6f}")
    print(f"  Fit: |C_max|(L) = {fit['a']:.3e} · L^{fit['d_f']:.4f}")
    
    print(f"\n  Fit table:")
    print(f"  {'L':<5} {'|C_max| obs':<20} {'|C_max| pred':<15} {'ratio':<10} {'residual (log)':<15}")
    print(f"  {'-'*5} {'-'*20} {'-'*15} {'-'*10} {'-'*15}")
    for row in fit['fit_table']:
        print(f"  {row['L']:<5} {row['Cmax_observed']:>10.1f} ± {row['Cmax_observed_sem']:<6.1f} "
              f"{row['Cmax_predicted']:<15.1f} {row['ratio_obs_pred']:<10.4f} {row['residual_log']:<+15.5f}")
    
    print(f"\n  Predictions to compare:")
    print(f"    Mean-field 8D Bernoulli (universal): d_f^MF = 4.0  exactly")
    print(f"    Framework d_BS:                       d_BS = 4  (if d_BS ≡ d_f)")
    
    tension_4 = (fit['d_f'] - 4.0) / fit['d_f_sem']
    print(f"\n  Tension d_f - 4 / sem = {tension_4:+.2f}σ")
    promotion = abs(tension_4) < 2.0
    print(f"  Compatible with d_f = 4 (|tension| < 2σ): {'YES ✓' if promotion else 'NO'}")
    if promotion:
        print(f"  Bridge claim promotion target reached.")
    
    # Save
    output = {
        'description': 'd_f via L-scaling from Sim B FSS data (L=8,10,12)',
        'directories_searched': args.directories,
        'n_L_values': len(by_L),
        'L_values': sorted(by_L.keys()),
        'per_L_stats': {
            str(L): {
                'L': L,
                'N': results_by_L[L]['N'],
                'n_trials': results_by_L[L]['n_trials'],
                'pc_per_trial_mean': results_by_L[L]['pc_per_trial_mean'],
                'pc_per_trial_sem': results_by_L[L]['pc_per_trial_sem'],
                'Cmax_mean': results_by_L[L]['Cmax_mean'],
                'Cmax_std': results_by_L[L]['Cmax_std'],
                'Cmax_sem': results_by_L[L]['Cmax_sem'],
                'Cmax_frac_N_mean': results_by_L[L]['Cmax_frac_N_mean'],
            } for L in sorted(by_L.keys())
        },
        'fit': fit,
        'predictions': {
            'd_f_MF': 4.0,
            'd_BS_framework': 4.0,
        },
        'tension_vs_4_sigma': float(tension_4),
        'compatible_with_4': bool(promotion),
        'computation_time_s': time.time() - t0,
    }
    
    # Strip per-trial arrays from output to keep JSON small
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
