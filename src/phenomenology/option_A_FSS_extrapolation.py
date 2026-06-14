"""
option_A_FSS_extrapolation.py
==============================
FSS extrapolation of p_c (E_8 percolation) to L=infinity,
discriminating between f_c^alg = 183 (Spin(10)-complete)
and f_c^IR = 176.25 (sub-seesaw EFT).

Data: per-trial p_c estimates from E_8 percolation simulations
      at L in {8, 10, 12}.

Result (Stratum 1):
  - f_c^alg = 183 REJECTED at > 8σ in all physically motivated FSS forms
  - f_c^IR = 176.25 CONSISTENT (≤ 2.2σ across forms)
"""

import json
import numpy as np
from scipy.optimize import curve_fit

def per_trial_pc(sum_sq_all, S_max_all, ps, N):
    """Per-trial p_c from peak of susceptibility chi = (sum_sq - S_max^2)/N."""
    chi_all = (sum_sq_all - S_max_all**2) / N
    return ps[chi_all.argmax(axis=1)]

def load_dataset(json_path, npz_path):
    with open(json_path) as f: meta = json.load(f)
    with np.load(npz_path) as cv:
        ps = cv['ps']; S_max_all = cv['S_max_all']; sum_sq_all = cv['sum_sq_all']
    pc_pt = per_trial_pc(sum_sq_all, S_max_all, ps, meta['N'])
    return pc_pt, meta

def fss_fit(Ls, pcs, sems, x):
    """Fit p_c(L) = p_c(infty) + a/L^x. Returns (pc_inf, pc_err, chi2)."""
    def f(L, pc_inf, a): return pc_inf + a / L**x
    popt, pcov = curve_fit(f, Ls, pcs, sigma=sems, absolute_sigma=True, p0=[0.0057, 0.01])
    perr = np.sqrt(np.diag(pcov))
    chi2 = np.sum(((pcs - f(Ls, *popt))/sems)**2)
    return popt[0], perr[0], chi2

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 7:
        print("Usage: option_A_FSS_extrapolation.py L8.json L8.npz L10.json L10.npz L12.json L12.npz")
        sys.exit(1)
    
    L_vals = [8, 10, 12]
    data = {}
    for i, L in enumerate(L_vals):
        pc_pt, meta = load_dataset(sys.argv[1+2*i], sys.argv[2+2*i])
        data[L] = {'pcs': pc_pt, 'mean': pc_pt.mean(), 
                   'sem': pc_pt.std(ddof=1)/np.sqrt(len(pc_pt))}
    
    Ls = np.array(L_vals)
    pcs = np.array([data[L]['mean'] for L in L_vals])
    sems = np.array([data[L]['sem'] for L in L_vals])
    
    print(f"{'L':>3} {'trials':>7} {'p_c ± SEM':>22} {'1/p_c':>9}")
    for L in L_vals:
        d = data[L]
        print(f"{L:>3} {len(d['pcs']):>7} {d['mean']:>11.7f}±{d['sem']:.7f} {1/d['mean']:>9.2f}")
    
    p_alg = 1/183; p_IR = 1/176.25
    print(f"\nTargets: f_c^alg = 183 → p = {p_alg:.7f}, f_c^IR = 176.25 → p = {p_IR:.7f}\n")
    
    print(f"{'Form':<25} {'1/p_c(infty)':>16} {'σ vs alg':>10} {'σ vs IR':>10}")
    for x, lbl in [(2.0, "MF (x=2)"), (4/3, "DIV (x=4/3)"), (1.0, "linear (x=1)")]:
        pc_inf, pc_err, chi2 = fss_fit(Ls, pcs, sems, x)
        sigma_alg = (pc_inf - p_alg) / pc_err
        sigma_IR  = (pc_inf - p_IR) / pc_err
        print(f"{lbl:<25} {1/pc_inf:>10.2f}±{pc_err/pc_inf**2:.2f}  "
              f"{sigma_alg:>+8.2f}  {sigma_IR:>+8.2f}")
