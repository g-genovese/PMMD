#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bd_action.py  --  Benincasa-Dowker causal-set route to the induced Einstein-Hilbert term,
                  on the PMMD icosian cut-and-project foam (the d_BS=4 construction).

This is the causal-sets-native alternative to the heat-kernel/conformal probe.  It builds the
Lorentzian causal order on the foam points (the paper's d_BS=4 infrastructure), validates the
dimension via Myrheim-Meyer, and forms the 4D Benincasa-Dowker action from interval abundances:

    S_BD/hbar = (4/sqrt6) ( N - N_0 + 9 N_1 - 16 N_2 + 8 N_3 )   -->   (1/2) \int sqrt(-g) R / ell^2,

with N_k = number of order-intervals containing exactly k elements strictly between the endpoints
(N_0 = causal links).  For flat (Minkowski) sprinkling <S_BD> -> 0 (up to BD fluctuations).

Myrheim-Meyer: for a sprinkling into a d-dim Alexandrov interval the ordering fraction is
    f(d) = Gamma(d+1) Gamma(d/2) / (4 Gamma(3d/2))      (f(2)=1/4, f(4)=1/20),
inverted to read d off the measured fraction.

WHAT THIS SESSION ESTABLISHES: the d_MM ~ 4 validation on the cut-and-project (the proven d_BS),
the N_k abundances, and S_BD vs a Poisson sprinkling.  The induced-G COEFFICIENT (the actual
1/(4 phi^10)) needs a curved background + the smeared BD operator (fluctuation control) and is the
next step, flagged below -- NOT claimed here.
"""

import argparse, json, math, sys, time
import numpy as np
from math import gamma

try:
    from foam_rigidity import e8_points, poisson_points
except Exception:
    # minimal fallback if run standalone without foam_rigidity.py on path
    def poisson_points(N, rng, dim=4):
        return rng.random((N, dim))
    e8_points = None

SQ6 = math.sqrt(6.0)


# ---------------- Myrheim-Meyer dimension ----------------
def mm_fraction(d):
    return gamma(d + 1.0) * gamma(d / 2.0) / (4.0 * gamma(1.5 * d))

def mm_dimension(r):
    """r = UNORDERED ordering fraction = (#related pairs)/(N choose 2) = 2 f(d).
       (Check: 2D causal diamond -> r = 1/2.)  Invert f(d) = r/2."""
    rr = r / 2.0
    if not (0 < rr < 0.25):
        return float("nan")
    lo, hi = 0.8, 12.0
    flo, fhi = mm_fraction(lo) - rr, mm_fraction(hi) - rr
    if flo * fhi > 0:
        return float("nan")
    for _ in range(80):                              # bisection (mm_fraction is decreasing)
        mid = 0.5 * (lo + hi); fm = mm_fraction(mid) - rr
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# ---------------- causal order / diamond / BD ----------------
def causal_matrix(P):
    """coord 0 = time; F[i,j]=True iff i strictly precedes j (future-timelike)."""
    t = P[:, 0]; x = P[:, 1:]
    dt = t[None, :] - t[:, None]                     # t_j - t_i
    x2 = (x * x).sum(1)
    sp = x2[:, None] + x2[None, :] - 2.0 * (x @ x.T)  # |x_j - x_i|^2
    np.fill_diagonal(sp, 0.0)
    F = (dt > 0) & (dt * dt > sp)
    np.fill_diagonal(F, False)
    return F

def restrict_to_diamond(P, frac=1.0):
    """Keep the Alexandrov interval between apexes (-T,0) and (+T,0); T = frac * max|t|."""
    t = P[:, 0]; r2 = (P[:, 1:] ** 2).sum(1)
    T = frac * np.abs(t).max()
    keep = (np.abs(t) < T) & ((t + T) ** 2 > r2) & ((T - t) ** 2 > r2)
    return P[keep]


def sprinkle_diamond_poisson(N, rng, T=1.0):
    """Uniform Poisson sprinkling of N points in the 4D Alexandrov interval {|x| < T-|t|}."""
    ts = []
    while len(ts) < N:
        tt = rng.uniform(-T, T, size=4 * N + 64)
        acc = rng.uniform(0, 1, size=tt.size) < (1 - np.abs(tt) / T) ** 3
        ts.extend(tt[acc].tolist())
    t = np.array(ts[:N])
    rho = T - np.abs(t)
    r = rng.uniform(0, 1, N) ** (1.0 / 3.0) * rho
    d = rng.normal(size=(N, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.column_stack([t, d * r[:, None]])

def bd_observables(P):
    F = causal_matrix(P).astype(np.float32)
    N = len(P)
    if N < 8:
        return {"N": N, "status": "too_few"}
    R = int(F.sum())
    M = F @ F                                        # M[i,j] = # elements strictly between
    inter = M[F > 0].astype(np.int64)
    N0 = int((inter == 0).sum()); N1 = int((inter == 1).sum())
    N2 = int((inter == 2).sum()); N3 = int((inter == 3).sum())
    S_BD = (4.0 / SQ6) * (N - N0 + 9 * N1 - 16 * N2 + 8 * N3)
    frac = R / (N * (N - 1) / 2.0)
    return {"N": N, "relations": R, "ordering_fraction": frac, "d_MM": mm_dimension(frac),
            "N0_links": N0, "N1": N1, "N2": N2, "N3": N3, "S_BD_sharp": S_BD,
            "_inter": inter}                          # keep interval counts for the smeared action


# ---------------- smeared BD action (Sorkin / Dowker-Glaser) ----------------
# K(n,eps) = sum_{i=0}^3 C_i C(n,i) eps^i (1-eps)^{n-i}  (= expectation of the layer coefficient
# under a binomial eps-thinning of the interval).  The smearing scale eps in (0,1] tames the
# notorious BD fluctuations: on flat Minkowski the per-sample fluctuation of S drops ~1000x at
# eps=0.1 vs the sharp operator (verified).  C = (1,-9,16,-8).
_BD4_C = np.array([1.0, -9.0, 16.0, -8.0])

def smeared_bd_action(inter, N, eps):
    """S_eps = (4/sqrt6)( -eps N + eps^2 sum_{related} K(n,eps) ), n = interval cardinalities."""
    n = inter.astype(float)
    binom = [np.ones_like(n), n, n * (n - 1) / 2.0, n * (n - 1) * (n - 2) / 6.0]
    K = np.zeros_like(n)
    for i in range(4):
        K += _BD4_C[i] * binom[i] * (eps ** i) * np.where(n >= i, (1 - eps) ** (n - i), 0.0)
    return (4.0 / SQ6) * (-eps * N + eps ** 2 * K.sum())
# CAVEAT (verified this session): the smeared operator controls fluctuations, but extracting the
# induced-gravity COEFFICIENT (the curvature term int R) from S_eps is NOT closed -- on a single
# finite realisation the cosmological/volume term (~ the density change) swamps the small quadratic
# curvature signal, because the bulk cancellation (box 1 = 0) is exact only in the sprinkling MEAN.
# A clean coefficient needs volume-preserving perturbations + large-N sprinkling-averaging.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=3000, help="target elements inside the diamond")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--foam", choices=["poisson", "e8", "both"], default="both")
    ap.add_argument("--frac", type=float, default=1.0, help="diamond half-extent / max|t|")
    ap.add_argument("--eps_smear", type=float, default=0.1, help="BD smearing scale (fluctuation control)")
    ap.add_argument("--oversample", type=float, default=6.0, help="generate this x N, then restrict")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed * 1_000_003 + 7)
    t0 = time.time()
    Ngen = int(args.N * args.oversample)

    out = {"target_N": args.N, "seed": args.seed, "frac": args.frac,
           "phi": (1 + 5 ** 0.5) / 2}
    foams = ["poisson", "e8"] if args.foam == "both" else [args.foam]
    for fm in foams:
        if fm == "poisson":
            dia = sprinkle_diamond_poisson(args.N, rng)
        else:
            if e8_points is None:
                out["e8"] = {"status": "foam_rigidity_not_importable"}; continue
            pts = e8_points(Ngen, rng); pts = pts - pts.mean(0)
            dia = restrict_to_diamond(pts, args.frac)
            if len(dia) > args.N:                    # cap: BD is O(N^2)-O(N^3); subsample
                dia = dia[rng.choice(len(dia), args.N, replace=False)]
        obs = bd_observables(dia)
        inter = obs.pop("_inter", None)
        if inter is not None:
            obs["S_BD_smeared"] = smeared_bd_action(inter, obs["N"], args.eps_smear)
            obs["eps_smear"] = args.eps_smear
        out[fm] = obs
    out["wall_s"] = round(time.time() - t0, 2)
    line = json.dumps(out)
    if args.out:
        open(args.out, "w").write(line + "\n")
    else:
        print(line)


if __name__ == "__main__":
    main()
