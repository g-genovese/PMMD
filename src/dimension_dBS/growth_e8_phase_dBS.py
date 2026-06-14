#!/usr/bin/env python3
"""
growth_e8_phase_dBS.py -- growth + cut-and-project d_BS WITH the CP1 phase.

Phase = the qubit's LOCAL TIME-ARROW DIRECTION (2 DOF, confirmed reading):
  each vertex carries a Bloch vector n(v) in S^2 (the CP1 state), propagated
  COHERENTLY along the growth DAG (Pancharatnam-like transport):
      n(child) = normalize( mean(n(parents)) + sigma_n * N(0,I_3) ).
  The local time-axis is the global u_t tilted toward n(v) by strength alpha:
      u_axis(v) = normalize( u_t + alpha * (n . e_basis) ),
  with {e1,e2,e3} an orthonormal basis of the 3-space orthogonal to u_t.
  alpha = tan(tilt angle): alpha=0 -> no tilt (baseline); 1 -> 45 deg; 2 -> 63 deg.

Causal relation a<b (longest chain) uses the EARLIER vertex's local cone, with
global processing order = projection onto u_t:
      d = b-a ; dt = d . u_axis(a) ; dx2 = |d|^2 - dt^2 ;  a<b iff dt>0 & dt^2 > beta^2 dx2.
alpha=0 reproduces growth_e8_dBS.py exactly.

CONTROL: an INCOHERENT Bloch field (random unit vectors, no DAG propagation) is
also measured -> tests that it is the COHERENCE of the phase that yields d=4.

Per trial we grow ONCE and measure every (mode, alpha) setting on the same foam.
Each setting's Lmax(N) curve is saved for cross-trial large-N extrapolation,
exactly as growth_e8_dBS.py. Compare coherent (should stay ~4) vs incoherent
(should collapse toward d~1) vs baseline.

USAGE (per server):
  python3 growth_e8_phase_dBS.py --target-N 128000 --n-trials 12 --trial-start 0 \
      --workers 12 --seed 20260521 --output phase_cp.json
Tunables: --alphas "0.5,1.0" (coherent) --inc-alphas "1.0" (incoherent)
          --sigma-n 0.2 (coherence noise) --max-cp-N 128000 --beta 1.0
"""
import argparse, json, math, time
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np

PHI = (1 + math.sqrt(5)) / 2

def e8_roots_2x():
    R = []
    for i, j in combinations(range(8), 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = [0]*8; v[i] = si; v[j] = sj; R.append(tuple(v))
    for b in range(256):
        s = [(-1 if (b >> k) & 1 else 1) for k in range(8)]
        if s.count(-1) % 2 == 0: R.append(tuple(s))
    return R
ROOTS = e8_roots_2x(); ROOTSET = set(ROOTS)
def _add(p, r): return tuple(p[k] + r[k] for k in range(8))
def _adj(p, q): return tuple(p[k] - q[k] for k in range(8)) in ROOTSET

# ---------------------------------------------------------------- growth + coherent Bloch
def grow_bloch(target_N, seed, sigma_n=0.2, wall=3600):
    rng = np.random.default_rng(seed)
    p0 = (0,)*8; a = (2,2,0,0,0,0,0,0); b = (2,0,2,0,0,0,0,0)
    present = {}; pts = []; parents = []; bloch = []
    seedn = np.array([0.0, 0.0, 1.0])
    for pt in (p0, a, b):
        present[pt] = len(pts); pts.append(pt); parents.append([]); bloch.append(seedn.copy())
    cand = {}; elig = []; elig_pos = {}
    def make_elig(q): elig_pos[q] = len(elig); elig.append(q)
    def drop_elig(q):
        i = elig_pos.pop(q); last = elig[-1]; elig[i] = last; elig_pos[last] = i; elig.pop()
    def register(wid):
        w = pts[wid]
        for r in ROOTS:
            q = _add(w, r)
            if q in present: continue
            c = cand.get(q)
            if c is None: c = {'nb': [], 'elig': False}; cand[q] = c
            if not c['elig']:
                A = [u for u in c['nb'] if _adj(pts[u], w)]; found = False
                for ii in range(len(A)):
                    pi = pts[A[ii]]
                    for jj in range(ii+1, len(A)):
                        if _adj(pi, pts[A[jj]]): found = True; break
                    if found: break
                c['nb'].append(wid)
                if found: c['elig'] = True; make_elig(q)
            else:
                c['nb'].append(wid)
    for i in range(len(pts)): register(i)
    t0 = time.time()
    while len(pts) < target_N and elig and time.time()-t0 < wall:
        q = elig[int(rng.integers(0, len(elig)))]
        c = cand.pop(q); drop_elig(q)
        nid = len(pts); present[q] = nid; pts.append(q); par = list(c['nb']); parents.append(par)
        m = np.mean([bloch[p] for p in par], axis=0) + sigma_n * rng.standard_normal(3)
        nrm = np.linalg.norm(m); bloch.append(m/nrm if nrm > 1e-9 else seedn.copy())
        register(nid)
    return pts, parents, np.asarray(bloch)

# ---------------------------------------------------------------- tilted longest chain
def project4d(pts):
    P = np.asarray(pts, float)
    return np.stack([P[:,0]+P[:,1]*PHI, P[:,2]+P[:,3]*PHI,
                     P[:,4]+P[:,5]*PHI, P[:,6]+P[:,7]*PHI], axis=1)
try:
    from numba import njit
    @njit(cache=True)
    def _lc_tilt(p4, axes, beta):
        n = len(p4); L = np.ones(n, np.int64); b2 = beta*beta
        for i in range(1, n):
            best = 0
            for j in range(i):
                dt = 0.0
                for k in range(4): dt += (p4[i,k]-p4[j,k])*axes[j,k]
                if dt <= 0: continue
                nrm2 = 0.0
                for k in range(4):
                    d = p4[i,k]-p4[j,k]; nrm2 += d*d
                dx2 = nrm2 - dt*dt
                if dt*dt > b2*dx2 and L[j] > best: best = L[j]
            L[i] = 1 + best
        return int(L.max())
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def _lc_tilt(p4, axes, beta):
        n = len(p4); L = np.ones(n, np.int64); b2 = beta*beta
        for i in range(1, n):
            d = p4[i]-p4[:i]; dt = (d*axes[:i]).sum(1)
            dx2 = (d*d).sum(1) - dt*dt
            m = (dt > 0) & (dt*dt > b2*dx2)
            if m.any(): L[i] = 1 + L[:i][m].max()
        return int(L.max())

def make_axes(bloch, u_t, e_basis, alpha):
    if alpha == 0.0: return np.tile(u_t, (len(bloch), 1))
    ax = u_t[None,:] + alpha*(bloch @ e_basis)
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    return ax

def lc_tilt(p4, bloch, u_t, e_basis, alpha, beta=1.0):
    tg = p4 @ u_t; o = np.argsort(tg)
    p4s = np.ascontiguousarray(p4[o])
    axes = np.ascontiguousarray(make_axes(bloch[o], u_t, e_basis, alpha))
    return _lc_tilt(p4s, axes, beta)

def setting_curve(p4, bloch, u_t, e_basis, alpha, beta, Ng, Ns, reps, seed):
    rng = np.random.default_rng(seed)
    Lm = []
    for N in Ns:
        vals = [lc_tilt(p4[idx], bloch[idx], u_t, e_basis, alpha, beta)
                for idx in (rng.choice(Ng, N, replace=False) for _ in range(reps[N]))]
        Lm.append(float(np.mean(vals)))
    return Lm

# ---------------------------------------------------------------- per-trial driver
def run_one(args):
    target_N, seed, sigma_n, alphas, inc_alphas, beta, max_cp, wall = args
    pts, parents, bl_coh = grow_bloch(target_N, seed, sigma_n, wall)
    Ng = len(pts); p4 = project4d(pts)
    rng = np.random.default_rng(seed + 7)
    g = rng.standard_normal((Ng, 3)); bl_inc = g / np.linalg.norm(g, axis=1, keepdims=True)
    u_t = np.array([1.0, math.sqrt(2), math.sqrt(3), math.sqrt(5)]); u_t /= np.linalg.norm(u_t)
    Q, _ = np.linalg.qr(np.column_stack([u_t, np.eye(4)])); e_basis = Q[:, 1:4].T  # (3,4)
    Ns = [n for n in [500,1000,2000,4000,8000,16000,32000,64000,128000,256000]
          if n <= Ng and n <= max_cp]
    reps = {500:8,1000:6,2000:4,4000:3,8000:2,16000:2,32000:1,64000:1,128000:1,256000:1}
    settings = {}
    settings["baseline"] = setting_curve(p4, bl_coh, u_t, e_basis, 0.0, beta, Ng, Ns, reps, seed)
    for al in alphas:
        settings[f"coh_a{al}"] = setting_curve(p4, bl_coh, u_t, e_basis, al, beta, Ng, Ns, reps, seed+11)
    for al in inc_alphas:
        settings[f"inc_a{al}"] = setting_curve(p4, bl_inc, u_t, e_basis, al, beta, Ng, Ns, reps, seed+13)
    nc = np.array([len(parents[i]) for i in range(3, Ng)])
    return {"seed": int(seed), "N_grow": int(Ng), "avg_parents": float(nc.mean()),
            "cp_Ns": Ns, "settings": settings}

def _localds(Ns, Lm):
    return [round(math.log(2)/math.log(Lm[i]/Lm[i-1]), 3) for i in range(1, len(Ns))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-N", type=int, default=128000)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--seed", type=int, default=20260521)
    ap.add_argument("--sigma-n", type=float, default=0.2)
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--inc-alphas", default="1.0")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-cp-N", type=int, default=128000)
    ap.add_argument("--wall", type=int, default=3600)
    ap.add_argument("--output", default="phase_cp.json")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    inc_alphas = [float(x) for x in a.inc_alphas.split(",") if x.strip()]
    jobs = [(a.target_N, a.seed + a.trial_start + k, a.sigma_n, alphas, inc_alphas,
             a.beta, a.max_cp_N, a.wall) for k in range(a.n_trials)]
    t0 = time.time()
    if a.workers > 1:
        with Pool(a.workers) as pool: res = pool.map(run_one, jobs)
    else:
        res = [run_one(j) for j in jobs]
    # cross-trial aggregate per setting
    Ns = res[0]["cp_Ns"]
    agg = {}
    names = list(res[0]["settings"].keys())
    for name in names:
        byN = defaultdict(list)
        for x in res:
            for N, L in zip(x["cp_Ns"], x["settings"][name]): byN[N].append(L)
        Lm = [float(np.mean(byN[N])) for N in Ns]
        d_last = float(math.log(2)/math.log(Lm[-1]/Lm[-2])) if len(Ns) >= 2 else None
        big = [k for k in range(len(Ns)) if Ns[k] >= 32000] or list(range(max(0,len(Ns)-3), len(Ns)))
        d_big = float(1/np.polyfit(np.log([Ns[k] for k in big]),
                                   np.log([Lm[k] for k in big]), 1)[0]) if len(big) >= 2 else None
        agg[name] = {"cp_Lmax_mean": Lm, "local_d": _localds(Ns, Lm),
                     "d_last_doubling": d_last, "d_largeN_slope": d_big}
    summary = {"n_trials": a.n_trials, "numba": HAVE_NUMBA, "target_N": a.target_N,
               "sigma_n": a.sigma_n, "alphas": alphas, "inc_alphas": inc_alphas,
               "beta": a.beta, "cp_Ns": Ns,
               "avg_parents_mean": float(np.mean([x["avg_parents"] for x in res])),
               "per_setting": agg, "wall_total_s": round(time.time()-t0, 1)}
    json.dump({"summary": summary, "per_trial": res}, open(a.output, "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
