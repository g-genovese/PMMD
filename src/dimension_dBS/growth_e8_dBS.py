#!/usr/bin/env python3
"""
growth_e8_dBS.py  --  framework-faithful E8 growth + cut-and-project d_BS.

Generation rule (the corrected, agreed logic):
  - vertex = a Bloch sphere; balance = E8 adjacency (mutual distance sqrt2).
  - seed = one triangle (3 mutually-adjacent spheres).
  - a new vertex is born as the 4th sphere balancing an EXISTING TRIANGLE
    (adjacent to all 3 vertices of a present triangle -> closes a tetrahedron).
    Minimum 3 parents; parents = ALL present E8 neighbours at birth.
  - order is a priori undetermined: uniform-random among balanceable positions
    (born when balance happens, no cost ranking). No phase needed for d_BS.
  - causal order = growth order -> parent->child DAG.

d_BS measurement (the framework's d=4 is the 4D cut-and-project image):
  - project grown 8D points to 4D:  x_par[k] = c[2k] + c[2k+1]*phi  (golden)
  - Minkowski causality: time = component along u_t=[1,sqrt2,sqrt3,sqrt5];
    a < b iff dt>0 and dt^2 > beta^2 dx^2 (beta=1, 45-deg structural cone).
  - longest chain L(N) ~ N^(1/d); FSS over subsample N -> d_BS, extrapolate N->inf.
  - Per-trial Lmax(N) curves are saved so the merge step does the cross-trial
    large-N extrapolation. Compare against the ideal-lattice reference (~4.06).

USAGE (per server):
  python3 growth_e8_dBS.py --target-N 32000 --n-trials 12 --trial-start 0 \
      --workers 12 --seed 20260521 --output growth_cp.json
Then extrapolate Lmax(N) across all trials to large N and read d_BS(N->inf).
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

# ---------------------------------------------------------------- growth
def grow(target_N, seed, wall=1800):
    rng = np.random.default_rng(seed)
    p0 = (0,)*8; a = (2,2,0,0,0,0,0,0); b = (2,0,2,0,0,0,0,0)
    present = {}; pts = []; parents = []
    for pt in (p0, a, b):
        present[pt] = len(pts); pts.append(pt); parents.append([])
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
                A = [u for u in c['nb'] if _adj(pts[u], w)]   # present nbrs of q adjacent to w
                found = False
                for ii in range(len(A)):
                    pi = pts[A[ii]]
                    for jj in range(ii+1, len(A)):
                        if _adj(pi, pts[A[jj]]): found = True; break
                    if found: break
                c['nb'].append(wid)
                if found: c['elig'] = True; make_elig(q)   # triangle {w,u,v} in nbrs(q)
            else:
                c['nb'].append(wid)
    for i in range(len(pts)): register(i)
    t0 = time.time()
    while len(pts) < target_N and elig and time.time()-t0 < wall:
        q = elig[int(rng.integers(0, len(elig)))]
        c = cand.pop(q); drop_elig(q)
        nid = len(pts); present[q] = nid; pts.append(q); parents.append(list(c['nb']))
        register(nid)
    return pts, parents

# ---------------------------------------------------------------- cut-and-project d_BS
def project4d(pts):
    P = np.asarray(pts, float)
    return np.stack([P[:,0]+P[:,1]*PHI, P[:,2]+P[:,3]*PHI,
                     P[:,4]+P[:,5]*PHI, P[:,6]+P[:,7]*PHI], axis=1)
try:
    from numba import njit
    @njit(cache=True)
    def _lc(t, xs, beta):
        n = len(t); L = np.ones(n, np.int64); b2 = beta*beta
        for i in range(1, n):
            best = 0; ti = t[i]
            for j in range(i):
                dt = ti - t[j]
                if dt <= 0: continue
                dx2 = 0.0
                for k in range(xs.shape[1]):
                    d = xs[i,k]-xs[j,k]; dx2 += d*d
                if dt*dt > b2*dx2 and L[j] > best: best = L[j]
            L[i] = 1 + best
        return int(L.max())
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def _lc(t, xs, beta):
        n = len(t); L = np.ones(n, np.int64); b2 = beta*beta
        for i in range(1, n):
            dt = t[i]-t[:i]; dx2 = ((xs[i]-xs[:i])**2).sum(1)
            m = (dt > 0) & (dt*dt > b2*dx2)
            if m.any(): L[i] = 1 + L[:i][m].max()
        return int(L.max())
def longest_chain(p4, u_t, beta=1.0):
    t = p4 @ u_t; o = np.argsort(t); t = t[o]; xs = (p4 - np.outer(p4 @ u_t, u_t))[o]
    return _lc(np.ascontiguousarray(t), np.ascontiguousarray(xs), beta)

def cutproject_curve(pts, seed):
    par = project4d(pts); Ng = len(par); rng = np.random.default_rng(seed)
    u_t = np.array([1.0, math.sqrt(2), math.sqrt(3), math.sqrt(5)]); u_t /= np.linalg.norm(u_t)
    Ns = [n for n in [500,1000,2000,4000,8000,16000,32000,64000,128000] if n <= Ng]
    reps = {500:8,1000:6,2000:4,4000:3,8000:2,16000:2,32000:1,64000:1,128000:1}
    Lm = [float(np.mean([longest_chain(par[rng.choice(Ng,N,replace=False)], u_t, 1.0)
                         for _ in range(reps[N])])) for N in Ns]
    return Ns, Lm

def run_one(args):
    target_N, seed, wall = args
    pts, parents = grow(target_N, seed, wall)
    N = len(pts)
    nc = np.array([len(parents[i]) for i in range(3, N)])
    Ns, Lm = cutproject_curve(pts, seed)
    d_big = None
    if len(Ns) >= 3:
        A = np.vstack([np.log(Ns[-3:]), np.ones(3)]).T
        d_big = float(1/np.linalg.lstsq(A, np.log(Lm[-3:]), rcond=None)[0][0])
    return {"seed": int(seed), "N_grow": int(N), "avg_parents": float(nc.mean()),
            "max_parents": int(nc.max()), "cp_Ns": Ns, "cp_Lmax": Lm, "d_BS_big3": d_big}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-N", type=int, default=32000)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--seed", type=int, default=20260521)
    ap.add_argument("--wall", type=int, default=3600)
    ap.add_argument("--output", default="growth_cp.json")
    a = ap.parse_args()
    jobs = [(a.target_N, a.seed + a.trial_start + k, a.wall) for k in range(a.n_trials)]
    t0 = time.time()
    if a.workers > 1:
        with Pool(a.workers) as pool: res = pool.map(run_one, jobs)
    else:
        res = [run_one(j) for j in jobs]
    # cross-trial large-N FSS: average Lmax at each N over trials, fit large-N slope
    byN = defaultdict(list)
    for x in res:
        for N, L in zip(x["cp_Ns"], x["cp_Lmax"]): byN[N].append(L)
    Ns = sorted(byN); Lm = [float(np.mean(byN[N])) for N in Ns]
    d_all = float(1/np.polyfit(np.log(Ns), np.log(Lm), 1)[0]) if len(Ns) >= 2 else None
    big = [N for N in Ns if N >= 4000] or Ns[-3:]
    d_big = float(1/np.polyfit(np.log(big), np.log([np.mean(byN[N]) for N in big]), 1)[0]) if len(big) >= 2 else None
    summary = {"n_trials": a.n_trials, "numba": HAVE_NUMBA,
               "avg_parents_mean": float(np.mean([x["avg_parents"] for x in res])),
               "N_grow_mean": float(np.mean([x["N_grow"] for x in res])),
               "cp_Ns": Ns, "cp_Lmax_mean": Lm,
               "d_BS_all": d_all, "d_BS_largeN": d_big,
               "ideal_ref_d_BS_largeN": 4.06, "wall_total_s": round(time.time()-t0,1)}
    json.dump({"summary": summary, "per_trial": res}, open(a.output,"w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
