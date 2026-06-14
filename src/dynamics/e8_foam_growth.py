#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e8_foam_growth.py
=================
PMMD v6.0 -- test of the growth-vs-threshold question (the open computational
target: does the exact phase-balance growth rule span precisely at the graph's
structural percolation threshold p_c ?).

It answers ONE concrete, well-posed question by simulation:

    Does balance-conserving margin growth on the E8 foam graph
    span precisely at the graph's percolation threshold p_c (~ 1/172),
    or at its own balance-critical density?

Two growth dynamics are implemented and compared on the SAME E8 graph
(z = 240, the kissing graph of the E8 root lattice):

  (A) INVASION  -- the quenched, monotone control.  Each site carries an
      i.i.d. random "favourability"; growth invades the most-favourable
      margin site.  THEOREM (Wilkinson-Willemsen 1983; Chayes-Chayes-Newman
      1985): the acceptance edge converges to p_c with NO tuned parameter.
      This is a property of the invasion ALGORITHM, used here purely as a
      parameter-free numerical ESTIMATOR of the graph's structural threshold
      p_c -- it is NOT the framework's dynamics (PMMD v6.0 reads p_c as a
      structural invariant of the E8 foam graph, crossed by the tetrahedral
      growth).  On the HPC it tightens the sandbox check (10k sites ->
      ~0.0057) to millions of sites.

  (B) PHASE-BALANCE -- the framework's actual (cooperative/conservative) rule:
      a Z3 phase per vertex; a margin vertex is admitted with a phase that
      keeps every local triangle balanced (1 + omega + omega^2 = 0, i.e.
      rainbow triangles) and does NOT destroy the pre-existing balance.
      Unbalanced growth is impossible (a vertex attached through an open
      face leaves the site unbalanced) -- growth proceeds vertex by vertex
      through balanced completions, exactly the "rise to the tetrahedron"
      simplex structure.  The OPEN question is whether the *spanning
      density* of this rule coincides with the graph's structural p_c (as
      measured by the invasion control), or sits at its own
      balance-critical density.

      NOTE: rule (B) is the best faithful reconstruction of the framework's
      "tetrahedral-balance vertex admission" CARRYING PHASES (the published
      growth sim used only the geometric skeleton and measured d_BS, not p_c).
      The admission predicate is isolated in `phase_balance_admissible()` so
      it can be replaced verbatim by the framework's exact rule.

Observables (per trial, then aggregated over trials / ranks):
  * invasion: acceptance-edge percentiles  -> should converge to p_c
  * balance : acceptance fraction at the frontier vs cluster size
              + avalanche-size distribution (criticality diagnostic)
  * both    : finite-size convergence of the above

Parallelism:
  * MPI (mpi4py) across the 3 servers: each rank runs independent trials with
    distinct seeds; rank 0 aggregates.  10 GbE is ample (only scalars/short
    arrays are reduced).
  * Falls back to multiprocessing, then to serial, if mpi4py is absent
    (e.g. the Windows workstation).  On the workstation use --serial or just
    run without mpirun.

Hardware sizing (from the user's setup):
  * Server C (2x E5-2690v4, 256 GB): can reach ~5-20 M invaded sites/trial.
    Memory ~ 250-400 bytes/site (dict of int8-tuples + phase + neighbour set
    for frontier only); 256 GB comfortably holds >1e8 frontier+invaded.
  * 3 servers x ~28 cores via MPI -> ~80 ranks -> thousands of trials.
  * Workstation (Ryzen 9950X, Windows): run serial/multiprocessing for the
    medium sizes and the analysis/plots.

Usage
-----
  # single machine (workstation), serial:
  python e8_foam_growth.py --mode both --max-sites 200000 --trials 8 --serial

  # one server, multiprocessing over local cores:
  python e8_foam_growth.py --mode both --max-sites 2000000 --trials 56

  # 3 servers via MPI (hostfile lists the 3 nodes):
  mpirun -np 84 --hostfile hosts python e8_foam_growth.py \
         --mode both --max-sites 5000000 --trials 840 --out run_e8

Output
------
  <out>_rank<k>.json   per-rank raw results
  <out>_summary.json   aggregated (rank 0): p_c estimates, convergence,
                       balance-rule acceptance density + verdict
"""

import argparse, json, os, sys, time, heapq, random
from itertools import combinations, product

import numpy as np

# ----------------------------------------------------------------------------
# E8 graph: the 240 minimal vectors (norm^2 = 2), in x2-integer representation
# so that lattice points are exact integer 8-tuples and hashing is exact.
# ----------------------------------------------------------------------------
def minimal_vectors_x2():
    vs = []
    # D8 roots x2: (+-2, +-2, 0^6)  -> 112
    for i, j in combinations(range(8), 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = [0]*8; v[i] = si; v[j] = sj
                vs.append(tuple(v))
    # half-integer x2: (+-1)^8 with an even number of minus signs -> 128
    for signs in product((1, -1), repeat=8):
        if signs.count(-1) % 2 == 0:
            vs.append(tuple(signs))
    assert len(vs) == 240, len(vs)
    return vs

MV = minimal_vectors_x2()
MV_SET = set(MV)            # for O(1) adjacency tests

def neighbours(pt):
    return [tuple(pt[k] + v[k] for k in range(8)) for v in MV]

def adjacent(x, y):
    """True iff x,y are nearest neighbours on E8 (difference is a minimal vector)."""
    return tuple(y[k] - x[k] for k in range(8)) in MV_SET

W3 = [1+0j, complex(np.cos(2*np.pi/3), np.sin(2*np.pi/3)),
              complex(np.cos(4*np.pi/3), np.sin(4*np.pi/3))]

P_C_REF = 1.0/172.55   # framework's FSS-measured E8 foam p_c (~0.005795)
P_C_BETHE = 1.0/239.0  # tree estimate

# ----------------------------------------------------------------------------
# (A) INVASION PERCOLATION on E8  -- quenched control: parameter-free p_c estimator
# ----------------------------------------------------------------------------
def invasion(nmax, seed):
    rng = np.random.default_rng(seed)
    weight = {}
    def w(pt):
        x = weight.get(pt)
        if x is None:
            x = rng.random(); weight[pt] = x
        return x
    start = (0,)*8
    visited = {start}
    heap = []
    for nb in neighbours(start):
        heapq.heappush(heap, (w(nb), nb))
    acc = []
    while heap and len(acc) < nmax:
        wt, pt = heapq.heappop(heap)
        if pt in visited:
            continue
        visited.add(pt); acc.append(wt)
        for nb in neighbours(pt):
            if nb not in visited:
                heapq.heappush(heap, (w(nb), nb))
    acc = np.asarray(acc)
    tail = acc[len(acc)//2:]
    return {
        "n": int(len(acc)),
        "edge_p99": float(np.percentile(tail, 99)),
        "edge_p995": float(np.percentile(tail, 99.5)),
        "edge_p999": float(np.percentile(tail, 99.9)),
        "max": float(acc.max()),
    }

# ----------------------------------------------------------------------------
# (B) PHASE-BALANCE GROWTH on E8 -- the framework's cooperative/conservative
# rule, carrying Z3 phases.  *** REPLACE phase_balance_admissible() with the
# framework's exact tetrahedral-balance predicate to test the real rule. ***
#
# Rule implemented here ("rainbow-triangle / balance-preserving"):
#   A margin vertex v with invaded neighbours N(v) is admissible with phase c
#   iff  for every pair {x,y} in N(v) that are mutually adjacent (a triangle
#   v-x-y), the three phases {c, ph[x], ph[y]} are all distinct (the triangle
#   is balanced, 1+omega+omega^2 = 0).  Such a c may not exist (e.g. a 4-clique
#   cannot be rainbow-3-coloured -> the "rise to a higher simplex/dimension"
#   frustration); then v is rejected.  Among admissible c, pick the one giving
#   the smallest local phase-sum magnitude (most balanced).
# ----------------------------------------------------------------------------
def phase_balance_admissible(v, invaded_phase, adj_invaded):
    """Return a valid phase c in {0,1,2} or None.
    adj_invaded: list of invaded neighbours of v.
    invaded_phase: dict site->phase. Also needs adjacency among neighbours."""
    N = adj_invaded
    if not N:
        return None
    # edges among invaded neighbours of v (triangles v-x-y), via O(1) adjacency
    tri_pairs = [(N[i], N[j]) for i in range(len(N)) for j in range(i+1, len(N))
                 if adjacent(N[i], N[j])]
    best_c, best_val = None, None
    for c in range(3):
        ok = True
        for (x, y) in tri_pairs:
            px, py = invaded_phase[x], invaded_phase[y]
            if len({c, px, py}) != 3:   # not rainbow -> triangle unbalanced
                ok = False; break
        if not ok:
            continue
        s = W3[c] + sum(W3[invaded_phase[u]] for u in N)
        val = abs(s)
        if best_val is None or val < best_val:
            best_val = val; best_c = c
    return best_c

def balance_growth(nmax, seed, record_avalanches=True):
    rng = random.Random(seed)
    start = (0,)*8
    invaded_phase = {start: 0}
    invaded = {start}
    # frontier: margin site -> set of invaded neighbours
    frontier = {}
    for nb in neighbours(start):
        frontier.setdefault(nb, set()).add(start)
    acc_frac_hist = []
    avalanches = []
    sweep = 0
    # finite-size checkpoints: instantaneous acceptance fraction vs cluster size
    checkpoints = [int(round(nmax * f)) for f in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)]
    ckpt_idx = 0
    fs_trend = []   # list of (size, acc_frac_recent)
    while len(invaded) < nmax and frontier:
        sweep += 1
        offered = list(frontier.keys())
        rng.shuffle(offered)
        admitted = 0
        for v in offered:
            if v in invaded:
                continue
            N = list(frontier.get(v, ()))
            c = phase_balance_admissible(v, invaded_phase, N)
            if c is None:
                continue
            invaded.add(v); invaded_phase[v] = c; admitted += 1
            for nb in neighbours(v):
                if nb not in invaded:
                    frontier.setdefault(nb, set()).add(v)
            frontier.pop(v, None)
            if len(invaded) >= nmax:
                break
        offered_n = max(1, len(offered))
        acc_frac_hist.append(admitted / offered_n)
        if record_avalanches:
            avalanches.append(admitted)
        # record finite-size trend at checkpoints
        while ckpt_idx < len(checkpoints) and len(invaded) >= checkpoints[ckpt_idx]:
            recent = acc_frac_hist[-max(1, len(acc_frac_hist)//5):]
            fs_trend.append([int(len(invaded)), float(np.mean(recent))])
            ckpt_idx += 1
        if admitted == 0:
            break
    h = np.asarray(acc_frac_hist) if acc_frac_hist else np.array([0.0])
    return {
        "n": int(len(invaded)),
        "sweeps": int(sweep),
        "acc_frac_early": float(h[:max(1, len(h)//5)].mean()),
        "acc_frac_late": float(h[-max(1, len(h)//5):].mean()),
        "acc_frac_min": float(h.min()),
        "fs_trend": fs_trend,   # [[size, acc_frac], ...] -> does it converge to p_c?
        "avalanche_mean": float(np.mean(avalanches)) if avalanches else 0.0,
        "avalanche_max": int(np.max(avalanches)) if avalanches else 0,
    }

# ----------------------------------------------------------------------------
# Trial driver
# ----------------------------------------------------------------------------
def run_trials(mode, max_sites, seeds):
    out = {"invasion": [], "balance": []}
    for s in seeds:
        if mode in ("invasion", "both"):
            out["invasion"].append(invasion(max_sites, s))
        if mode in ("balance", "both"):
            out["balance"].append(balance_growth(max_sites, s))
    return out

def aggregate(all_results):
    agg = {"p_c_ref": P_C_REF, "p_c_bethe": P_C_BETHE}
    inv = [r for chunk in all_results for r in chunk.get("invasion", [])]
    bal = [r for chunk in all_results for r in chunk.get("balance", [])]
    if inv:
        edge = np.array([r["edge_p995"] for r in inv])
        agg["invasion"] = {
            "trials": len(inv),
            "mean_sites": float(np.mean([r["n"] for r in inv])),
            "acceptance_edge_mean": float(edge.mean()),
            "acceptance_edge_std": float(edge.std()),
            "vs_p_c_ref_ratio": float(edge.mean() / P_C_REF),
            "verdict": ("consistent with percolation p_c"
                        if abs(edge.mean() - P_C_REF) < 3*edge.std()/max(1,len(inv))**0.5 + 0.0006
                        else "deviates from p_c"),
        }
    if bal:
        late = np.array([r["acc_frac_late"] for r in bal])
        ratio = float(late.mean() / P_C_REF)
        # aggregate finite-size trend (size -> acc_frac), averaged over trials
        from collections import defaultdict
        acc_by_size = defaultdict(list)
        for r in bal:
            for size, af in r.get("fs_trend", []):
                acc_by_size[size].append(af)
        trend = sorted([[s, float(np.mean(v))] for s, v in acc_by_size.items()])
        decreasing = (len(trend) >= 2 and trend[-1][1] < trend[0][1])
        if ratio < 1.5:
            verdict = "spans at ~p_c (the graph's structural threshold)"
        elif ratio < 10:
            verdict = (f"spans at a small critical density ~{ratio:.1f}x p_c; "
                       + ("trend DECREASES with size -> may converge to p_c (check at HPC scale)"
                          if decreasing else
                          "trend flat -> own balance-critical density, distinct from percolation p_c"))
        else:
            verdict = "supercritical (dense) -- balance alone does NOT give p_c"
        agg["balance"] = {
            "trials": len(bal),
            "mean_sites": float(np.mean([r["n"] for r in bal])),
            "acceptance_fraction_late_mean": float(late.mean()),
            "acceptance_fraction_late_std": float(late.std()),
            "vs_p_c_ref_ratio": ratio,
            "finite_size_trend": trend,   # [[size, acc_frac], ...]
            "trend_decreasing_toward_p_c": bool(decreasing),
            "verdict": verdict,
            "note": ("acceptance fraction is the order parameter. The decisive "
                     "datum is finite_size_trend: if acc_frac DECREASES toward "
                     "p_c_ref as cluster size grows, the conservative balance "
                     "dynamics spans at the graph's percolation p_c; if it "
                     "plateaus above p_c, it sits at its own "
                     "balance-critical density (still criticality, but not the "
                     "geometric percolation threshold). Run to the largest "
                     "feasible --max-sites to discriminate."),
        }
    return agg

# ----------------------------------------------------------------------------
# Parallel orchestration: MPI -> multiprocessing -> serial
# ----------------------------------------------------------------------------
def split_seeds(n_trials, base_seed, n_parts, part):
    seeds = [base_seed + i for i in range(n_trials)]
    return seeds[part::n_parts]

def main():
    ap = argparse.ArgumentParser(description="E8 foam-growth -> p_c spanning test")
    ap.add_argument("--mode", choices=["invasion", "balance", "both"], default="both")
    ap.add_argument("--max-sites", type=int, default=200000)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--out", default="run_e8")
    ap.add_argument("--serial", action="store_true", help="force serial (no MPI/mp)")
    args = ap.parse_args()

    # ---- MPI path ----
    comm = None
    if not args.serial:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except Exception:
            comm = None

    t0 = time.time()
    if comm is not None and comm.Get_size() > 1:
        rank, size = comm.Get_rank(), comm.Get_size()
        my_seeds = split_seeds(args.trials, args.base_seed, size, rank)
        my = run_trials(args.mode, args.max_sites, my_seeds)
        with open(f"{args.out}_rank{rank}.json", "w") as f:
            json.dump(my, f)
        gathered = comm.gather(my, root=0)
        if rank == 0:
            agg = aggregate(gathered)
            agg["wall_seconds"] = time.time() - t0
            agg["config"] = vars(args); agg["mpi_ranks"] = size
            with open(f"{args.out}_summary.json", "w") as f:
                json.dump(agg, f, indent=2)
            print(json.dumps(agg, indent=2))
        return

    # ---- multiprocessing path (single machine, many cores) ----
    if not args.serial:
        try:
            import multiprocessing as mp
            ncpu = mp.cpu_count()
            if ncpu > 1 and args.trials > 1:
                seeds = [args.base_seed + i for i in range(args.trials)]
                chunks = [seeds[i::ncpu] for i in range(ncpu)]
                chunks = [c for c in chunks if c]
                with mp.Pool(len(chunks)) as pool:
                    results = pool.starmap(
                        run_trials, [(args.mode, args.max_sites, c) for c in chunks])
                agg = aggregate(results)
                agg["wall_seconds"] = time.time() - t0
                agg["config"] = vars(args); agg["workers"] = len(chunks)
                with open(f"{args.out}_summary.json", "w") as f:
                    json.dump(agg, f, indent=2)
                print(json.dumps(agg, indent=2))
                return
        except Exception as e:
            print(f"[mp fallback -> serial] {e}", file=sys.stderr)

    # ---- serial path (workstation / debugging) ----
    seeds = [args.base_seed + i for i in range(args.trials)]
    res = run_trials(args.mode, args.max_sites, seeds)
    agg = aggregate([res])
    agg["wall_seconds"] = time.time() - t0
    agg["config"] = vars(args)
    with open(f"{args.out}_summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
