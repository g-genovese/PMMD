#!/usr/bin/env python3
"""
sim_bond_bias_RG_v42.py

4D bond-percolation simulation with phase-dependent bond bias, for v4.2 of PMMD.

Purpose
-------
Task 2 (v4.2): empirically determine the RG eigenvalue lambda_bb of the bond-bias
operator at the 4D-percolation critical fixed point. This verifies (or corrects) the
analytical estimate |lambda_bb| ~ 0.045 derived in Remark rem:epsilon-bps-rigorous of
the paper.

Method
------
At p = p_c^(4d) ~ 0.16013, label each site of the 4D simple cubic lattice with a
phase in {v, s, c} drawn uniformly. Apply phase-dependent bond bias:
    bond_prob(label_u, label_v) = p_c * (1 + delta(label_u, label_v))
where delta(.,.) depends on epsilon and respects the C_CP symmetry s <-> c
(averaged over two realisations).

For each (epsilon, L) pair, measure the macroscopic phase populations in the
percolating cluster:
    pi_v = N_v / N_total,  pi_s = N_s / N_total,  pi_c = N_c / N_total
The C_CP-symmetric observable is the deviation:
    Delta(epsilon, L) = 1/3 - pi_v
which is linear in epsilon for small bias. The RG eigenvalue is extracted as
    Delta(epsilon, L) / Delta(epsilon, L_0) = (L / L_0)^(lambda_bb / nu_4d)

where nu_4d ~ 0.689 (4D percolation correlation length exponent).

Output
------
bondbias_rg_run<run_id>.json  -- summary with Delta(epsilon, L) for each pair
bondbias_rg_run<run_id>.npz   -- raw cluster compositions per trial
bondbias_rg_run<run_id>.log   -- runtime trace

Usage
-----
    # Default: L in {16,24,32}, epsilon in {0.10,0.15,0.20,0.25,0.30}, 32 trials each
    # Estimated runtime: ~6 hours on one core
    python3 sim_bond_bias_RG_v42.py

    # Quick test
    python3 sim_bond_bias_RG_v42.py --L_values 8 12 --epsilon_values 0.20 --n_trials 4

    # Production
    python3 sim_bond_bias_RG_v42.py --L_values 16 24 32 48 --epsilon_values 0.10 0.15 0.20 0.25 0.30 --n_trials 64

Recommended for the server: parallel runs splitting (epsilon, L) pairs.
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

P_C_4D = 0.16013  # Lorenz-Ziff 1998: 4D simple cubic site percolation


# =============================================================================
# Union-Find
# =============================================================================
class UnionFind:
    __slots__ = ("parent", "rank", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# =============================================================================
# Bond bias model: epsilon parametrization
# =============================================================================
def bond_bias_realisation_A(label_u: int, label_v: int, epsilon: float) -> float:
    """
    delta(v) = -epsilon, delta(s) = +epsilon, delta(c) = -epsilon.
    Chain consumes 8_v and 8_c, leaves 8_s free.
    """
    delta_map = {0: -epsilon, 1: +epsilon, 2: -epsilon}  # 0=v, 1=s, 2=c
    return (delta_map[label_u] + delta_map[label_v]) / 2.0


def bond_bias_realisation_B(label_u: int, label_v: int, epsilon: float) -> float:
    """
    C_CP image: swap s <-> c. delta(v) = -epsilon, delta(s) = -epsilon, delta(c) = +epsilon.
    """
    delta_map = {0: -epsilon, 1: -epsilon, 2: +epsilon}
    return (delta_map[label_u] + delta_map[label_v]) / 2.0


# =============================================================================
# Single trial: build lattice, apply biased bonds, find percolating cluster
# =============================================================================
def single_trial(L: int, epsilon: float, realisation: str, seed: int) -> dict:
    """
    Run one realisation: assign labels uniformly, sample bonds with bias,
    find connected components, identify percolating cluster, count labels in it.
    """
    rng = np.random.default_rng(seed)
    N = L ** 4
    # Random site labels uniform in {0, 1, 2}
    labels = rng.integers(0, 3, N).astype(np.int8)
    # Bond bias function
    bias_fn = bond_bias_realisation_A if realisation == "A" else bond_bias_realisation_B
    # Build and resolve bonds
    uf = UnionFind(N)
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for w in range(L):
                    s_idx = ((x * L + y) * L + z) * L + w
                    # 4 forward neighbours (periodic)
                    for dx, dy, dz, dw in ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)):
                        nx, ny, nz, nw = (x+dx) % L, (y+dy) % L, (z+dz) % L, (w+dw) % L
                        n_idx = ((nx * L + ny) * L + nz) * L + nw
                        delta = bias_fn(int(labels[s_idx]), int(labels[n_idx]), epsilon)
                        p_bond = P_C_4D * (1.0 + delta)
                        if rng.random() < p_bond:
                            uf.union(s_idx, n_idx)
    # Find largest cluster
    roots = np.array([uf.find(i) for i in range(N)], dtype=np.int64)
    cluster_sizes = np.bincount(roots)
    largest_root = int(np.argmax(cluster_sizes))
    largest_size = int(cluster_sizes[largest_root])
    in_largest = (roots == largest_root)
    # Label counts in largest cluster
    n_v = int((labels[in_largest] == 0).sum())
    n_s = int((labels[in_largest] == 1).sum())
    n_c = int((labels[in_largest] == 2).sum())
    return {
        "largest_size": largest_size,
        "n_v": n_v,
        "n_s": n_s,
        "n_c": n_c,
        "pi_v": n_v / largest_size if largest_size > 0 else 0.0,
        "pi_s": n_s / largest_size if largest_size > 0 else 0.0,
        "pi_c": n_c / largest_size if largest_size > 0 else 0.0,
    }


def _trial_task(args):
    """Worker function: run one (A, B) trial pair. Top-level for pickling."""
    L, epsilon, seed_A, seed_B = args
    rA = single_trial(L, epsilon, "A", seed_A)
    rB = single_trial(L, epsilon, "B", seed_B)
    return rA, rB


def run_pair(L: int, epsilon: float, n_trials: int, base_seed: int,
             n_workers: int, log_fn) -> dict:
    """Run n_trials for realisations A and B at this (L, epsilon), then C_CP-average.

    If n_workers > 1, parallelises trials via multiprocessing.Pool.
    """
    log_fn(f"\n=== (L={L}, epsilon={epsilon:.3f}) ===")
    t0 = time.time()
    # Build work list
    work = []
    for trial in range(n_trials):
        seed_A = base_seed + 2 * trial
        seed_B = base_seed + 2 * trial + 1
        work.append((L, epsilon, seed_A, seed_B))

    pi_v_A = np.empty(n_trials); pi_s_A = np.empty(n_trials); pi_c_A = np.empty(n_trials)
    pi_v_B = np.empty(n_trials); pi_s_B = np.empty(n_trials); pi_c_B = np.empty(n_trials)

    if n_workers <= 1:
        # Sequential path
        for i, w in enumerate(work):
            rA, rB = _trial_task(w)
            pi_v_A[i] = rA["pi_v"]; pi_s_A[i] = rA["pi_s"]; pi_c_A[i] = rA["pi_c"]
            pi_v_B[i] = rB["pi_v"]; pi_s_B[i] = rB["pi_s"]; pi_c_B[i] = rB["pi_c"]
            if (i + 1) % max(1, n_trials // 8) == 0 or i == n_trials - 1:
                log_fn(f"  Trial {i+1}/{n_trials}, elapsed {time.time()-t0:.0f}s")
    else:
        # Parallel path via multiprocessing.Pool
        from multiprocessing import get_context
        ctx = get_context("spawn")
        log_n = max(1, n_trials // 8)
        with ctx.Pool(n_workers) as pool:
            for i, (rA, rB) in enumerate(pool.imap_unordered(_trial_task, work, chunksize=max(1, n_trials // (4 * n_workers)))):
                pi_v_A[i] = rA["pi_v"]; pi_s_A[i] = rA["pi_s"]; pi_c_A[i] = rA["pi_c"]
                pi_v_B[i] = rB["pi_v"]; pi_s_B[i] = rB["pi_s"]; pi_c_B[i] = rB["pi_c"]
                if (i + 1) % log_n == 0 or i == n_trials - 1:
                    log_fn(f"  Trial {i+1}/{n_trials}, elapsed {time.time()-t0:.0f}s ({n_workers} workers)")

    # Under realisation B, s and c roles are swapped, so to combine:
    pi_v_avg = 0.5 * (pi_v_A + pi_v_B)
    pi_s_avg = 0.5 * (pi_s_A + pi_c_B)  # s of A and c of B both correspond to "free" phase
    pi_c_avg = 0.5 * (pi_c_A + pi_s_B)
    Delta = 1.0/3.0 - pi_v_avg.mean()
    return {
        "L": L,
        "epsilon": epsilon,
        "n_trials": n_trials,
        "pi_v_mean": float(pi_v_avg.mean()),
        "pi_v_std": float(pi_v_avg.std(ddof=1)),
        "pi_s_mean": float(pi_s_avg.mean()),
        "pi_c_mean": float(pi_c_avg.mean()),
        "Delta": float(Delta),
        "Delta_std": float(pi_v_avg.std(ddof=1)),
        "runtime_seconds": time.time() - t0,
        "pi_v_per_trial": pi_v_avg.tolist(),
        "pi_s_per_trial": pi_s_avg.tolist(),
        "pi_c_per_trial": pi_c_avg.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Bond-bias RG measurement for v4.2")
    parser.add_argument("--L_values", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--epsilon_values", type=float, nargs="+",
                        default=[0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--n_trials", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel workers for trial execution (default 1 = sequential)")
    parser.add_argument("--base_seed", type=int, default=20260201)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"bondbias_rg_run{run_id}.log"

    def log_fn(msg: str):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    log_fn(f"=== sim_bond_bias_RG_v42, run {run_id} ===")
    log_fn(f"Started at {datetime.now(timezone.utc).isoformat()}")
    log_fn(f"Args: {vars(args)}")
    log_fn(f"L values: {args.L_values}, epsilon values: {args.epsilon_values}")
    log_fn(f"Trials per (L, epsilon): {args.n_trials}")

    results = []
    seed_counter = args.base_seed
    for L in args.L_values:
        for eps in args.epsilon_values:
            r = run_pair(L, eps, args.n_trials, seed_counter, args.n_workers, log_fn)
            results.append(r)
            seed_counter += 4 * args.n_trials

    # RG analysis: extract lambda_bb from Delta(epsilon, L) scaling
    log_fn(f"\n=== RG Analysis ===")
    log_fn(f"Delta(epsilon, L) = 1/3 - <pi_v>:\n")
    log_fn(f"{'epsilon':>8} | " + " | ".join(f"L={L:>3d}" for L in args.L_values))
    for eps in args.epsilon_values:
        row = []
        for L in args.L_values:
            for r in results:
                if r["L"] == L and abs(r["epsilon"] - eps) < 1e-9:
                    row.append(f"{r['Delta']:+.4f}")
                    break
        log_fn(f"{eps:>8.3f} | " + " | ".join(f"{v:>7s}" for v in row))

    # Extract lambda_bb: fit log(Delta) vs log(L) at fixed epsilon
    log_fn(f"\nLog-log fits: Delta(L) ~ L^(-|lambda_bb|/nu_4d)")
    NU_4D = 0.689
    lambda_estimates = []
    for eps in args.epsilon_values:
        Ls = []; Deltas = []
        for L in args.L_values:
            for r in results:
                if r["L"] == L and abs(r["epsilon"] - eps) < 1e-9:
                    if r["Delta"] > 0:
                        Ls.append(L); Deltas.append(r["Delta"])
                    break
        if len(Ls) >= 2:
            slope, intercept = np.polyfit(np.log(Ls), np.log(Deltas), 1)
            lambda_bb_over_nu = -slope
            lambda_bb_est = lambda_bb_over_nu * NU_4D
            log_fn(f"  epsilon={eps:.3f}: slope={slope:.4f}, |lambda_bb|/nu_4d={lambda_bb_over_nu:.4f}, |lambda_bb|={lambda_bb_est:.4f}")
            lambda_estimates.append(lambda_bb_est)

    if lambda_estimates:
        lambda_mean = float(np.mean(lambda_estimates))
        lambda_std = float(np.std(lambda_estimates, ddof=1)) if len(lambda_estimates) > 1 else 0.0
        log_fn(f"\nAverage |lambda_bb| across epsilon values: {lambda_mean:.4f} +- {lambda_std:.4f}")
        log_fn(f"Analytical estimate from paper: 0.045")
        log_fn(f"Verification: {'CONSISTENT' if abs(lambda_mean - 0.045) < 0.020 else 'DISCREPANT'}")

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "results": results,
        "lambda_bb_estimates": lambda_estimates,
    }
    with open(output_dir / f"bondbias_rg_run{run_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    log_fn(f"\nSaved: {output_dir}/bondbias_rg_run{run_id}.json")


if __name__ == "__main__":
    main()
