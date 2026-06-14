"""
sim_c_cutproject_dBS.py  --  PMMD Sim C (cut-and-project estimator)
===================================================================
SECOND, INDEPENDENT d_BS estimator for the DISORDERED critical-foam ensemble.

Pipeline (per trial):
  1. Load the per-trial partial-order NPZ produced by
     e8_percolation_v43_partial_order.py  --track-partial-order
     (keys: activated_vertices, activation_time, parent_snapshot, L).
  2. Reconstruct the GIANT percolating cluster (most frequent union-find root).
  3. Reconstruct the 8D E8 coordinates of every cluster vertex from its
     global index:  h = idx // L^8 ;  lex = idx % L^8 ;
                    coords = base-L digits of lex (+0.5 if h==1).
  4. CUT-AND-PROJECT 8D -> 4D via the golden-ratio (icosian) projection:
        q1 = coords[0:4], q2 = coords[4:8]
        x_par  = q1 + phi*q2      (physical 4-plane E_par)
        x_perp = q1 - phi*q2      (internal 4-plane; Galois conjugate)
     The percolation has ALREADY selected the sub-set (the cluster), so no
     extra acceptance window is applied: every cluster vertex is projected.
  5. Measure d_BS on the projected 4D point cloud with the THREE estimators
     validated on the ideal lattice (cutproject_e8_dBS.py):
        (1) box-counting (geometric)
        (2) longest-chain causal, light-cone aperture beta=1 fixed by
            c = l*/tau* (isotropic internal space, no free parameter)
        (3) finite-size extrapolation of (2) over growing subsample sizes
  6. Write per-trial JSON; merge across trials/servers with merge_sim_c.py.

This complements compute_dBS.py (which measures d_BS on the INTRINSIC 8D
activation-time partial order). Convergence of BOTH estimators on d_BS = 4
over the disordered ensemble is the Stratum-2 promotion target of
Remark rem:dBS-cutproject-v6 / Remark rem:foam-continuum-limit claim (c).

Requires: numpy, numba.
"""
import argparse, glob, json, time
import numpy as np

try:
    from numba import njit
except ImportError:
    raise SystemExit("numba required: pip install numba")

PHI = (1.0 + np.sqrt(5.0)) / 2.0


# --------------------------------------------------------------------------
# Cluster reconstruction + 8D coordinate recovery
# --------------------------------------------------------------------------
def giant_cluster_vertices(trial):
    """Return global indices of the largest percolating cluster."""
    verts = trial["activated_vertices"].astype(np.int64)
    parent = trial["parent_snapshot"].astype(np.int64)
    # parent_snapshot stores the union-find root (already path-compressed at
    # snapshot in v4.3); the giant cluster is the most frequent root.
    roots, counts = np.unique(parent, return_counts=True)
    giant_root = roots[np.argmax(counts)]
    mask = parent == giant_root
    return verts[mask], int(counts.max())


def indices_to_e8_coords(idx, L):
    """Vectorised reconstruction of 8D E8 coordinates from global indices."""
    L8 = L ** 8
    h = idx // L8
    lex = idx % L8
    coords = np.empty((idx.size, 8), dtype=np.float64)
    for k in range(8):
        coords[:, k] = lex % L
        lex //= L
    coords += 0.5 * h[:, None]          # D8+half shift for h==1
    return coords


def cut_and_project(coords):
    """Golden-ratio E8 -> 4D icosian projection. Returns x_par (N,4)."""
    q1, q2 = coords[:, 0:4], coords[:, 4:8]
    x_par = q1 + PHI * q2
    return x_par


# --------------------------------------------------------------------------
# Estimator 1: box-counting
# --------------------------------------------------------------------------
def box_counting_dim(points, n_scales=12):
    pts = points - points.min(axis=0)
    span = pts.max()
    if span <= 0:
        return float("nan")
    pts = pts / span
    scales = np.logspace(-0.1, -1.4, n_scales)
    counts = []
    for eps in scales:
        idx = np.floor(pts / eps).astype(np.int64)
        v = np.ascontiguousarray(idx).view(
            np.dtype((np.void, idx.dtype.itemsize * idx.shape[1])))
        counts.append(len(np.unique(v)))
    counts = np.array(counts, float)
    sl = slice(1, -1)
    A = np.vstack([np.log(1.0 / scales)[sl], np.ones(n_scales)[sl]]).T
    slope = np.linalg.lstsq(A, np.log(counts)[sl], rcond=None)[0][0]
    return float(slope)


# --------------------------------------------------------------------------
# Estimator 2/3: longest causal chain (structural beta=1, finite-size series)
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def longest_chain_nb(t, xs, b2):
    n = len(t)
    L = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        best = 0
        ti = t[i]
        for j in range(i):
            dt = ti - t[j]
            if dt <= 0.0:
                continue
            dx2 = 0.0
            for k in range(xs.shape[1]):
                d = xs[i, k] - xs[j, k]
                dx2 += d * d
            if dt * dt > b2 * dx2 and L[j] > best:
                best = L[j]
        L[i] = 1 + best
    return int(L.max())


def longest_chain(pts, u_t, beta=1.0):
    t = pts @ u_t
    xs = pts - np.outer(t, u_t)
    o = np.argsort(t)
    return longest_chain_nb(np.ascontiguousarray(t[o]),
                            np.ascontiguousarray(xs[o]), beta * beta)


def chain_dim_series(pts, rng, sizes, reps, u_t, beta=1.0):
    """Longest-chain dimension over growing subsample sizes."""
    N = len(pts)
    Lmean = []
    used = []
    for n, r in zip(sizes, reps):
        if n > N:
            break
        Ls = [longest_chain(pts[rng.choice(N, n, replace=False)], u_t, beta)
              for _ in range(r)]
        Lmean.append(np.mean(Ls))
        used.append(n)
    if len(used) < 2:
        return float("nan"), float("nan"), used, Lmean
    used = np.array(used, float)
    Lmean = np.array(Lmean, float)
    A_all = np.vstack([np.log(used), np.ones(len(used))]).T
    d_all = 1.0 / np.linalg.lstsq(A_all, np.log(Lmean), rcond=None)[0][0]
    k = min(3, len(used))
    A_big = np.vstack([np.log(used[-k:]), np.ones(k)]).T
    d_big = 1.0 / np.linalg.lstsq(A_big, np.log(Lmean[-k:]), rcond=None)[0][0]
    return float(d_all), float(d_big), list(used), list(Lmean)


# --------------------------------------------------------------------------
# Per-trial analysis
# --------------------------------------------------------------------------
def analyze_trial(tpath, rng, sizes, reps, max_box_points):
    trial = dict(np.load(tpath))
    L = int(trial["L"])
    target_p = float(trial["target_p"]) if "target_p" in trial else None
    cluster_idx, giant_size = giant_cluster_vertices(trial)
    if giant_size < 50:
        return {"file": tpath, "L": L, "target_p": target_p,
                "giant_size": giant_size, "status": "cluster_too_small"}
    coords = indices_to_e8_coords(cluster_idx, L)
    x_par = cut_and_project(coords)

    # time direction: irrational generic 4-vector (avoids lattice alignment)
    u_t = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0)])
    u_t /= np.linalg.norm(u_t)

    box_pts = x_par if len(x_par) <= max_box_points else \
        x_par[rng.choice(len(x_par), max_box_points, replace=False)]
    d_box = box_counting_dim(box_pts)
    d_chain_all, d_chain_big, used, Lmean = chain_dim_series(
        x_par, rng, sizes, reps, u_t, beta=1.0)

    return {"file": tpath, "L": L, "target_p": target_p,
            "giant_size": giant_size,
            "n_projected": int(len(x_par)),
            "d_box": d_box,
            "d_chain_all_N": d_chain_all,
            "d_chain_big_N": d_chain_big,
            "chain_sizes": used, "chain_Lmean": Lmean,
            "status": "ok"}


def main():
    ap = argparse.ArgumentParser(
        description="Sim C cut-and-project d_BS estimator on percolating cluster")
    ap.add_argument("--po-files", type=str, required=True,
                    help="glob for per-trial partial-order NPZ files")
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--chain-sizes", type=int, nargs="+",
                    default=[1000, 2000, 4000, 8000, 16000, 32000, 64000])
    ap.add_argument("--chain-reps", type=int, nargs="+",
                    default=[40, 30, 20, 12, 8, 4, 2])
    ap.add_argument("--max-box-points", type=int, default=200000)
    ap.add_argument("--output", type=str, default="sim_c_cutproject_results.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.po_files))
    if not files:
        raise SystemExit(f"no files match {args.po_files}")
    print(f"Sim C cut-and-project: {len(files)} per-trial files")
    rng = np.random.default_rng(args.seed)

    results = []
    t0 = time.time()
    for i, f in enumerate(files):
        r = analyze_trial(f, rng, args.chain_sizes, args.chain_reps,
                          args.max_box_points)
        results.append(r)
        if r.get("status") == "ok":
            print(f"  [{i+1}/{len(files)}] L={r['L']} giant={r['giant_size']} "
                  f"proj={r['n_projected']}  d_box={r['d_box']:.3f}  "
                  f"d_chain(bigN)={r['d_chain_big_N']:.3f}")
        else:
            print(f"  [{i+1}/{len(files)}] {r.get('status')}")

    ok = [r for r in results if r.get("status") == "ok"]
    summary = {"n_files": len(files), "n_ok": len(ok),
               "seed": args.seed, "wall_s": time.time() - t0}
    if ok:
        for key in ["d_box", "d_chain_all_N", "d_chain_big_N"]:
            vals = np.array([r[key] for r in ok], float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                summary[f"{key}_mean"] = float(vals.mean())
                summary[f"{key}_sem"] = float(vals.std(ddof=1) / np.sqrt(vals.size)) \
                    if vals.size > 1 else 0.0
    out = {"summary": summary, "per_trial": results}
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
