"""
merge_sim_c.py  --  PMMD Sim C aggregation + finite-size scaling
================================================================
Merges per-trial cut-and-project d_BS results (sim_c_cutproject_dBS.py output)
across trials and servers, groups by (L, p), and performs the finite-size
extrapolation L -> infinity for each estimator.

Foam regime: d_BS is measured on the connected percolating cluster in the
foam-connected regime (p just above the finite-L percolation threshold). The
asymptotic value is obtained by extrapolating the per-L ensemble means to
1/L -> 0 (linear fit in 1/L), which removes the low-side finite-size bias of
the longest-chain and box-counting estimators.

USAGE:
  python merge_sim_c.py --inputs "sim_c_cp_*.json" --output sim_c_merged.json

Optionally combine with the intrinsic estimator (compute_dBS.py / merge_dBS_results.py)
by passing --intrinsic-merged dBS_merged.json ; the script will print both
estimators side by side for the convergence check.
"""
import argparse, glob, json
import numpy as np
from collections import defaultdict


def load_per_trial(paths):
    """Collect per-trial ok records, tagged by (L, p) if present in filename/record."""
    recs = []
    for p in paths:
        with open(p) as fh:
            blob = json.load(fh)
        for r in blob.get("per_trial", []):
            if r.get("status") == "ok":
                recs.append(r)
    return recs


def ensemble_by_L(recs, key):
    """Group estimator `key` by L; return {L: (mean, sem, n)}."""
    by_L = defaultdict(list)
    for r in recs:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            by_L[int(r["L"])].append(v)
    out = {}
    for L, vals in sorted(by_L.items()):
        a = np.array(vals, float)
        out[L] = (float(a.mean()),
                  float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
                  a.size)
    return out


def group_by_p(recs):
    """Split records by target_p (rounded to 6 dp). Returns {p: [recs]}."""
    by_p = defaultdict(list)
    for r in recs:
        p = r.get("target_p")
        key = round(p, 6) if p is not None else None
        by_p[key].append(r)
    return by_p


def fss_extrapolate(by_L):
    """Linear fit of d(L) vs 1/L, extrapolated to 1/L -> 0. Weighted by 1/sem^2."""
    Ls = sorted(by_L)
    if len(Ls) < 2:
        return None
    x = np.array([1.0 / L for L in Ls])
    y = np.array([by_L[L][0] for L in Ls])
    sem = np.array([max(by_L[L][1], 1e-6) for L in Ls])
    w = 1.0 / sem ** 2
    # weighted least squares y = a + b x ; intercept a = value at 1/L -> 0
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    beta, *_ = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)
    a, b = beta
    # intercept uncertainty from weighted covariance
    cov = np.linalg.inv(X.T @ W @ X)
    a_err = float(np.sqrt(cov[0, 0]))
    return {"d_inf": float(a), "d_inf_err": a_err, "slope": float(b),
            "L_used": Ls, "d_at_L": {L: by_L[L][0] for L in Ls}}


def main():
    ap = argparse.ArgumentParser(description="Merge Sim C cut-and-project results + FSS")
    ap.add_argument("--inputs", type=str, required=True,
                    help="glob for sim_c_cutproject_dBS.py output JSONs")
    ap.add_argument("--intrinsic-merged", type=str, default=None,
                    help="optional merged JSON from the intrinsic estimator")
    ap.add_argument("--output", type=str, default="sim_c_merged.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise SystemExit(f"no files match {args.inputs}")
    recs = load_per_trial(paths)
    print(f"Loaded {len(recs)} ok trials from {len(paths)} files")

    by_p = group_by_p(recs)
    report = {"n_files": len(paths), "n_trials": len(recs), "p_scan": {}}
    key = "d_chain_big_N"   # robust estimator (longest-chain, finite-size corrected)

    print(f"\n{'='*60}\nd_BS(p) SCAN  [estimator: {key}]\n{'='*60}")
    for p in sorted(k for k in by_p if k is not None):
        sub = by_p[p]
        by_L = ensemble_by_L(sub, key)
        if not by_L:
            continue
        fss = fss_extrapolate(by_L)
        per_L_str = ", ".join(f"L{L}={by_L[L][0]:.2f}(n{by_L[L][2]})" for L in by_L)
        entry = {"per_L": {str(L): {"mean": m, "sem": s, "n": n}
                           for L, (m, s, n) in by_L.items()}}
        line = f"  p={p:.6f}:  {per_L_str}"
        if fss:
            entry["fss"] = fss
            line += f"   ->  d_inf = {fss['d_inf']:.3f} +/- {fss['d_inf_err']:.3f}"
        print(line)
        report["p_scan"][f"{p:.6f}"] = entry

    # also report the other estimators per p (box + chain_all) without FSS verbosity
    report["secondary_estimators"] = {}
    for k2 in ["d_box", "d_chain_all_N"]:
        report["secondary_estimators"][k2] = {}
        for p in sorted(kk for kk in by_p if kk is not None):
            by_L = ensemble_by_L(by_p[p], k2)
            fss = fss_extrapolate(by_L) if len(by_L) >= 2 else None
            report["secondary_estimators"][k2][f"{p:.6f}"] = {
                "per_L": {str(L): by_L[L][0] for L in by_L},
                "d_inf": fss["d_inf"] if fss else None}

    if args.intrinsic_merged:
        try:
            with open(args.intrinsic_merged) as fh:
                report["intrinsic_estimator"] = json.load(fh)
            print("\n(intrinsic estimator JSON attached for convergence check)")
        except Exception as e:
            print(f"  (could not load intrinsic merged: {e})")

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nPromotion check: d_inf should be ~4 across the foam-connected p-range.")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
