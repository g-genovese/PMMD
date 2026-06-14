#!/usr/bin/env python3
"""
extrapolate_koide.py  --  trusted continuum extrapolation of the Koide Q_K.

Re-analyses the per-L JSON-lines results written by pmmd_koide_hpc.py
(records with fields: L, Q, nzero, chir, koide{higgs_n3,conn_a2,flux}).

Why this exists
---------------
The farm's built-in fit included EVERY L, also the large-L points where the
overlap computation had lost one or more chiral zero modes (nzero < |Q|). At
those points the three lowest modes are NOT the three generations, so their
Q_K is meaningless; including them dragged the reported Q_K(L->inf) down to a
spurious 0.5044. This script keeps only VALID points (nzero == |Q|) and fits
    Q_K(L) = Q_inf + c/L         (and, for >=4 points, + d/L^2 as a check).
On the existing run the valid points (L <= ~160-192) extrapolate to ~2/3.

It does NOT re-run anything and needs only numpy -- safe, instant.

Usage
-----
    python3 extrapolate_koide.py pmmd_res_L*.jsonl
    python3 extrapolate_koide.py --key conn_a2 --chir-min 0.95 *.jsonl
    python3 extrapolate_koide.py --dir /home/aziz/pmmd_results
"""
import argparse, glob, json, os, sys
import numpy as np


def load(files):
    recs = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def fit_invL(L, q, with_L2=False):
    L = np.asarray(L, float); q = np.asarray(q, float)
    cols = [np.ones_like(L), 1.0 / L] + ([1.0 / L**2] if with_L2 else [])
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    return coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="result .jsonl files (globs ok)")
    ap.add_argument("--dir", default=None, help="dir to glob pmmd_res_*.jsonl from")
    ap.add_argument("--key", default="conn_a2",
                    choices=["conn_a2", "higgs_n3", "flux"],
                    help="which mass-operator Q_K to extrapolate")
    ap.add_argument("--chir-min", type=float, default=None,
                    help="extra cut: require min(|chirality|) >= this (e.g. 0.95)")
    args = ap.parse_args()

    files = list(args.files)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "pmmd_res_*.jsonl")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        sys.exit("no result files found (give .jsonl paths or --dir)")

    recs = load(files)
    # keep the best (highest min-chirality) record per L when duplicates exist,
    # but ONLY among valid ones; otherwise the lowest available for reporting.
    recs.sort(key=lambda r: r["L"])

    print(f"loaded {len(recs)} record(s) from {len(files)} file(s)")
    print(f"\n{'L':>5} {'Q_K('+args.key+')':>14} {'nzero':>6} {'min|chir|':>9} {'use':>5}")
    Ls, qs = [], []
    seen_valid = set()
    for r in recs:
        Q = abs(int(r.get("Q", 3)))
        nz = r.get("nzero", -1)
        chir = r.get("chir", [])
        cmin = min((abs(c) for c in chir), default=0.0)
        qk = r.get("koide", {}).get(args.key, float("nan"))
        valid = (nz == Q) and (args.chir_min is None or cmin >= args.chir_min)
        # avoid double-counting an L that has several valid records: take first
        use = valid and (r["L"] not in seen_valid)
        print(f"{r['L']:>5} {qk:>14.4f} {nz:>6} {cmin:>9.3f} {'yes' if use else 'NO':>5}")
        if use:
            Ls.append(r["L"]); qs.append(qk); seen_valid.add(r["L"])

    nval = len(set(Ls))
    print(f"\nvalid points (nzero==|Q|"
          + (f", min|chir|>={args.chir_min}" if args.chir_min is not None else "")
          + f"): {nval}")
    if nval < 3:
        sys.exit("need >= 3 valid L for a continuum fit; "
                 "fix the large-L runs (deflation / eigensolver) and re-run.")

    c1 = fit_invL(Ls, qs, with_L2=False)
    print(f"\n  Q_K(L->inf) = {c1[0]:.4f}      [Q_K = Q_inf + c/L],  c = {c1[1]:.3f}")
    print(f"  target 2/3  = {2/3:.4f}   (deviation {c1[0]-2/3:+.4f})")
    if nval >= 4:
        c2 = fit_invL(Ls, qs, with_L2=True)
        print(f"  [robustness, +d/L^2 term]: Q_K(L->inf) = {c2[0]:.4f}")
    # leave-one-out spread as a crude uncertainty
    if nval >= 4:
        Ls_a = np.array(Ls, float); qs_a = np.array(qs, float)
        loo = []
        for i in range(len(Ls_a)):
            m = np.arange(len(Ls_a)) != i
            loo.append(fit_invL(Ls_a[m], qs_a[m])[0])
        print(f"  [leave-one-out spread]: {min(loo):.4f} .. {max(loo):.4f}")


if __name__ == "__main__":
    main()
