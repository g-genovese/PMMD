"""
analyze_3fold_merger_combined.py
==================================
Aggregates 3-fold merger statistics across L = 6, 8, 10, 12 and computes
the framework prediction test:
    H_0: 1/4 of 3-fold events are unanimous (3:0 or 0:3 orientation pattern)
         3/4 are split (2:1 or 1:2)
    
Output:
    - Per-L sample sizes and observed fractions
    - Z-score and p-value of the framework prediction across all data
    - Bias trend across L (if any)
    - Markdown table suitable for v5.2 paper update

Existing v5.1 data: L=6, L=8 with 53,611 events, z = -0.39 sigma.

USAGE:
    python analyze_3fold_merger_combined.py \\
        --L6 e8_3fold_L6_run1.json \\
        --L8 e8_3fold_L8_run1.json \\
        --L10 e8_3fold_L10_serverA.json \\
        --L12 e8_3fold_L12_serverB.json e8_3fold_L12_serverC_sock0.json e8_3fold_L12_serverC_sock1.json
"""

import argparse
import json
import math
import sys
import numpy as np


def extract_3fold_counts(data):
    """From a v2 merger stats JSON, extract:
       - total_3fold: total 3-fold events at target_p (or final)
       - unanimous: count of 3:0 or 0:3 patterns
       - split: count of 2:1 or 1:2 patterns
       - with_zero: count of 3-fold events involving a cluster with orient_sum=0
    """
    if "framework_test_at_target_p" in data:
        ft = data["framework_test_at_target_p"]
        patterns = ft.get("patterns", {})
        unanimous_30 = patterns.get("3:0", 0)
        unanimous_03 = patterns.get("0:3", 0)
        split_21 = patterns.get("2:1", 0)
        split_12 = patterns.get("1:2", 0)
        with_zero = ft.get("with_zero_balance_cluster", 0)
        unanimous = unanimous_30 + unanimous_03
        split = split_21 + split_12
        total = unanimous + split  # excluding with_zero
        return {
            "total_3fold": total,
            "unanimous": unanimous,
            "split": split,
            "unanimous_30": unanimous_30,
            "unanimous_03": unanimous_03,
            "split_21": split_21,
            "split_12": split_12,
            "with_zero": with_zero,
        }
    elif "framework_test" in data:
        # Legacy format
        ft = data["framework_test"]
        # Map to new structure
        n_30 = ft.get("count_3to0", 0)
        n_03 = ft.get("count_0to3", 0)
        n_21 = ft.get("count_2to1", 0)
        n_12 = ft.get("count_1to2", 0)
        unanimous = n_30 + n_03
        split = n_21 + n_12
        total = unanimous + split
        return {
            "total_3fold": total,
            "unanimous": unanimous,
            "split": split,
            "unanimous_30": n_30,
            "unanimous_03": n_03,
            "split_21": n_21,
            "split_12": n_12,
            "with_zero": ft.get("with_zero", 0),
        }
    raise KeyError("No framework_test data in input JSON")


def binomial_z_score(observed, total, expected_frac):
    """Z-score for observed/total against null hypothesis p = expected_frac.
    Returns (z, two_sided_p_value).
    """
    if total < 1:
        return 0.0, 1.0
    p_hat = observed / total
    se = math.sqrt(expected_frac * (1 - expected_frac) / total)
    if se == 0:
        return 0.0, 1.0
    z = (p_hat - expected_frac) / se
    # Two-sided p-value (using erfc for normal CDF tail)
    try:
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(abs(z)))
    except ImportError:
        # Use math.erfc as fallback
        p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L6", nargs="*", default=[], help="L=6 result JSONs")
    ap.add_argument("--L8", nargs="*", default=[], help="L=8 result JSONs")
    ap.add_argument("--L10", nargs="*", default=[], help="L=10 result JSONs")
    ap.add_argument("--L12", nargs="*", default=[], help="L=12 result JSONs")
    args = ap.parse_args()

    sources = {6: args.L6, 8: args.L8, 10: args.L10, 12: args.L12}

    print("=" * 76)
    print("3-fold merger framework prediction: aggregated analysis")
    print("Framework prediction: 1/4 unanimous, 3/4 split  (random binomial)")
    print("=" * 76)
    print()

    by_L = {}
    combined = {"total": 0, "unanimous": 0, "split": 0}
    for L, files in sources.items():
        if not files:
            continue
        L_data = {"total": 0, "unanimous": 0, "split": 0,
                  "by_run": [], "files": [],
                  "with_zero": 0}
        for fpath in files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"  WARN: {fpath} not found, skipping")
                continue
            c = extract_3fold_counts(data)
            L_data["total"] += c["total_3fold"]
            L_data["unanimous"] += c["unanimous"]
            L_data["split"] += c["split"]
            L_data["with_zero"] += c.get("with_zero", 0)
            L_data["by_run"].append({"file": fpath, **c})
            L_data["files"].append(fpath)
        by_L[L] = L_data

    # Per-L analysis
    print(f"{'L':>3}  {'files':<6} {'total':>10} {'unanim':>10} {'split':>10}  "
          f"{'frac_unanim':>12} {'z':>7} {'p':>10}")
    print("-" * 80)
    for L in sorted(by_L.keys()):
        d = by_L[L]
        n_files = len(d["files"])
        tot = d["total"]
        un = d["unanimous"]
        sp = d["split"]
        if tot > 0:
            frac = un / tot
            z, p = binomial_z_score(un, tot, 0.25)
            print(f"{L:>3}  {n_files:<6} {tot:>10,} {un:>10,} {sp:>10,}  "
                  f"{frac:>11.4f}  {z:>+6.2f}  {p:>10.3e}")
        else:
            print(f"{L:>3}  {n_files:<6} {tot:>10,} {un:>10,} {sp:>10,}  "
                  f"{'(no data)':>12}")
        combined["total"] += tot
        combined["unanimous"] += un
        combined["split"] += sp

    print("-" * 80)
    if combined["total"] > 0:
        frac = combined["unanimous"] / combined["total"]
        z, p = binomial_z_score(combined["unanimous"], combined["total"], 0.25)
        print(f"{'ALL':>3}  {'':<6} {combined['total']:>10,} "
              f"{combined['unanimous']:>10,} {combined['split']:>10,}  "
              f"{frac:>11.4f}  {z:>+6.2f}  {p:>10.3e}")
        print()
        print(f"Combined framework test:")
        print(f"  Total 3-fold events: {combined['total']:,}")
        print(f"  Unanimous (predicted 1/4 = 25.00%):  "
              f"{combined['unanimous']:,} = {100*frac:.3f}%")
        print(f"  Split     (predicted 3/4 = 75.00%):  "
              f"{combined['split']:,} = {100*(1-frac):.3f}%")
        print(f"  Z-score vs framework: {z:+.3f} sigma")
        print(f"  Two-sided p-value:    {p:.4f}")
        if abs(z) < 2.0:
            print(f"  STATUS: Framework prediction CONSISTENT at {abs(z):.2f} sigma")
        elif abs(z) < 3.0:
            print(f"  STATUS: Framework prediction at 2-3 sigma tension")
        else:
            print(f"  STATUS: Framework prediction REJECTED at > 3 sigma")

        # Compare with v5.1 baseline (53,611 events, z = -0.39)
        if combined["total"] > 55000:
            print(f"\n  Compared to v5.1 baseline (L=6,8 only, 53,611 events, z=-0.39):")
            improvement = (combined["total"] - 53611) / 53611 * 100
            print(f"    New events added: {combined['total'] - 53611:,} "
                  f"(+{improvement:.0f}%)")

    # Markdown table for v5.2 paper
    print()
    print("MARKDOWN TABLE FOR v5.2 PAPER:")
    print("-" * 76)
    print("| L  | trials | 3-fold events | unanimous frac | predicted | z-score |")
    print("|----|--------|---------------|----------------|-----------|---------|")
    for L in sorted(by_L.keys()):
        d = by_L[L]
        if d["total"] > 0:
            frac = d["unanimous"] / d["total"]
            z, _ = binomial_z_score(d["unanimous"], d["total"], 0.25)
            # Try to get trials count from first file
            n_trials_total = 0
            for fpath in d["files"]:
                try:
                    with open(fpath) as f:
                        n_trials_total += json.load(f).get("trials", 0)
                except FileNotFoundError:
                    pass
            print(f"| {L:>2} | {n_trials_total:>6} | {d['total']:>13,} | "
                  f"{frac:>14.4f} | {0.25:>9.4f} | {z:+7.3f} |")
    if combined["total"] > 0:
        z, _ = binomial_z_score(combined["unanimous"], combined["total"], 0.25)
        frac = combined["unanimous"] / combined["total"]
        print(f"| **all** | — | **{combined['total']:,}** | "
              f"**{frac:.4f}** | **0.2500** | **{z:+.3f}** |")


if __name__ == "__main__":
    main()
