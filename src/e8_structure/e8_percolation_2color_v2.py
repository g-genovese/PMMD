"""
e8_percolation_2color.py
========================
TWO-COLOR competitive site percolation on the E_8 foam graph.

Tests the Z/2 substrate symmetry breaking mechanism articulated in
Section 12 of the Autoconsistency framework v4.0:

  - Each vertex is assigned color +1 (matter) or -1 (antimatter) at random
    when it occurs in the Newman-Ziff random occupation order.
  - Clusters grow by same-color connectivity (only neighbours of the
    same color merge into the same cluster).
  - INTERACTION between opposite-color clusters is controlled by --mode:

    mode = 0  BASELINE (no interaction)
        + and - are two independent percolation processes on the same
        random-order Newman-Ziff occupation. Density per color: p/2.
        Expected: each color percolates at p/2 = p_c_single ≈ 1/175,
        i.e., total p ≈ 2/175 ≈ 1/87.5. No spontaneous breaking.

    mode = 1  HARD ANNIHILATION (vertex-level blocking)
        When a vertex v with color c would be added, if v has any
        neighbour of color -c, then v is REJECTED (does not activate).
        Models the Z/2 mutual exclusion: matter and antimatter cannot
        coexist at the same percolation interface.
        Expected: percolation onset shifts UP (effective per-color
        density reduced by annihilation), spontaneous symmetry breaking
        emerges (one color tends to occupy contiguous regions, locking
        out the other locally).

    mode = 2  3-FOLD MAJORITY (framework's natural mechanism)
        When v with color c is added, look at all its neighbours:
          * Same-color neighbours: standard union-find merge.
          * Opposite-color neighbours: count cluster-distinct opposite
            clusters touched. If 1 opposite cluster touched: 2-fold
            merger - both v and the opposite cluster are annihilated
            (set to vacuum). If >=2 disjoint opposite clusters touched:
            v joins same-color clusters and triggers a 3-fold merger
            in which the smaller opposite cluster annihilates with v.
        Most faithful to the framework's mechanism (Theorem 12.5).

Observables tracked over the occupation step k (k = 0,...,N_valid-1):
  * n_plus(k), n_minus(k): occupied vertex counts per color
  * S_max_plus(k), S_max_minus(k): largest same-color cluster size
  * sum_sq_plus(k), sum_sq_minus(k): sum of squares of cluster sizes
    (for per-color susceptibility chi(p) = (sum_sq - S_max^2) / N)
  * mode 1: n_rejected(k) - vertices killed by annihilation
  * mode 2: n_2fold, n_3fold - merger event counts

The framework's prediction (Section 12.3) is that the asymmetry
  A(k) = (S_max_plus - S_max_minus) / N
fluctuates symmetrically in mode 0 (no breaking), but in modes 1 and 2
develops |A| ~ const at the critical step (spontaneous breaking),
with the residual minority cluster size scaling as ~ p_c * mu_9 * J_CKM
times N (the framework's eta_B prediction).

USAGE EXAMPLES:
  # Validation runs (quick, <30 min on Linux server):
  python e8_percolation_2color.py --L 4 --trials 32 --workers 16 --mode 0
  python e8_percolation_2color.py --L 4 --trials 32 --workers 16 --mode 1
  python e8_percolation_2color.py --L 4 --trials 32 --workers 16 --mode 2

  # Production runs at L=6,8 (each ~1-2 hours):
  python e8_percolation_2color.py --L 6 --trials 32 --workers 24 --mode 1
  python e8_percolation_2color.py --L 8 --trials 32 --workers 24 --mode 2

  # Critical L=10 test for Z/2 mechanism (memory ~80 GB, ~3-6 hours):
  python e8_percolation_2color.py --L 10 --trials 32 --workers 24 --mode 2

ANALYSIS TIPS (after running):
  - Compare pc_plus and pc_minus across modes: should be ~equal by Z/2 symmetry.
  - In mode 0: p_total_critical should match 2 * single-color p_c ≈ 0.01143.
  - In mode 1: p_total_critical shifts UP due to rejected vertices.
  - In mode 2: count of 2-fold vs 3-fold events at critical step.
  - Asymmetry |S_max_+ - S_max_-|/N at p_total_critical: framework predicts
    in mode 2 this should scale ~ eta_B * (correlation length)^d_f / N ~ p_c * mu_9
    when properly normalized.

OUTPUT: JSON file e8_2color_L{L}_mode{M}.json with binned susceptibility curves
for both colors plus mode-specific event counts.
"""

import argparse
import json
import sys
import time
import platform
import numpy as np
from multiprocessing import cpu_count, get_context

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator


# =====================================================================
# E_8 root shifts (same as e8_percolation_implicit.py)
# =====================================================================

def make_e8_shifts():
    """Returns typeA (112,8), typeB_h0 (128,8), typeB_h1 (128,8)."""
    typeA = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0] * 8
                    v[i] = s1
                    v[j] = s2
                    typeA.append(v)
    typeA = np.array(typeA, dtype=np.int8)
    assert typeA.shape == (112, 8)

    typeB_h0 = []
    typeB_h1 = []
    for code in range(128):
        signs = [(1 if (code >> i) & 1 == 0 else -1) for i in range(7)]
        prod = 1
        for s in signs:
            prod *= s
        signs.append(prod)
        t = [(1 if s == -1 else 0) for s in signs]
        b0 = [-tt for tt in t]
        b1 = [1 - tt for tt in t]
        typeB_h0.append(b0)
        typeB_h1.append(b1)
    typeB_h0 = np.array(typeB_h0, dtype=np.int8)
    typeB_h1 = np.array(typeB_h1, dtype=np.int8)

    return typeA, typeB_h0, typeB_h1


# =====================================================================
# Valid vertex order (sum(c) even)
# =====================================================================

@njit(cache=True)
def make_valid_order(L, seed):
    """Random permutation of all valid vertex IDs (h, c) with sum(c) even."""
    L8 = L ** 8
    np.random.seed(seed)
    valid_count = 0
    valid = np.empty(2 * L8, dtype=np.int64)

    c = np.empty(8, dtype=np.int64)
    for v in range(2 * L8):
        if v >= L8:
            h = 1
            lex_v = v - L8
        else:
            h = 0
            lex_v = v
        x = lex_v
        s = 0
        for k in range(8):
            c[k] = x % L
            x //= L
            s += c[k]
        if s % 2 == 0:
            valid[valid_count] = v
            valid_count += 1

    valid = valid[:valid_count]
    perm = np.random.permutation(valid_count)
    order = valid[perm]
    return order


# =====================================================================
# Two-color Newman-Ziff core (mode = 0)
# =====================================================================

@njit(cache=True)
def trial_mode0(L, order, typeA, typeB_h0, typeB_h1, n_bins, color_seed):
    """
    Mode 0: BASELINE — two independent percolation processes.
    Each vertex gets random color +1 or -1; merging only with same color.
    No interaction between colors.
    """
    L8 = L ** 8
    N_total = 2 * L8
    N_valid = order.shape[0]

    parent = np.full(N_total, -1, dtype=np.int32)
    size = np.zeros(N_total, dtype=np.int32)
    color = np.zeros(N_total, dtype=np.int8)  # 0 = unactivated, 1 = +, -1 = -

    largest_plus = 0
    largest_minus = 0
    sum_sq_plus = np.int64(0)
    sum_sq_minus = np.int64(0)
    n_plus_count = 0
    n_minus_count = 0

    bin_size = max(N_valid // n_bins, 1)
    S_max_plus_bin = np.zeros(n_bins, dtype=np.float64)
    S_max_minus_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_plus_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_minus_bin = np.zeros(n_bins, dtype=np.float64)

    # color RNG state (separate from order RNG)
    np.random.seed(color_seed)
    # pre-generate colors for all valid vertices: ±1 with prob 1/2
    colors = np.where(np.random.random(N_valid) < 0.5, 1, -1).astype(np.int8)

    c = np.empty(8, dtype=np.int64)

    for i in range(N_valid):
        v = order[i]
        col_v = colors[i]

        if v >= L8:
            h = 1
            lex_v = v - L8
        else:
            h = 0
            lex_v = v

        x = lex_v
        for k in range(8):
            c[k] = x % L
            x //= L

        # Activate v
        parent[v] = v
        size[v] = 1
        color[v] = col_v
        if col_v == 1:
            n_plus_count += 1
            sum_sq_plus += 1
            if largest_plus < 1:
                largest_plus = 1
        else:
            n_minus_count += 1
            sum_sq_minus += 1
            if largest_minus < 1:
                largest_minus = 1
        current = np.int64(v)

        # ---- Type-A neighbours (112) ----
        for ri in range(112):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeA[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h * L8 + lex_u

            if parent[u] == -1:
                continue
            # Same color only
            if color[u] != col_v:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt

            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt

            if ra == rb:
                continue

            sa = size[ra]
            sb = size[rb]
            delta = 2 * np.int64(sa) * np.int64(sb)
            if col_v == 1:
                sum_sq_plus += delta
            else:
                sum_sq_minus += delta

            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                if col_v == 1 and sa + sb > largest_plus:
                    largest_plus = sa + sb
                elif col_v == -1 and sa + sb > largest_minus:
                    largest_minus = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                if col_v == 1 and sa + sb > largest_plus:
                    largest_plus = sa + sb
                elif col_v == -1 and sa + sb > largest_minus:
                    largest_minus = sa + sb

        # ---- Type-B neighbours (128) ----
        if h == 0:
            typeB = typeB_h0
            h_new = 1
        else:
            typeB = typeB_h1
            h_new = 0

        for ri in range(128):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeB[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h_new * L8 + lex_u

            if parent[u] == -1:
                continue
            if color[u] != col_v:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt

            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt

            if ra == rb:
                continue

            sa = size[ra]
            sb = size[rb]
            delta = 2 * np.int64(sa) * np.int64(sb)
            if col_v == 1:
                sum_sq_plus += delta
            else:
                sum_sq_minus += delta

            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                if col_v == 1 and sa + sb > largest_plus:
                    largest_plus = sa + sb
                elif col_v == -1 and sa + sb > largest_minus:
                    largest_minus = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                if col_v == 1 and sa + sb > largest_plus:
                    largest_plus = sa + sb
                elif col_v == -1 and sa + sb > largest_minus:
                    largest_minus = sa + sb

        bin_idx = i // bin_size
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        S_max_plus_bin[bin_idx] = max(S_max_plus_bin[bin_idx], np.float64(largest_plus))
        S_max_minus_bin[bin_idx] = max(S_max_minus_bin[bin_idx], np.float64(largest_minus))
        sum_sq_plus_bin[bin_idx] = max(sum_sq_plus_bin[bin_idx], np.float64(sum_sq_plus))
        sum_sq_minus_bin[bin_idx] = max(sum_sq_minus_bin[bin_idx], np.float64(sum_sq_minus))

    return S_max_plus_bin, S_max_minus_bin, sum_sq_plus_bin, sum_sq_minus_bin


# =====================================================================
# Two-color Newman-Ziff with HARD ANNIHILATION (mode = 1)
# =====================================================================

@njit(cache=True)
def trial_mode1(L, order, typeA, typeB_h0, typeB_h1, n_bins, color_seed):
    """
    Mode 1: HARD ANNIHILATION — vertex-level blocking.
    When v of color c is being added, if v has any neighbour of color -c,
    then v is REJECTED (does not activate; parent stays -1, color stays 0).
    """
    L8 = L ** 8
    N_total = 2 * L8
    N_valid = order.shape[0]

    parent = np.full(N_total, -1, dtype=np.int32)
    size = np.zeros(N_total, dtype=np.int32)
    color = np.zeros(N_total, dtype=np.int8)

    largest_plus = 0
    largest_minus = 0
    sum_sq_plus = np.int64(0)
    sum_sq_minus = np.int64(0)
    n_plus_count = 0
    n_minus_count = 0
    n_rejected = 0

    bin_size = max(N_valid // n_bins, 1)
    S_max_plus_bin = np.zeros(n_bins, dtype=np.float64)
    S_max_minus_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_plus_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_minus_bin = np.zeros(n_bins, dtype=np.float64)
    rejected_bin = np.zeros(n_bins, dtype=np.float64)

    np.random.seed(color_seed)
    colors = np.where(np.random.random(N_valid) < 0.5, 1, -1).astype(np.int8)

    c = np.empty(8, dtype=np.int64)

    for i in range(N_valid):
        v = order[i]
        col_v = colors[i]

        if v >= L8:
            h = 1
            lex_v = v - L8
        else:
            h = 0
            lex_v = v

        x = lex_v
        for k in range(8):
            c[k] = x % L
            x //= L

        # ===== PASS 1: check for opposite-color neighbours (annihilation) =====
        blocked = False
        # Type-A
        for ri in range(112):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeA[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h * L8 + lex_u
            if parent[u] != -1 and color[u] == -col_v:
                blocked = True
                break

        if not blocked:
            # Type-B
            if h == 0:
                typeB = typeB_h0
                h_new = 1
            else:
                typeB = typeB_h1
                h_new = 0
            for ri in range(128):
                lex_u = np.int64(0)
                pw = np.int64(1)
                for k in range(8):
                    ck = c[k] + typeB[ri, k]
                    if ck < 0:
                        ck += L
                    elif ck >= L:
                        ck -= L
                    lex_u += ck * pw
                    pw *= L
                u = h_new * L8 + lex_u
                if parent[u] != -1 and color[u] == -col_v:
                    blocked = True
                    break

        if blocked:
            n_rejected += 1
            # Update bin (no change in state, but record current cumulative)
            bin_idx = i // bin_size
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            S_max_plus_bin[bin_idx] = max(S_max_plus_bin[bin_idx], np.float64(largest_plus))
            S_max_minus_bin[bin_idx] = max(S_max_minus_bin[bin_idx], np.float64(largest_minus))
            sum_sq_plus_bin[bin_idx] = max(sum_sq_plus_bin[bin_idx], np.float64(sum_sq_plus))
            sum_sq_minus_bin[bin_idx] = max(sum_sq_minus_bin[bin_idx], np.float64(sum_sq_minus))
            rejected_bin[bin_idx] = max(rejected_bin[bin_idx], np.float64(n_rejected))
            continue

        # ===== PASS 2: activate v and merge with same-color neighbours =====
        parent[v] = v
        size[v] = 1
        color[v] = col_v
        if col_v == 1:
            n_plus_count += 1
            sum_sq_plus += 1
            if largest_plus < 1:
                largest_plus = 1
        else:
            n_minus_count += 1
            sum_sq_minus += 1
            if largest_minus < 1:
                largest_minus = 1
        current = np.int64(v)

        # Type-A neighbours (same color only)
        for ri in range(112):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeA[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h * L8 + lex_u
            if parent[u] == -1 or color[u] != col_v:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt
            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt
            if ra == rb:
                continue
            sa = size[ra]
            sb = size[rb]
            delta = 2 * np.int64(sa) * np.int64(sb)
            if col_v == 1:
                sum_sq_plus += delta
            else:
                sum_sq_minus += delta
            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                new_size = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                new_size = sa + sb
            if col_v == 1 and new_size > largest_plus:
                largest_plus = new_size
            elif col_v == -1 and new_size > largest_minus:
                largest_minus = new_size

        # Type-B neighbours
        if h == 0:
            typeB = typeB_h0
            h_new = 1
        else:
            typeB = typeB_h1
            h_new = 0
        for ri in range(128):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeB[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h_new * L8 + lex_u
            if parent[u] == -1 or color[u] != col_v:
                continue

            ra = current
            while parent[ra] != ra:
                ra = parent[ra]
            x2 = current
            while parent[x2] != ra:
                nxt = parent[x2]
                parent[x2] = ra
                x2 = nxt
            rb = u
            while parent[rb] != rb:
                rb = parent[rb]
            x2 = u
            while parent[x2] != rb:
                nxt = parent[x2]
                parent[x2] = rb
                x2 = nxt
            if ra == rb:
                continue
            sa = size[ra]
            sb = size[rb]
            delta = 2 * np.int64(sa) * np.int64(sb)
            if col_v == 1:
                sum_sq_plus += delta
            else:
                sum_sq_minus += delta
            if sa < sb:
                parent[ra] = rb
                size[rb] = sa + sb
                current = rb
                new_size = sa + sb
            else:
                parent[rb] = ra
                size[ra] = sa + sb
                current = ra
                new_size = sa + sb
            if col_v == 1 and new_size > largest_plus:
                largest_plus = new_size
            elif col_v == -1 and new_size > largest_minus:
                largest_minus = new_size

        bin_idx = i // bin_size
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        S_max_plus_bin[bin_idx] = max(S_max_plus_bin[bin_idx], np.float64(largest_plus))
        S_max_minus_bin[bin_idx] = max(S_max_minus_bin[bin_idx], np.float64(largest_minus))
        sum_sq_plus_bin[bin_idx] = max(sum_sq_plus_bin[bin_idx], np.float64(sum_sq_plus))
        sum_sq_minus_bin[bin_idx] = max(sum_sq_minus_bin[bin_idx], np.float64(sum_sq_minus))
        rejected_bin[bin_idx] = max(rejected_bin[bin_idx], np.float64(n_rejected))

    return S_max_plus_bin, S_max_minus_bin, sum_sq_plus_bin, sum_sq_minus_bin, rejected_bin


# =====================================================================
# Two-color with 3-FOLD MAJORITY rule (mode = 2)
# =====================================================================

@njit(cache=True)
def trial_mode2(L, order, typeA, typeB_h0, typeB_h1, n_bins, color_seed):
    """
    Mode 2: 3-FOLD MAJORITY rule.
    When v of color c is added:
      1. Gather all opposite-color neighbour cluster roots (deduplicated).
      2. If 0 opposite clusters touched: standard merge with same-color neighbours.
      3. If 1 opposite cluster: 2-fold annihilation - the opposite cluster
         is removed (set to parent=-1), v also rejected (vacuum).
      4. If >=2 distinct opposite clusters: 3-fold majority merger -
         v joins same-color neighbours (forming the majority side);
         the smallest opposite cluster is annihilated; others kept.
    """
    L8 = L ** 8
    N_total = 2 * L8
    N_valid = order.shape[0]

    parent = np.full(N_total, -1, dtype=np.int32)
    size = np.zeros(N_total, dtype=np.int32)
    color = np.zeros(N_total, dtype=np.int8)

    largest_plus = 0
    largest_minus = 0
    sum_sq_plus = np.int64(0)
    sum_sq_minus = np.int64(0)
    n_annihilated_2fold = 0
    n_annihilated_3fold = 0

    bin_size = max(N_valid // n_bins, 1)
    S_max_plus_bin = np.zeros(n_bins, dtype=np.float64)
    S_max_minus_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_plus_bin = np.zeros(n_bins, dtype=np.float64)
    sum_sq_minus_bin = np.zeros(n_bins, dtype=np.float64)
    n_2fold_bin = np.zeros(n_bins, dtype=np.float64)
    n_3fold_bin = np.zeros(n_bins, dtype=np.float64)

    np.random.seed(color_seed)
    colors = np.where(np.random.random(N_valid) < 0.5, 1, -1).astype(np.int8)

    c = np.empty(8, dtype=np.int64)

    # Buffers for unique opposite-cluster roots (at most 240 different roots)
    opp_roots = np.empty(240, dtype=np.int64)
    opp_sizes = np.empty(240, dtype=np.int64)

    # NOTE: an unused find_root closure originally lived here and prevented
    # numba from compiling trial_mode2 in nopython mode (closure captured
    # `parent`, which numba refuses). All path compression below is inlined,
    # so we just drop the closure.

    for i in range(N_valid):
        v = order[i]
        col_v = colors[i]

        if v >= L8:
            h = 1
            lex_v = v - L8
        else:
            h = 0
            lex_v = v

        x = lex_v
        for k in range(8):
            c[k] = x % L
            x //= L

        # ===== PASS 1: scan neighbours, partition into same/opposite =====
        n_opp = 0
        # Type-A
        for ri in range(112):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeA[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h * L8 + lex_u
            if parent[u] != -1 and color[u] == -col_v:
                # find root
                r = u
                while parent[r] != r:
                    r = parent[r]
                # FIX: skip annihilated cluster (sentinel size<0)
                if size[r] < 0:
                    continue
                # deduplicate
                already = False
                for j in range(n_opp):
                    if opp_roots[j] == r:
                        already = True
                        break
                if not already:
                    opp_roots[n_opp] = r
                    opp_sizes[n_opp] = size[r]
                    n_opp += 1
        # Type-B
        if h == 0:
            typeB = typeB_h0
            h_new = 1
        else:
            typeB = typeB_h1
            h_new = 0
        for ri in range(128):
            lex_u = np.int64(0)
            pw = np.int64(1)
            for k in range(8):
                ck = c[k] + typeB[ri, k]
                if ck < 0:
                    ck += L
                elif ck >= L:
                    ck -= L
                lex_u += ck * pw
                pw *= L
            u = h_new * L8 + lex_u
            if parent[u] != -1 and color[u] == -col_v:
                r = u
                while parent[r] != r:
                    r = parent[r]
                # FIX: skip annihilated cluster
                if size[r] < 0:
                    continue
                already = False
                for j in range(n_opp):
                    if opp_roots[j] == r:
                        already = True
                        break
                if not already:
                    opp_roots[n_opp] = r
                    opp_sizes[n_opp] = size[r]
                    n_opp += 1

        # ===== DECISION =====
        if n_opp == 0:
            # Standard merger with same-color neighbours
            parent[v] = v
            size[v] = 1
            color[v] = col_v
            if col_v == 1:
                sum_sq_plus += 1
                if largest_plus < 1:
                    largest_plus = 1
            else:
                sum_sq_minus += 1
                if largest_minus < 1:
                    largest_minus = 1
            current = np.int64(v)

            # Type-A
            for ri in range(112):
                lex_u = np.int64(0)
                pw = np.int64(1)
                for k in range(8):
                    ck = c[k] + typeA[ri, k]
                    if ck < 0:
                        ck += L
                    elif ck >= L:
                        ck -= L
                    lex_u += ck * pw
                    pw *= L
                u = h * L8 + lex_u
                if parent[u] == -1 or color[u] != col_v:
                    continue
                ra = current
                while parent[ra] != ra:
                    ra = parent[ra]
                rb = u
                while parent[rb] != rb:
                    rb = parent[rb]
                if ra == rb:
                    continue
                if size[ra] < 0 or size[rb] < 0:
                    continue  # FIX: skip annihilated cluster
                sa = size[ra]
                sb = size[rb]
                delta = 2 * np.int64(sa) * np.int64(sb)
                if col_v == 1:
                    sum_sq_plus += delta
                else:
                    sum_sq_minus += delta
                if sa < sb:
                    parent[ra] = rb
                    size[rb] = sa + sb
                    current = rb
                    new_size = sa + sb
                else:
                    parent[rb] = ra
                    size[ra] = sa + sb
                    current = ra
                    new_size = sa + sb
                if col_v == 1 and new_size > largest_plus:
                    largest_plus = new_size
                elif col_v == -1 and new_size > largest_minus:
                    largest_minus = new_size
            # Type-B
            for ri in range(128):
                lex_u = np.int64(0)
                pw = np.int64(1)
                for k in range(8):
                    ck = c[k] + typeB[ri, k]
                    if ck < 0:
                        ck += L
                    elif ck >= L:
                        ck -= L
                    lex_u += ck * pw
                    pw *= L
                u = h_new * L8 + lex_u
                if parent[u] == -1 or color[u] != col_v:
                    continue
                ra = current
                while parent[ra] != ra:
                    ra = parent[ra]
                rb = u
                while parent[rb] != rb:
                    rb = parent[rb]
                if ra == rb:
                    continue
                if size[ra] < 0 or size[rb] < 0:
                    continue  # FIX: skip annihilated cluster
                sa = size[ra]
                sb = size[rb]
                delta = 2 * np.int64(sa) * np.int64(sb)
                if col_v == 1:
                    sum_sq_plus += delta
                else:
                    sum_sq_minus += delta
                if sa < sb:
                    parent[ra] = rb
                    size[rb] = sa + sb
                    current = rb
                    new_size = sa + sb
                else:
                    parent[rb] = ra
                    size[ra] = sa + sb
                    current = ra
                    new_size = sa + sb
                if col_v == 1 and new_size > largest_plus:
                    largest_plus = new_size
                elif col_v == -1 and new_size > largest_minus:
                    largest_minus = new_size

        elif n_opp == 1:
            # 2-fold annihilation: opposite cluster removed, v stays as vacuum
            r_opp = opp_roots[0]
            s_opp = opp_sizes[0]
            n_annihilated_2fold += 1
            # Subtract contribution from sum_sq of opposite color
            if -col_v == 1:
                sum_sq_plus -= np.int64(s_opp) * np.int64(s_opp)
                if s_opp == largest_plus:
                    # need to find new largest_plus among remaining clusters
                    # but for speed we just keep the recorded value (a small bias)
                    # in practice will be recomputed in post-processing
                    pass
            else:
                sum_sq_minus -= np.int64(s_opp) * np.int64(s_opp)
                if s_opp == largest_minus:
                    pass

            # Mark all vertices of opposite cluster as vacuum (parent = -2, color = 0)
            # To avoid O(N) scan, we use a sentinel: parent[r_opp] = -2 means "annihilated root"
            # Future find_root will skip it
            # But this complicates the algorithm. Simpler: leave the cluster in UF
            # but set color to 0 — future neighbours won't merge with it (color check).
            # Mark by setting color[r_opp] = 0 (the root). Members keep their color
            # because we never re-check them — only via find_root.
            # Actually, we want NEW vertices to not see this cluster.
            # Set color of ALL members to 0 — but expensive.
            # Approximation: set color of root to 0; new vertices checking will see
            # root has color 0 and skip. This works because color is checked via
            # color[u], and find_root will give r_opp.
            # WAIT — color[u] is checked, not color[root]. So we need to mark each member.
            #
            # COMPROMISE: instead of marking annihilated cluster as vacuum, we 
            # FREEZE it (it can't accept new members of its own color either).
            # This is incorrect physically but algorithmically simpler.
            # 
            # CORRECT approach: maintain a separate `alive` flag per cluster.
            # When checking color[u] for new neighbours, also check alive[find_root(u)].
            # 
            # For numba compatibility, use size[r_opp] = -1 as "annihilated marker".
            size[r_opp] = -1  # sentinel: cluster is annihilated
            # v itself is not activated (vacuum)

        else:
            # n_opp >= 2: 3-fold (or higher) merger.
            # Smallest opposite cluster is annihilated; v joins same-color neighbours.
            # Identify smallest opposite cluster
            min_idx = 0
            min_size = opp_sizes[0]
            for j in range(1, n_opp):
                if opp_sizes[j] < min_size:
                    min_size = opp_sizes[j]
                    min_idx = j
            r_min = opp_roots[min_idx]
            s_min = opp_sizes[min_idx]
            n_annihilated_3fold += 1
            if -col_v == 1:
                sum_sq_plus -= np.int64(s_min) * np.int64(s_min)
            else:
                sum_sq_minus -= np.int64(s_min) * np.int64(s_min)
            size[r_min] = -1  # annihilate smallest opposite

            # Now activate v and merge with same-color
            parent[v] = v
            size[v] = 1
            color[v] = col_v
            if col_v == 1:
                sum_sq_plus += 1
                if largest_plus < 1:
                    largest_plus = 1
            else:
                sum_sq_minus += 1
                if largest_minus < 1:
                    largest_minus = 1
            current = np.int64(v)

            # Type-A same-color merge
            for ri in range(112):
                lex_u = np.int64(0)
                pw = np.int64(1)
                for k in range(8):
                    ck = c[k] + typeA[ri, k]
                    if ck < 0:
                        ck += L
                    elif ck >= L:
                        ck -= L
                    lex_u += ck * pw
                    pw *= L
                u = h * L8 + lex_u
                if parent[u] == -1 or color[u] != col_v:
                    continue
                ra = current
                while parent[ra] != ra:
                    ra = parent[ra]
                rb = u
                while parent[rb] != rb:
                    rb = parent[rb]
                if ra == rb:
                    continue
                if size[ra] < 0 or size[rb] < 0:
                    continue  # one is annihilated
                sa = size[ra]
                sb = size[rb]
                delta = 2 * np.int64(sa) * np.int64(sb)
                if col_v == 1:
                    sum_sq_plus += delta
                else:
                    sum_sq_minus += delta
                if sa < sb:
                    parent[ra] = rb
                    size[rb] = sa + sb
                    current = rb
                    new_size = sa + sb
                else:
                    parent[rb] = ra
                    size[ra] = sa + sb
                    current = ra
                    new_size = sa + sb
                if col_v == 1 and new_size > largest_plus:
                    largest_plus = new_size
                elif col_v == -1 and new_size > largest_minus:
                    largest_minus = new_size

            # Type-B
            for ri in range(128):
                lex_u = np.int64(0)
                pw = np.int64(1)
                for k in range(8):
                    ck = c[k] + typeB[ri, k]
                    if ck < 0:
                        ck += L
                    elif ck >= L:
                        ck -= L
                    lex_u += ck * pw
                    pw *= L
                u = h_new * L8 + lex_u
                if parent[u] == -1 or color[u] != col_v:
                    continue
                ra = current
                while parent[ra] != ra:
                    ra = parent[ra]
                rb = u
                while parent[rb] != rb:
                    rb = parent[rb]
                if ra == rb:
                    continue
                if size[ra] < 0 or size[rb] < 0:
                    continue
                sa = size[ra]
                sb = size[rb]
                delta = 2 * np.int64(sa) * np.int64(sb)
                if col_v == 1:
                    sum_sq_plus += delta
                else:
                    sum_sq_minus += delta
                if sa < sb:
                    parent[ra] = rb
                    size[rb] = sa + sb
                    current = rb
                    new_size = sa + sb
                else:
                    parent[rb] = ra
                    size[ra] = sa + sb
                    current = ra
                    new_size = sa + sb
                if col_v == 1 and new_size > largest_plus:
                    largest_plus = new_size
                elif col_v == -1 and new_size > largest_minus:
                    largest_minus = new_size

        # ===== Bin update (end-of-bin sampling with fresh max scan) =====
        # FIX: replace running-max with fresh O(N_total) scan only at bin
        # boundaries, so the staleness of `largest_*` after annihilation
        # events does not contaminate the recorded S_max.
        if (i + 1) % bin_size == 0:
            bin_idx = (i + 1) // bin_size - 1
            if bin_idx < n_bins:
                cur_max_plus = np.int32(0)
                cur_max_minus = np.int32(0)
                for kk in range(N_total):
                    if parent[kk] == kk:
                        s_kk = size[kk]
                        if s_kk > 0:
                            c_kk = color[kk]
                            if c_kk == 1 and s_kk > cur_max_plus:
                                cur_max_plus = s_kk
                            elif c_kk == -1 and s_kk > cur_max_minus:
                                cur_max_minus = s_kk
                S_max_plus_bin[bin_idx] = np.float64(cur_max_plus)
                S_max_minus_bin[bin_idx] = np.float64(cur_max_minus)
                sum_sq_plus_bin[bin_idx] = np.float64(sum_sq_plus)
                sum_sq_minus_bin[bin_idx] = np.float64(sum_sq_minus)
                n_2fold_bin[bin_idx] = np.float64(n_annihilated_2fold)
                n_3fold_bin[bin_idx] = np.float64(n_annihilated_3fold)

    return S_max_plus_bin, S_max_minus_bin, sum_sq_plus_bin, sum_sq_minus_bin, n_2fold_bin, n_3fold_bin


# =====================================================================
# Worker dispatch
# =====================================================================

def worker(args):
    L, mode, n_bins, order_seed, color_seed, trial_idx, total = args
    typeA, typeB_h0, typeB_h1 = make_e8_shifts()
    order = make_valid_order(L, order_seed)
    t0 = time.time()
    if mode == 0:
        result = trial_mode0(L, order, typeA, typeB_h0, typeB_h1, n_bins, color_seed)
    elif mode == 1:
        result = trial_mode1(L, order, typeA, typeB_h0, typeB_h1, n_bins, color_seed)
    elif mode == 2:
        result = trial_mode2(L, order, typeA, typeB_h0, typeB_h1, n_bins, color_seed)
    else:
        raise ValueError(f"Invalid mode {mode}")
    dt = time.time() - t0
    print(f"[trial {trial_idx+1}/{total}] L={L} mode={mode} done in {dt:.1f}s", flush=True)
    return [np.asarray(x) for x in result]


def find_pc_first_local_max(chi, p_max=0.1, n_bins=None):
    """
    FIX: find the FIRST local maximum of chi in p < p_max,
    instead of the global argmax. Avoids the artefact where chi keeps
    growing slowly toward p=1 after the real percolation peak.

    Returns (p_at_peak, chi_at_peak, index).
    """
    if n_bins is None:
        n_bins = len(chi)
    idx_max = int(np.floor(p_max * n_bins))
    idx_max = min(idx_max, len(chi) - 1)
    if idx_max < 2:
        return None, None, None
    # Find first local max in [1, idx_max-1]: chi[i] > chi[i-1] and chi[i] > chi[i+1]
    for i in range(1, idx_max):
        if chi[i] > chi[i - 1] and chi[i] >= chi[i + 1]:
            # Quadratic interpolation around the peak
            if 0 < i < len(chi) - 1 and chi[i] > 0:
                x = np.array([i - 1, i, i + 1], dtype=float)
                y = np.array([chi[i - 1], chi[i], chi[i + 1]], dtype=float)
                a, b, _ = np.polyfit(x, y, 2)
                if a < 0:
                    x_peak = -b / (2 * a)
                    return float(x_peak / n_bins), float(chi[i]), i
            return float((i + 0.5) / n_bins), float(chi[i]), i
    # No local max in window: fall back to argmax in window
    i_arg = int(np.argmax(chi[: idx_max + 1]))
    return float((i_arg + 0.5) / n_bins), float(chi[i_arg]), i_arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--mode", type=int, default=1, choices=[0, 1, 2],
                    help="0=baseline, 1=hard annihilation, 2=3-fold majority")
    ap.add_argument("--n-bins", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--pc-search-pmax", type=float, default=0.1,
                    help="Upper bound on p when searching for chi peak (default 0.1)")
    ap.add_argument("--save-pertrial-curves", action="store_true", default=True,
                    help="Save per-trial bin curves to an npz sidecar file")
    args = ap.parse_args()

    L = args.L
    L8 = L ** 8
    N_valid = L8  # half of 2*L^8 due to even-sum constraint
    print(f"L = {L}, N_valid = {N_valid}, mode = {args.mode}, trials = {args.trials}, workers = {args.workers}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Numba: {HAS_NUMBA}")

    rng = np.random.RandomState(args.seed)
    seeds = rng.randint(0, 2**31 - 1, size=(args.trials, 2))

    work = [(L, args.mode, args.n_bins, int(seeds[k, 0]), int(seeds[k, 1]), k, args.trials)
            for k in range(args.trials)]

    ctx = get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        results = pool.map(worker, work)
    dt = time.time() - t0
    print(f"Total time: {dt:.1f}s ({dt / args.trials:.1f}s/trial avg)")

    # Aggregate results
    if args.mode == 1:
        keys = ["S_max_plus", "S_max_minus", "sum_sq_plus", "sum_sq_minus", "n_rejected"]
    elif args.mode == 2:
        keys = ["S_max_plus", "S_max_minus", "sum_sq_plus", "sum_sq_minus", "n_2fold", "n_3fold"]
    else:
        keys = ["S_max_plus", "S_max_minus", "sum_sq_plus", "sum_sq_minus"]

    arrays = {k: np.zeros((args.trials, args.n_bins)) for k in keys}
    for i, res in enumerate(results):
        for k, v in zip(keys, res):
            arrays[k][i] = v

    bin_centers = (np.arange(args.n_bins) + 0.5) / args.n_bins  # p = k/N_valid

    output = {
        "L": L,
        "N_valid": N_valid,
        "mode": args.mode,
        "trials": args.trials,
        "n_bins": args.n_bins,
        "platform": platform.system(),
        "time_s": dt,
        "bin_centers": bin_centers.tolist(),
        "pc_search_pmax": args.pc_search_pmax,
        "script_version": "v2 (size<0 filter, end-of-bin sampling, per-trial output)",
    }
    for k in keys:
        mean = arrays[k].mean(axis=0)
        sem = arrays[k].std(axis=0, ddof=1) / np.sqrt(args.trials) if args.trials > 1 else np.zeros_like(mean)
        output[k + "_mean"] = mean.tolist()
        output[k + "_sem"] = sem.tolist()

    # ===== Per-trial p_c estimates (proper local-max search) =====
    chi_plus_per = (arrays["sum_sq_plus"] - arrays["S_max_plus"] ** 2) / N_valid
    chi_minus_per = (arrays["sum_sq_minus"] - arrays["S_max_minus"] ** 2) / N_valid
    pc_plus_per_trial = []
    pc_minus_per_trial = []
    chi_plus_peak_per_trial = []
    chi_minus_peak_per_trial = []
    for t_idx in range(args.trials):
        pp, cp, _ = find_pc_first_local_max(chi_plus_per[t_idx], args.pc_search_pmax, args.n_bins)
        pm, cm, _ = find_pc_first_local_max(chi_minus_per[t_idx], args.pc_search_pmax, args.n_bins)
        pc_plus_per_trial.append(pp)
        pc_minus_per_trial.append(pm)
        chi_plus_peak_per_trial.append(cp)
        chi_minus_peak_per_trial.append(cm)
    output["pc_plus_per_trial"] = pc_plus_per_trial
    output["pc_minus_per_trial"] = pc_minus_per_trial
    output["chi_plus_peak_per_trial"] = chi_plus_peak_per_trial
    output["chi_minus_peak_per_trial"] = chi_minus_peak_per_trial

    # ===== p_c on the trial-averaged chi (preferred for reporting) =====
    sq_plus_m = np.array(output["sum_sq_plus_mean"])
    sq_minus_m = np.array(output["sum_sq_minus_mean"])
    S_plus_m = np.array(output["S_max_plus_mean"])
    S_minus_m = np.array(output["S_max_minus_mean"])
    chi_plus_m = (sq_plus_m - S_plus_m ** 2) / N_valid
    chi_minus_m = (sq_minus_m - S_minus_m ** 2) / N_valid
    pc_plus, chi_plus_peak, _ = find_pc_first_local_max(chi_plus_m, args.pc_search_pmax, args.n_bins)
    pc_minus, chi_minus_peak, _ = find_pc_first_local_max(chi_minus_m, args.pc_search_pmax, args.n_bins)
    output["pc_plus"] = pc_plus
    output["pc_minus"] = pc_minus
    output["chi_plus_peak"] = chi_plus_peak
    output["chi_minus_peak"] = chi_minus_peak
    output["inv_pc_plus"] = 1.0 / pc_plus if pc_plus and pc_plus > 0 else None
    output["inv_pc_minus"] = 1.0 / pc_minus if pc_minus and pc_minus > 0 else None

    # ===== Per-trial finals + per-trial peak observables =====
    final_S_plus = arrays["S_max_plus"][:, -1]
    final_S_minus = arrays["S_max_minus"][:, -1]
    output["final_S_max_plus_per_trial"] = final_S_plus.tolist()
    output["final_S_max_minus_per_trial"] = final_S_minus.tolist()
    output["final_asymmetry_per_trial"] = ((final_S_plus - final_S_minus) / N_valid).tolist()
    output["final_winner_per_trial"] = ['plus' if a > 0 else 'minus' for a in (final_S_plus - final_S_minus)]

    # ===== Asymmetry at trial-averaged critical p =====
    if pc_plus and pc_plus > 0:
        ic = int(round(pc_plus * args.n_bins))
        ic = max(0, min(args.n_bins - 1, ic))
        asymm_at_pc = float(abs(S_plus_m[ic] - S_minus_m[ic]) / N_valid)
        output["asymmetry_at_pc"] = asymm_at_pc

    # ===== Summary printout =====
    print("\n" + "=" * 70)
    print(f"  RESULTS — L={L}, mode={args.mode}, {args.trials} trials")
    print("=" * 70)
    if pc_plus is not None and pc_plus > 0:
        print(f"  p_c(+) [trial-avg]    = {pc_plus:.6f}  =  1/{1/pc_plus:.2f}  (chi_peak={chi_plus_peak:.2f})")
    if pc_minus is not None and pc_minus > 0:
        print(f"  p_c(-) [trial-avg]    = {pc_minus:.6f}  =  1/{1/pc_minus:.2f}  (chi_peak={chi_minus_peak:.2f})")
    print(f"  1-color reference p_c = {1/175:.6f}  =  1/175")

    # Per-trial p_c distribution
    pp_arr = np.array([p for p in pc_plus_per_trial if p is not None and p > 0])
    pm_arr = np.array([p for p in pc_minus_per_trial if p is not None and p > 0])
    if len(pp_arr) > 1:
        print(f"  p_c(+) trial-by-trial:  mean={pp_arr.mean():.6f}, std={pp_arr.std():.6f}")
    if len(pm_arr) > 1:
        print(f"  p_c(-) trial-by-trial:  mean={pm_arr.mean():.6f}, std={pm_arr.std():.6f}")

    # SSB diagnostic: trial-to-trial spread of S_max difference
    diff = final_S_plus - final_S_minus
    print(f"\n  Final state per trial:")
    print(f"    mean(S_max^+) = {final_S_plus.mean():,.0f}  (std={final_S_plus.std():,.0f})")
    print(f"    mean(S_max^-) = {final_S_minus.mean():,.0f}  (std={final_S_minus.std():,.0f})")
    print(f"    mean|S^+ - S^-| per trial = {np.abs(diff).mean():,.0f}  ({100*np.abs(diff).mean()/N_valid:.4f}% of N)")
    print(f"    Winner counts: + won in {sum(1 for w in output['final_winner_per_trial'] if w=='plus')}/{args.trials} trials")
    print(f"                   - won in {sum(1 for w in output['final_winner_per_trial'] if w=='minus')}/{args.trials} trials")
    print(f"    SSB signal:    std/mean of S^+ = {final_S_plus.std()/final_S_plus.mean() if final_S_plus.mean()>0 else 0:.3f}")
    print(f"                   (large std/mean ~ bimodal winner/loser distribution = SSB)")

    if args.mode == 1:
        n_rej = arrays["n_rejected"].mean(axis=0)
        print(f"\n  Total rejected (annihilated) at end: {int(n_rej[-1])} ({100*n_rej[-1]/N_valid:.2f}%)")
    elif args.mode == 2:
        n_2 = arrays["n_2fold"].mean(axis=0)
        n_3 = arrays["n_3fold"].mean(axis=0)
        print(f"\n  Total 2-fold annihilations: {int(n_2[-1])}")
        print(f"  Total 3-fold mergers:       {int(n_3[-1])}")
        if n_2[-1] + n_3[-1] > 0:
            frac3 = n_3[-1] / (n_2[-1] + n_3[-1])
            print(f"  Fraction 3-fold of total mergers: {frac3:.4f}")

    out_path = args.output or f"e8_2color_L{L}_mode{args.mode}_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved JSON to {out_path}")

    if args.save_pertrial_curves:
        npz_path = out_path.replace(".json", "_pertrial.npz")
        np.savez_compressed(
            npz_path,
            bin_centers=bin_centers,
            **{k: arrays[k] for k in keys},
        )
        print(f"Saved per-trial curves to {npz_path}")


if __name__ == "__main__":
    main()
