#!/usr/bin/env python3
"""
berry_and_su5_split.py
======================
Stage 2 + SU(5) sub-split of the matter-cycle catalogue.

Three computations on the 27 matter-rich Z_3-orbit representatives:

(a) BERRY-LIKE INVARIANT for each cycle:
    Project the cycle's 3 vertices onto the 3-dim 'family' space
       f_1 = (e_1-e_2)/sqrt(2)       (1st A_2 root direction)
       f_2 = (e_1+e_2-2 e_3)/sqrt(6) (2nd A_2 root direction)
       f_3 = (e_1+e_2+e_3)/sqrt(3)   (perpendicular U(1) within (e_1,e_2,e_3))
    Compute the spherical triangle area on the unit sphere of these
    projections (normalised). This is the natural framework-A_2 Berry proxy.

(b) SU(5) sub-split of the 16 (I,S,S) orbit reps:
    Look for an invariant that splits 16 -> 10 + 5(bar) + 1.
    Natural candidates: integer-root support inside vs outside the family
    A_2 plane; spinor-root parity (number of minus signs); 'U(1) charge'
    proxy via integer-root coordinates.

(c) Doubled (0,2,4) support in the 10:
    Identify the 2 (I,I,I) orbit reps with coordinate support {0,2,4}
    and characterise what distinguishes them (e.g. their integer-root sign
    pattern, paired by some Z_2 = 5 vs 5bar of SU(5)?).
"""

import numpy as np
import itertools
from itertools import combinations_with_replacement as cwr, combinations
from collections import Counter, defaultdict

# ---- E_8 roots and helpers (as before) ----
def e8_roots():
    roots = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8)
                v[i], v[j] = si, sj
                roots.append(v)
    for s in itertools.product((0.5, -0.5), repeat=8):
        if sum(1 for x in s if x < 0) % 2 == 0:
            roots.append(np.array(s))
    return np.array(roots)

def root_key(v, scale=2):
    return tuple(int(round(scale*x)) for x in v)

def root_type(v):
    return 'I' if np.max(np.abs(v)) > 0.75 else 'S'

def eval_monomials(R, deg):
    N = R.shape[0]
    if deg == 0: return np.ones((N, 1))
    cols = []
    for combo in cwr(range(8), deg):
        col = np.ones(N)
        for idx in combo: col *= R[:, idx]
        cols.append(col)
    return np.array(cols).T

def harmonic_grades(R):
    cum = None; grades = []
    for d in range(5):
        Md = eval_monomials(R, d)
        if cum is not None: Md = Md - cum @ (cum.T @ Md)
        U, s, _ = np.linalg.svd(Md, full_matrices=False)
        V = U[:, s > 1e-9 * s[0]]
        grades.append(V)
        cum = V if cum is None else np.hstack([cum, V])
    return grades

def reflect_matrix(a):
    a = np.asarray(a, dtype=float)
    return np.eye(8) - 2.0 * np.outer(a, a) / (a @ a)

def a2_coxeter_order3():
    a = np.zeros(8); a[0]=1; a[1]=-1
    b = np.zeros(8); b[1]=1; b[2]=-1
    return reflect_matrix(a) @ reflect_matrix(b)

# ---- spherical triangle area (Girard formula) ----
def spherical_triangle_area(a, b, c):
    """Signed area of spherical triangle with vertices a, b, c on unit S^2."""
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b); c = c/np.linalg.norm(c)
    # L'Huilier formula via half-angle: use ord cosines for robustness
    # Use Van Oosterom-Strackee formula:
    num = np.dot(a, np.cross(b, c))
    den = 1 + a@b + b@c + c@a
    return 2*np.arctan2(num, den)

# ---- get the 27 matter-rich orbit reps from scratch ----
def get_matter_rich_orbits():
    R = e8_roots()
    keys = {root_key(r): i for i, r in enumerate(R)}
    grades = harmonic_grades(R)
    V112 = grades[3]
    C = a2_coxeter_order3()
    perm_idx = np.zeros(len(R), dtype=int)
    perm = np.zeros((len(R), len(R)))
    for i, r in enumerate(R):
        j = keys[root_key(C @ r)]
        perm_idx[i] = j
        perm[j, i] = 1.0
    A = V112.T @ (perm @ V112)
    om = np.exp(2j*np.pi/3); omb = np.conj(om)
    I8 = np.eye(A.shape[0])
    P_1   = (I8 + A + A@A) / 3.0
    P_om  = (I8 + omb*A + om*(A@A)) / 3.0
    P_omb = (I8 + om*A + omb*(A@A)) / 3.0

    matter_oriented = []
    for i in range(len(R)):
        for j in range(i+1, len(R)):
            s = -(R[i] + R[j])
            k = keys.get(root_key(s))
            if k is not None and k > j:
                v = np.zeros(240); v[i]=v[j]=v[k]=1.0
                c = V112.T @ v
                sig = (round(float(np.linalg.norm(P_1@c)**2),3),
                       round(float(np.linalg.norm(P_om@c)**2),3),
                       round(float(np.linalg.norm(P_omb@c)**2),3))
                if sig == (0.167, 0.667, 0.667):
                    matter_oriented.append((i, j, k))

    # A_2 subsystems (closed under negation)
    def a2_of(i, j, k):
        s = {i, j, k}
        for idx in (i, j, k):
            s.add(keys[root_key(-R[idx])])
        return frozenset(s)

    a2_set = set()
    for tri in matter_oriented:
        a2_set.add(a2_of(*tri))

    # Z_3 orbits: collect reps
    def apply_C(a2): return frozenset(perm_idx[i] for i in a2)
    seen = set(); orbits = []
    for a2 in a2_set:
        if a2 in seen: continue
        orb = []; cur = a2
        while cur not in seen:
            seen.add(cur); orb.append(cur); cur = apply_C(cur)
        orbits.append(orb)
    return R, keys, orbits

def a2_pos_roots(R, a2):
    idxs = sorted(a2)
    for trip in combinations(idxs, 3):
        if np.allclose(R[trip[0]]+R[trip[1]]+R[trip[2]], 0):
            return list(trip)
    return None

# ---- main ----
def main():
    R, keys, orbits = get_matter_rich_orbits()
    print(f"Matter-rich Z_3 orbit representatives: {len(orbits)} (expect 27)\n")

    # Family 3-D basis
    f1 = np.zeros(8); f1[0]=1; f1[1]=-1; f1 /= np.sqrt(2)
    f2 = np.zeros(8); f2[0]=1; f2[1]=1; f2[2]=-2; f2 /= np.sqrt(6)
    f3 = np.zeros(8); f3[0]=1; f3[1]=1; f3[2]=1;  f3 /= np.sqrt(3)
    F = np.array([f1, f2, f3])

    # Partition orbits by type
    iss_orbits = []
    iii_orbits = []
    for orb in orbits:
        rep = orb[0]
        pos = a2_pos_roots(R, rep)
        sig = tuple(sorted(root_type(R[i]) for i in pos))
        if sig == ('I','S','S'): iss_orbits.append((orb, pos))
        elif sig == ('I','I','I'): iii_orbits.append((orb, pos))

    print(f"(I,S,S) orbits: {len(iss_orbits)}   (I,I,I) orbits: {len(iii_orbits)}\n")

    # ============================================================
    # (a) BERRY-LIKE INVARIANT (spherical triangle area in family 3-D)
    # ============================================================
    print("===== (a) Family-projection Berry-like areas =====")
    print("Spherical triangle area of cycle vertices projected onto")
    print("the family 3-space spanned by (e_1-e_2, e_1+e_2-2e_3, e_1+e_2+e_3).\n")

    def family_proj_area(pos):
        verts = []
        for idx in pos:
            v3 = F @ R[idx]
            n = np.linalg.norm(v3)
            if n < 1e-9: return 0.0
            verts.append(v3 / n)
        return abs(spherical_triangle_area(*verts))

    iss_areas = [(family_proj_area(p), p) for _, p in iss_orbits]
    iii_areas = [(family_proj_area(p), p) for _, p in iii_orbits]

    iss_area_counts = Counter(round(a, 4) for a, _ in iss_areas)
    iii_area_counts = Counter(round(a, 4) for a, _ in iii_areas)
    print(f"(I,S,S) family-Berry-area distribution: {dict(iss_area_counts)}")
    print(f"(I,I,I) family-Berry-area distribution: {dict(iii_area_counts)}")

    # ============================================================
    # (b) SU(5) SUB-SPLIT of the 16 (I,S,S) orbits
    # ============================================================
    print("\n===== (b) Sub-split of the 16 (I,S,S) orbits =====")
    # For each (I,S,S) cycle, identify the integer root and the 2 spinor roots.
    # Compute several invariants.

    def iss_breakdown(pos):
        """Returns (int_root, spinor_root_1, spinor_root_2)."""
        for idx in pos:
            if root_type(R[idx]) == 'I':
                int_root = R[idx]
                others = [R[k] for k in pos if k != idx]
                return int_root, others
        return None, None

    print("\n--- Integer-root support (which 2 of 8 coords are nonzero):")
    int_support_counts = Counter()
    for _, pos in iss_orbits:
        ir, _ = iss_breakdown(pos)
        sup = tuple(sorted(i for i,x in enumerate(ir) if abs(x)>0.5))
        int_support_counts[sup] += 1
    for sup, n in sorted(int_support_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  coords {sup}: {n} orbits")

    print("\n--- Integer-root in/out of family A_2 plane (coords {0,1,2}):")
    a2_plane = {0,1,2}
    for cat, fn in [
        ("entirely_in", lambda sup: set(sup).issubset(a2_plane)),
        ("one_in_one_out", lambda sup: len(set(sup) & a2_plane)==1),
        ("entirely_out", lambda sup: set(sup).isdisjoint(a2_plane)),
    ]:
        cnt = sum(1 for sup, n in int_support_counts.items() if fn(sup) for _ in range(n))
        print(f"  {cat}: {cnt} orbits")

    print("\n--- Spinor-root parity (# of minus signs, sum over the 2 spinors):")
    parity_counts = Counter()
    for _, pos in iss_orbits:
        _, spinors = iss_breakdown(pos)
        total_minus = sum(int(sum(1 for x in s if x<0)) for s in spinors)
        parity_counts[total_minus] += 1
    print(f"  {dict(parity_counts)}")

    print("\n--- Integer-root sign pattern (++, +-, -+, --):")
    sign_counts = Counter()
    for _, pos in iss_orbits:
        ir, _ = iss_breakdown(pos)
        signs = tuple(sorted([int(np.sign(x)) for i,x in enumerate(ir) if abs(x)>0.5]))
        sign_counts[signs] += 1
    print(f"  {dict(sign_counts)}")

    # ============================================================
    # (c) The (0,2,4) doubled support in the 10
    # ============================================================
    print("\n===== (c) Doubled (0,2,4) support in the 10 =====")
    doubled_orbits = []
    for orb, pos in iii_orbits:
        coords = set()
        for idx in pos:
            for c in range(8):
                if abs(R[idx][c]) > 0.5: coords.add(c)
        if tuple(sorted(coords)) == (0,2,4):
            doubled_orbits.append((orb, pos))
    print(f"Found {len(doubled_orbits)} orbits with support (0,2,4):")
    for k, (orb, pos) in enumerate(doubled_orbits):
        print(f"\nOrbit #{k+1}, positive roots:")
        for idx in pos:
            print(f"  {R[idx]}")
        # Distinguishing features: sign pattern of the 3 roots
        signs = []
        for idx in pos:
            sig = tuple(int(np.sign(x)) for i,x in enumerate(R[idx]) if abs(x)>0.5)
            signs.append(sig)
        print(f"  sign patterns (per root): {signs}")

    # ============================================================
    # PUTTING IT TOGETHER: 16 = 10 + 5bar + 1 candidates
    # ============================================================
    print("\n===== Synthesis: candidate 16 = 10 + 5bar + 1 split =====")
    # The integer-root support inside/outside the A_2 plane is the natural
    # 'SU(5) charge' candidate. Let's combine with spinor parity.
    classes = defaultdict(list)
    for orb, pos in iss_orbits:
        ir, sp = iss_breakdown(pos)
        int_sup = tuple(sorted(i for i,x in enumerate(ir) if abs(x)>0.5))
        in_a2 = len(set(int_sup) & a2_plane)
        out_a2 = len(set(int_sup) - a2_plane)
        parity = sum(int(sum(1 for x in s if x<0)) for s in sp)
        key = (in_a2, out_a2, parity)
        classes[key].append((orb, pos))
    print(f"Combined-invariant classes (int_in_A2, int_out_A2, spinor_parity):")
    for key, members in sorted(classes.items()):
        print(f"  {key}: {len(members)} orbits")

if __name__ == "__main__":
    main()
