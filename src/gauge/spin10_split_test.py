#!/usr/bin/env python3
"""
spin10_split_test.py
====================
Test whether the integer/spinor type composition of the 27 matter-rich
Z_3-orbit representatives reproduces the Spin(10) decomposition
  27 = 16 + 10 + 1
of the E_6 matter representation.

Suggestive pre-count from the type composition of the 162 oriented
matter-rich triangles:
  66 (I,I,I) + 96 (I,S,S)  (oriented)
 -> 33 (I,I,I) + 48 (I,S,S)  (unoriented A_2's)
 -> 11 (I,I,I) + 16 (I,S,S)  Z_3-orbit reps

11 + 16 = 27; 16 = dim(16 spinor); 11 = 10 + 1 = dim(vector-like).
If matter (the 16) is purely spinor-content in E_8 and vector-like
(10+1) is purely integer-content, this is the explicit geometric
realisation of Spin(10) inside the cycle catalogue.

Then we look for a further invariant that splits the 11 (I,I,I) reps
into 10 + 1.
"""

import numpy as np
import itertools
from itertools import combinations_with_replacement as cwr
from collections import Counter

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

# ---- harmonic grades and V_112 / Z_3 split ----
def eval_monomials(R, deg):
    N = R.shape[0]
    if deg == 0:
        return np.ones((N, 1))
    cols = []
    for combo in cwr(range(8), deg):
        col = np.ones(N)
        for idx in combo:
            col = col * R[:, idx]
        cols.append(col)
    return np.array(cols).T

def harmonic_grades(R):
    cum = None
    grades = []
    for d in range(5):
        Md = eval_monomials(R, d)
        if cum is not None:
            Md = Md - cum @ (cum.T @ Md)
        U, s, _ = np.linalg.svd(Md, full_matrices=False)
        V = U[:, s > 1e-9 * s[0]]
        grades.append(V)
        cum = V if cum is None else np.hstack([cum, V])
    return grades

def reflect_matrix(a):
    a = np.asarray(a, dtype=float)
    return np.eye(8) - 2.0 * np.outer(a, a) / (a @ a)

def a2_coxeter_order3():
    a = np.zeros(8); a[0] = 1; a[1] = -1
    b = np.zeros(8); b[1] = 1; b[2] = -1
    return reflect_matrix(a) @ reflect_matrix(b)

# ---- main ----
def main():
    R = e8_roots()
    keys = {root_key(r): i for i, r in enumerate(R)}
    grades = harmonic_grades(R)
    V112 = grades[3]

    C = a2_coxeter_order3()
    perm_idx = np.zeros(len(R), dtype=int)
    for i, r in enumerate(R):
        perm_idx[i] = keys[root_key(C @ r)]

    # Build the perm operator on 240 then on V_112 to define Z_3 projectors
    perm = np.zeros((len(R), len(R)))
    for i, r in enumerate(R):
        perm[keys[root_key(C @ r)], i] = 1.0
    A = V112.T @ (perm @ V112)
    om = np.exp(2j*np.pi/3); omb = np.conj(om)
    I = np.eye(A.shape[0])
    P_1   = (I + A + A@A) / 3.0
    P_om  = (I + omb*A + om*(A@A)) / 3.0
    P_omb = (I + om*A + omb*(A@A)) / 3.0

    # Find all 120-degree triangles
    tris = []
    for i in range(len(R)):
        for j in range(i+1, len(R)):
            s = -(R[i] + R[j])
            k = keys.get(root_key(s))
            if k is not None and k > j:
                tris.append((i, j, k))

    # Classify by Z_3 signature and pick matter-rich
    matter_rich_oriented = []
    for (i, j, k) in tris:
        v = np.zeros(240); v[i]=v[j]=v[k]=1.0
        c = V112.T @ v
        s58  = float(np.linalg.norm(P_1 @ c)**2)
        s27  = float(np.linalg.norm(P_om @ c)**2)
        s27b = float(np.linalg.norm(P_omb @ c)**2)
        sig = (round(s58, 3), round(s27, 3), round(s27b, 3))
        if sig == (0.167, 0.667, 0.667):
            matter_rich_oriented.append((i, j, k))
    print(f"Matter-rich oriented triangles: {len(matter_rich_oriented)} (expect 162)")

    # Reduce to unoriented A_2 subsystems
    def a2_of(i, j, k):
        s = {i, j, k}
        for idx in (i, j, k):
            neg_key = root_key(-R[idx])
            s.add(keys[neg_key])
        return frozenset(s)
    matter_a2s = set()
    for tri in matter_rich_oriented:
        matter_a2s.add(a2_of(*tri))
    print(f"Matter-rich A_2 subsystems: {len(matter_a2s)} (expect 81)")

    # Z_3 orbits on A_2's
    def apply_C(a2): return frozenset(perm_idx[i] for i in a2)
    seen = set(); orbits = []
    for a2 in matter_a2s:
        if a2 in seen: continue
        orb = []; cur = a2
        while cur not in seen:
            seen.add(cur); orb.append(cur); cur = apply_C(cur)
        orbits.append(orb)
    print(f"Z_3 orbits among matter-rich A_2's: {len(orbits)} (expect 27)")

    # ---- type composition of each orbit (representatives) ----
    def a2_pos_roots(a2):
        # 6 indices: 3 positive + 3 negative. Pick the 'positive' half by lex order
        idxs = sorted(a2)
        # Use first 3 as 'positive' (an arbitrary but consistent choice)
        # Actually identify the 3 with alpha+beta+gamma=0 sum
        from itertools import combinations
        for trip in combinations(idxs, 3):
            s = R[trip[0]] + R[trip[1]] + R[trip[2]]
            if np.allclose(s, 0):
                return list(trip)
        return None

    type_counts = Counter()
    for orb in orbits:
        rep = orb[0]
        pos = a2_pos_roots(rep)
        sig = tuple(sorted(root_type(R[i]) for i in pos))
        type_counts[sig] += 1
    print(f"\nType composition of the 27 Z_3 orbit representatives:")
    for sig, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {sig}: {n}")

    iss_orbits = [orb for orb in orbits
                  if tuple(sorted(root_type(R[i]) for i in a2_pos_roots(orb[0])))
                     == ('I','S','S')]
    iii_orbits = [orb for orb in orbits
                  if tuple(sorted(root_type(R[i]) for i in a2_pos_roots(orb[0])))
                     == ('I','I','I')]
    print(f"\n(I,S,S) orbits: {len(iss_orbits)}  <- candidate 16 of Spin(10) (matter)")
    print(f"(I,I,I) orbits: {len(iii_orbits)}  <- candidate 10+1 (vector-like)")

    # ---- look for sub-invariant splitting 11 (I,I,I) into 10 + 1 ----
    # Natural candidate: the 'planar coordinate support' of the I,I,I A_2's.
    # An (I,I,I) A_2 spans a 2-plane in coordinate space. Which planes appear?
    print(f"\nFurther structure on the 11 (I,I,I) orbits:")
    plane_sigs = []
    for orb in iii_orbits:
        rep = orb[0]
        pos = a2_pos_roots(rep)
        # Take the set of coordinates that are nonzero in the 3 roots
        coords_used = set()
        for idx in pos:
            for c in range(8):
                if abs(R[idx][c]) > 0.5:
                    coords_used.add(c)
        plane_sigs.append(tuple(sorted(coords_used)))
    plane_counts = Counter(plane_sigs)
    print(f"  Coordinate-support patterns of (I,I,I) orbits: {len(plane_counts)} distinct")
    for plane, n in sorted(plane_counts.items(), key=lambda x: -x[1]):
        print(f"    coords {plane}: {n} orbits")

    # Try another invariant: stabilizer size proxy via the orbit's W(E_8) reflection-closure
    # within the matter_rich set
    def reflect_a2(a2, alpha_idx):
        a = R[alpha_idx]
        new = []
        for idx in a2:
            r = R[idx]
            rf = r - 2.0*(r @ a)/(a @ a) * a
            j = keys.get(root_key(rf))
            if j is None: return None
            new.append(j)
        return frozenset(new)

    print(f"\nW(E_8)-stabiliser-orbit sizes within matter_rich, starting from each (I,I,I) rep:")
    sizes = []
    for orb in iii_orbits:
        seed = orb[0]
        visited = {seed}; frontier = {seed}
        # restrict reflections to a SPECIFIC stabiliser of the framework's A_2:
        # reflections through roots ORTHOGONAL to span(e1-e2, e2-e3)
        # i.e. roots r with r[0]=r[1]=r[2] (zero or equal)
        # but actually let's just check W(E_8) restricted to matter_rich
        while frontier:
            nf = set()
            for a2 in frontier:
                for alpha_idx in range(len(R)):
                    img = reflect_a2(a2, alpha_idx)
                    if img and img not in visited and img in matter_a2s:
                        visited.add(img); nf.add(img)
            frontier = nf
        sizes.append(len(visited))
    print(f"  (I,I,I) full-mr orbits: counter = {Counter(sizes)}")

    # Restrict to reflections only through roots in the orthogonal complement
    # of the framework's A_2 plane.  Such roots have coordinates with
    # the first 3 components equal (zero or all equal half-integer).
    def in_a2_orthogonal_complement(r):
        # Vector in span(e_1+e_2+e_3, e_4..e_8): means r[0]=r[1]=r[2]
        return abs(r[0]-r[1])<1e-6 and abs(r[1]-r[2])<1e-6
    stab_root_idx = [i for i,r in enumerate(R) if in_a2_orthogonal_complement(r)]
    print(f"\nRoots in A_2-orthogonal complement (E_6 root candidates): {len(stab_root_idx)} (expect 72)")
    sizes_stab = []
    for orb in iii_orbits:
        seed = orb[0]
        visited = {seed}; frontier = {seed}
        while frontier:
            nf = set()
            for a2 in frontier:
                for alpha_idx in stab_root_idx:
                    img = reflect_a2(a2, alpha_idx)
                    if img and img not in visited and img in matter_a2s:
                        visited.add(img); nf.add(img)
            frontier = nf
        sizes_stab.append(len(visited))
    print(f"  (I,I,I) E_6-stabiliser orbits: counter = {Counter(sizes_stab)}")
    sizes_iss = []
    for orb in iss_orbits:
        seed = orb[0]
        visited = {seed}; frontier = {seed}
        while frontier:
            nf = set()
            for a2 in frontier:
                for alpha_idx in stab_root_idx:
                    img = reflect_a2(a2, alpha_idx)
                    if img and img not in visited and img in matter_a2s:
                        visited.add(img); nf.add(img)
            frontier = nf
        sizes_iss.append(len(visited))
    print(f"  (I,S,S) E_6-stabiliser orbits: counter = {Counter(sizes_iss)}")

if __name__ == "__main__":
    main()
