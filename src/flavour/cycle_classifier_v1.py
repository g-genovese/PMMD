#!/usr/bin/env python3
"""
cycle_classifier_v1.py
======================
Step 1 of the flavour-soliton programme: classify the minimal closed cycles
(120-degree triangles) on the E_8 root graph by their character under V_112
and its A_2-Coxeter Z_3 decomposition 58 (+) 27 (+) 27bar.

For each triangle:
  - indicator vector on the 240 roots (3 ones, 237 zeros)
  - project onto V_112 (degree-3 harmonic component on the roots)
  - decompose the projection into 58 / 27 / 27bar parts
  - record the squared norms (the "Z_3 character" of the triangle)

If the framework's three-generation identification is right, the triangles
should split into orbits under W(E_8) whose Z_3 fingerprints carry generational
structure. The triangles maximally in 27 (or 27bar) are the candidates for the
matter-loop side of the catalogue.
"""

import numpy as np
import itertools
from itertools import combinations_with_replacement as cwr
from collections import Counter

# -----------------------------------------------------------------------------
# E_8 roots (240) and integer-shift trick to make them hashable
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# V_112 = degree-3 harmonics on the 240 roots
# (degrees 0,1,2,3,4 give 1+8+35+112+84 = 240, the full permutation rep)
# -----------------------------------------------------------------------------

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

def orthonormal_complement_after(R, prev_components):
    """Build orthonormal basis of harmonic-grade by subtracting previous grades."""
    pass

def harmonic_grades(R):
    """Return orthonormal bases V_d (240 x dim_d) for d = 0,1,2,3,4."""
    cum_basis = None
    grades = []
    for d in range(5):
        Md = eval_monomials(R, d)
        if cum_basis is not None:
            # subtract projection on previous grades
            Md = Md - cum_basis @ (cum_basis.T @ Md)
        U, s, _ = np.linalg.svd(Md, full_matrices=False)
        keep = s > 1e-9 * s[0]
        V = U[:, keep]
        grades.append(V)
        cum_basis = V if cum_basis is None else np.hstack([cum_basis, V])
    return grades  # list of 5 ON bases

# -----------------------------------------------------------------------------
# A_2-Coxeter element of order 3 in W(E_8) (used in v112_construction.py)
# Build it as product of three commuting reflections that has order 3
# -----------------------------------------------------------------------------

def reflect_matrix(a):
    a = np.asarray(a, dtype=float)
    return np.eye(8) - 2.0 * np.outer(a, a) / (a @ a)

def a2_coxeter_order3():
    """A Coxeter element of an A_2 (sub)system: product of two reflections in
    roots at 120 degrees, which has order 3 (= h_{A2})."""
    a = np.zeros(8); a[0] = 1; a[1] = -1            # root e_1 - e_2 (norm sq 2)
    b = np.zeros(8); b[1] = 1; b[2] = -1            # root e_2 - e_3 (norm sq 2, 120 deg from a)
    return reflect_matrix(a) @ reflect_matrix(b)

# -----------------------------------------------------------------------------
# Find all 120-degree triangles (alpha + beta + gamma = 0, all in roots)
# -----------------------------------------------------------------------------

def find_triangles(roots):
    keys = {root_key(r): i for i, r in enumerate(roots)}
    seen = set()
    tris = []
    for i in range(len(roots)):
        for j in range(i+1, len(roots)):
            s = -(roots[i] + roots[j])
            k = root_key(s)
            if k in keys:
                idx = keys[k]
                if idx > j:
                    tris.append((i, j, idx))
                    seen.add(tuple(sorted((i, j, idx))))
    return tris

# -----------------------------------------------------------------------------
# Project each triangle's indicator onto V_112, then decompose by Z_3 character
# -----------------------------------------------------------------------------

def main():
    R = e8_roots()
    print(f"E_8 roots: {len(R)}")
    print(f"Coxeter A_2 element order (sanity): "
          f"{np.linalg.matrix_power(a2_coxeter_order3(), 3).round(6)[0,0]:.0f} on diag (should be 1)")

    grades = harmonic_grades(R)
    dims = [g.shape[1] for g in grades]
    print(f"Harmonic grades dim: {dims}  (expect 1,8,35,112,84)")
    assert dims == [1, 8, 35, 112, 84], "harmonic decomposition unexpected"

    V112 = grades[3]                  # 240 x 112
    V_const = grades[0]               # 240 x 1 (scale mode = uniform)

    # ---- Z_3 split of V_112 using the A_2 Coxeter ----
    C = a2_coxeter_order3()           # 8x8 orthogonal, order 3
    # Build its action on the 240-root permutation rep
    # by mapping each root through C and looking up the resulting permutation index
    keys = {root_key(r): i for i, r in enumerate(R)}
    perm = np.zeros((len(R), len(R)))
    for i, r in enumerate(R):
        r_im = C @ r
        j = keys[root_key(r_im)]
        perm[j, i] = 1.0

    # Restrict permutation to V_112 (in its ON basis)
    A_on_V112 = V112.T @ (perm @ V112)
    eig, vec = np.linalg.eig(A_on_V112)
    # Expected eigenvalues: 1 (58x), omega (27x), omega-bar (27x)
    eig_round = [complex(round(e.real, 4), round(e.imag, 4)) for e in eig]
    cnt = Counter(eig_round)
    print(f"V_112 eigenvalues under A_2-Coxeter (counts): {dict(cnt)}")

    # Build projectors onto the three Z_3 eigenspaces of V_112
    # P_1 = (1 + A + A^2)/3 ; P_omega = (1 + omega_bar A + omega A^2)/3 ; etc.
    om = np.exp(2j*np.pi/3)
    om_bar = np.conj(om)
    I = np.eye(A_on_V112.shape[0])
    A = A_on_V112
    A2 = A @ A
    P_1 = (I + A + A2) / 3.0
    P_om = (I + om_bar*A + om*A2) / 3.0
    P_omb = (I + om*A + om_bar*A2) / 3.0
    rks = [np.linalg.matrix_rank(P_1, tol=1e-6),
           np.linalg.matrix_rank(P_om, tol=1e-6),
           np.linalg.matrix_rank(P_omb, tol=1e-6)]
    print(f"Ranks of Z_3 projectors on V_112: {rks}  (expect 58, 27, 27)")

    # ---- 120-degree triangles ----
    tris = find_triangles(R)
    print(f"Number of distinct 120 deg triangles: {len(tris)}  "
          f"(expect 240*56/(3*2) = {240*56//6})")

    # ---- Project each triangle's indicator onto V_112 and split by Z_3 ----
    chars = []
    for (i, j, k) in tris:
        v = np.zeros(240)
        v[i] = v[j] = v[k] = 1.0
        coefs = V112.T @ v                 # 112-dim coef vector
        # Decompose into Z_3 components in the V_112 basis
        c_1   = P_1   @ coefs
        c_om  = P_om  @ coefs
        c_omb = P_omb @ coefs
        n_total = np.linalg.norm(coefs)**2
        n_1     = np.linalg.norm(c_1)**2
        n_om    = np.linalg.norm(c_om)**2
        n_omb   = np.linalg.norm(c_omb)**2
        # overlap with scale mode (constant): always 3/sqrt(240) for any triangle
        a_const = float((V_const.T @ v).flatten()[0])
        chars.append((n_total, n_1, n_om, n_omb, a_const))

    chars = np.array(chars).real
    # All triangles look the same up to W(E_8) symmetry; let's verify by stats
    print("\nTriangle Z_3 character statistics over all triangles:")
    labels = ["||V_112 proj||^2", "||58-part||^2", "||27-part||^2", "||27bar-part||^2", "overlap with const"]
    for k, lab in enumerate(labels):
        col = chars[:, k]
        print(f"  {lab:24s}: min={col.min():.4f} max={col.max():.4f} mean={col.mean():.4f}")
    # Cluster: count distinct (n_1, n_om, n_omb) signatures (rounded)
    sigs = Counter()
    for row in chars:
        sigs[(round(row[1], 4), round(row[2], 4), round(row[3], 4))] += 1
    print(f"\nDistinct Z_3 signatures among triangles: {len(sigs)}")
    for sig, count in sorted(sigs.items(), key=lambda x: -x[1])[:10]:
        print(f"  signature (58, 27, 27bar) = {sig}  count = {count}")

    # ---- Deeper analysis: separate triangles by signature class and inspect geometry ----
    classes = {
        "pure_58":       (1.5,    0.0,    0.0),
        "matter_rich":   (0.1667, 0.6667, 0.6667),
        "mid_A":         (0.8333, 0.3333, 0.3333),
        "mid_B":         (0.5,    0.5,    0.5),
    }
    by_class = {name: [] for name in classes}
    for k, (i, j, kk) in enumerate(tris):
        row = chars[k]
        sig = (round(row[1], 4), round(row[2], 4), round(row[3], 4))
        for name, target in classes.items():
            if all(abs(sig[m] - target[m]) < 1e-3 for m in range(3)):
                by_class[name].append((i, j, kk))
                break
    print("\nClass sizes (sanity):", {k: len(v) for k, v in by_class.items()})

    def root_type(v):
        # 'I' for integer (length sq = 2 with two +/- 1 entries),
        # 'S' for spinor (eight +/- 1/2)
        if np.allclose(np.abs(v) - np.array([1 if abs(x) > 0.4 else 0 for x in v]), 0, atol=1e-6):
            return 'I'
        return 'S'

    for name in ["pure_58", "matter_rich"]:
        tlist = by_class[name]
        print(f"\n=== Class '{name}' ({len(tlist)} triangles) ===")
        # Type composition of triangle vertices
        type_sigs = Counter()
        for (i, j, kk) in tlist:
            tsig = tuple(sorted([root_type(R[i]), root_type(R[j]), root_type(R[kk])]))
            type_sigs[tsig] += 1
        print(f"  Vertex-type composition: {dict(type_sigs)}")
        # Distinct dot-product spectra between the 3 vertices
        dot_sigs = Counter()
        for (i, j, kk) in tlist:
            dps = tuple(sorted([round(R[i]@R[j], 4), round(R[j]@R[kk], 4), round(R[i]@R[kk], 4)]))
            dot_sigs[dps] += 1
        print(f"  Pairwise dot products: {dict(dot_sigs)} (expect all -1: 120 deg pairs)")
        # W(E_8) orbit test: how many distinct triangle types under reflections of one root?
        # Quick proxy: sum-of-squared-distance-from-origin signature
        # All roots have ||r||^2 = 2 so this is trivial; use a more informative invariant:
        # the rank of the 3x8 matrix of vertices (= 2 if A_2 plane, 3 if a higher A_2 chain)
        ranks = Counter()
        for (i, j, kk) in tlist:
            M = np.array([R[i], R[j], R[kk]])
            ranks[np.linalg.matrix_rank(M, tol=1e-6)] += 1
        print(f"  Rank of triangle's 3x8 vertex matrix: {dict(ranks)} "
              f"(rank 2 = lies in an A_2 plane)")
        # Plane containment: triangle ALWAYS lies in a 2-plane since alpha+beta+gamma=0
        # So rank should be 2 always. The interesting question is which 2-plane and its symmetries.
        # For matter_rich: list a few example triangles to inspect
        if name == "matter_rich":
            print(f"  Examples of matter-rich triangles (root indices):")
            for (i, j, kk) in tlist[:5]:
                print(f"    [{i}, {j}, {kk}]  vertices:")
                for idx in (i, j, kk):
                    print(f"      {R[idx]}")
    # ============================================================
    # Orbit analysis: do the 81 matter-rich A_2 subsystems split
    # under W(E_8) into 3 orbits of 27 (= explicit 3 generations)?
    # We test by computing the W(E_8) orbit starting from a representative.
    # ============================================================
    print("\n=== Orbit analysis: matter-rich A_2 subsystems under W(E_8) ===")

    # Represent each A_2 subsystem by the SET of 6 roots (3 positive, 3 negative)
    # Two triangles {alpha,beta,gamma} and {-alpha,-beta,-gamma} share an A_2.
    def a2_from_triangle(i, j, k):
        # 6-element frozenset of root indices: i, j, k and their negatives
        neg = {root_key(-R[idx]): None for idx in (i, j, k)}
        all_idx = list({i, j, k})
        for kkey in neg:
            all_idx.append(keys[kkey])
        return frozenset(all_idx)

    matter_a2s = set()
    for (i, j, kk) in by_class["matter_rich"]:
        matter_a2s.add(a2_from_triangle(i, j, kk))
    print(f"Matter-rich A_2 subsystems: {len(matter_a2s)}  (expect 81)")

    # Build a reflection-generator: reflect through any root.
    # Apply iteratively from a seed A_2, collecting all images.
    def reflect_a2(a2_indices_set, alpha_idx):
        a = R[alpha_idx]
        new_idxs = []
        for idx in a2_indices_set:
            r = R[idx]
            r_refl = r - 2.0 * (r @ a) / (a @ a) * a
            j = keys.get(root_key(r_refl))
            if j is None:
                return None
            new_idxs.append(j)
        return frozenset(new_idxs)

    # Start from one matter-rich A_2 and grow by reflections through ALL roots
    seed = next(iter(matter_a2s))
    orbit = {seed}
    frontier = {seed}
    while frontier:
        new_frontier = set()
        for a2 in frontier:
            for alpha_idx in range(len(R)):
                img = reflect_a2(a2, alpha_idx)
                if img is not None and img not in orbit and img in matter_a2s:
                    orbit.add(img)
                    new_frontier.add(img)
        frontier = new_frontier
    print(f"Orbit of one matter-rich A_2 under W(E_8) (within matter-rich set): {len(orbit)}")
    if len(orbit) == len(matter_a2s):
        print("  -> SINGLE ORBIT: all 81 matter-rich A_2's are W(E_8)-conjugate.")
    elif len(orbit) == 27:
        print("  -> Orbit of size 27! Two more orbits of 27 expected.")
        # find the other orbits
        remaining = matter_a2s - orbit
        seed2 = next(iter(remaining))
        orbit2 = {seed2}
        frontier2 = {seed2}
        while frontier2:
            nf = set()
            for a2 in frontier2:
                for alpha_idx in range(len(R)):
                    img = reflect_a2(a2, alpha_idx)
                    if img is not None and img not in orbit2 and img in matter_a2s:
                        orbit2.add(img)
                        nf.add(img)
            frontier2 = nf
        print(f"  Second orbit size: {len(orbit2)}")
        remaining = remaining - orbit2
        if remaining:
            seed3 = next(iter(remaining))
            orbit3 = {seed3}
            frontier3 = {seed3}
            while frontier3:
                nf = set()
                for a2 in frontier3:
                    for alpha_idx in range(len(R)):
                        img = reflect_a2(a2, alpha_idx)
                        if img is not None and img not in orbit3 and img in matter_a2s:
                            orbit3.add(img)
                            nf.add(img)
                frontier3 = nf
            print(f"  Third orbit size: {len(orbit3)}")
    else:
        print(f"  -> Orbit size {len(orbit)} (neither full 81 nor 27).")

    # Same for pure-58 (121 A_2's: 11 x 11?)
    print("\n=== Orbit analysis: pure-58 A_2 subsystems under W(E_8) ===")
    pure_a2s = set()
    for (i, j, kk) in by_class["pure_58"]:
        pure_a2s.add(a2_from_triangle(i, j, kk))
    print(f"Pure-58 A_2 subsystems: {len(pure_a2s)}  (expect 121)")
    seed = next(iter(pure_a2s))
    orbit = {seed}
    frontier = {seed}
    while frontier:
        new_frontier = set()
        for a2 in frontier:
            for alpha_idx in range(len(R)):
                img = reflect_a2(a2, alpha_idx)
                if img is not None and img not in orbit and img in pure_a2s:
                    orbit.add(img)
                    new_frontier.add(img)
        frontier = new_frontier
    print(f"Orbit of one pure-58 A_2 under W(E_8) (within pure-58 set): {len(orbit)}")

    # ============================================================
    # The KEY TEST: framework's Z_3 (the A_2-Coxeter that defined
    # the 58/27/27bar split) acting on the 81 matter-rich A_2's.
    # If orbits are uniformly size 3, then 81 = 27 x 3 = three generations
    # of 27 matter cycles each, with Z_3 permuting generations.
    # ============================================================
    print("\n=== Framework Z_3 (A_2-Coxeter) action on the 81 matter-rich A_2's ===")

    # Build the index-permutation induced on roots by C
    perm_idx = np.zeros(len(R), dtype=int)
    for i, r in enumerate(R):
        perm_idx[i] = keys[root_key(C @ r)]

    def apply_C_to_a2(a2):
        return frozenset(perm_idx[i] for i in a2)

    matter_orbits = []
    seen = set()
    for a2 in matter_a2s:
        if a2 in seen:
            continue
        orb = []
        cur = a2
        while cur not in seen:
            seen.add(cur)
            orb.append(cur)
            cur = apply_C_to_a2(cur)
        matter_orbits.append(orb)
    orbit_sizes = Counter(len(o) for o in matter_orbits)
    print(f"Orbit sizes under framework's Z_3 (matter-rich): {dict(orbit_sizes)}")
    print(f"Number of orbits: {len(matter_orbits)}")
    if list(orbit_sizes.keys()) == [3]:
        n_orb = len(matter_orbits)
        print(f"  ALL orbits of size 3 -> {n_orb} orbits x 3 = {n_orb*3}.")
        if n_orb == 27:
            print(f"  *** 81 = 27 x 3 confirmed: 27 matter-cycle representatives,")
            print(f"      Z_3 permuting 3 generations. ***")

    # Same test for pure-58
    print("\n=== Framework Z_3 on the 121 pure-58 A_2's ===")
    seen = set()
    pure_orbits = []
    for a2 in pure_a2s:
        if a2 in seen:
            continue
        orb = []
        cur = a2
        while cur not in seen:
            seen.add(cur)
            orb.append(cur)
            cur = apply_C_to_a2(cur)
        pure_orbits.append(orb)
    orbit_sizes = Counter(len(o) for o in pure_orbits)
    print(f"Orbit sizes under framework's Z_3 (pure-58): {dict(orbit_sizes)}")
    print(f"Number of orbits: {len(pure_orbits)}")

if __name__ == "__main__":
    main()
