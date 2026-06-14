#!/usr/bin/env python3
"""
v112_construction.py
--------------------
Explicit-vector construction of the W(E8) irreducible V_112 and verification
of its A2-Coxeter Z3 decomposition, for the PMMD framework (v6.0).

What it shows
=============
1. Grading the permutation representation R_perm of W(E8) on the 240 roots by
   polynomial degree gives, on the roots,
        R_perm = 1 (+) 8 (+) 35 (+) 112 (+) 84      (degrees 0,1,2,3,4),
   so V_112 IS the space of degree-3 harmonic functions on the root system.

2. Applying the A2-Coxeter element (order 3) directly to an explicit orthonormal
   basis of this 112-dim space returns the eigenspace dimensions
        58  (eigenvalue 1) ,  27 (omega) ,  27 (omega-bar) ,
   i.e. the Koide decomposition V_112 = 58 (+) 27 (+) 27bar, reproduced from
   explicit vectors (the v5.2 result was at the character level only).
   This distinguishes V_112 from the graph-Laplacian degeneracy-112 eigenspace,
   whose Z3 split is the non-Koide 40+36+36.

3. Probing the explicit V_112 with natural geometric states (root indicators and
   generic vectors) returns the DEMOCRATIC amplitude ratio c = A_cyc/A_inv ~ 1
   (Koide Q ~ 1/2), NOT the Koide value c = sqrt(2) (Q = 2/3). The Koide value
   is therefore supplied by the maximal-mutual-determination weighting of the
   two isotypic modes, not by the bare geometry of V_112.

Dependencies: numpy only. Runtime: a few seconds.
"""

import numpy as np
import itertools
from itertools import combinations_with_replacement as cwr


def e8_roots():
    """Return the 240 roots of E8 as an array of shape (240, 8)."""
    roots = []
    # 112 integer roots of type (+-1, +-1, 0, ..., 0)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8)
                v[i], v[j] = si, sj
                roots.append(v)
    # 128 spinor roots (+-1/2)^8 with an even number of minus signs
    for s in itertools.product((0.5, -0.5), repeat=8):
        if sum(1 for x in s if x < 0) % 2 == 0:
            roots.append(np.array(s))
    return np.array(roots)


def eval_monomials(R, deg):
    """Evaluate all degree-`deg` monomials in the 8 coordinates on the rows of R."""
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


def onb(M, tol=1e-9):
    """Orthonormal basis of the column space of M (SVD, relative tolerance)."""
    if M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > tol * s[0]]


def reflect(x, a):
    return x - 2.0 * (x @ a) / (a @ a) * a


def main():
    R = e8_roots()
    N = len(R)
    assert N == 240, f"expected 240 roots, got {N}"

    # ---- 1. graded pieces of R_perm by polynomial degree ----
    print("Graded decomposition of R_perm on the 240 roots (by polynomial degree):")
    cum = np.ones((N, 1))
    prev_dim = 1
    pieces = {0: 1}
    for d in range(1, 6):
        stacked = np.hstack([cum, eval_monomials(R, d)])
        Q = onb(stacked)
        dim = Q.shape[1]
        pieces[d] = dim - prev_dim
        print(f"   degree {d}: new dimension = {pieces[d]:4d}   (cumulative {dim})")
        cum, prev_dim = Q, dim
        if dim >= N:
            break
    print(f"   => R_perm = 1 + 8 + 35 + 112 + 84 ; V_112 = degree-3 harmonics\n")

    # ---- explicit orthonormal basis of V_112 (degree-3 piece) ----
    Q2 = onb(np.hstack([eval_monomials(R, d) for d in range(3)]))   # deg <= 2
    Q3 = onb(np.hstack([eval_monomials(R, d) for d in range(4)]))   # deg <= 3
    B = onb(Q3 - Q2 @ (Q2.T @ Q3))                                  # V_112, (240,112)
    print(f"dim V_112 (explicit basis) = {B.shape[1]}")

    # ---- 2. A2-Coxeter element as a permutation of the 240 roots ----
    a = R[0].copy()
    b = next(r for r in R
             if abs((a @ r) / np.sqrt((a @ a) * (r @ r)) + 0.5) < 1e-9)  # 120 deg
    gamma = lambda x: reflect(reflect(x, b), a)                          # order-3 Coxeter
    perm = np.array([np.argmin(np.linalg.norm(R - gamma(r), axis=1)) for r in R])
    assert sorted(perm.tolist()) == list(range(N)), "gamma is not a permutation"

    # action on functions: (gamma . f)(r) = f(gamma^{-1} r)
    Pi = np.zeros((N, N))
    inv = np.argsort(perm)
    for i in range(N):
        Pi[i, inv[i]] = 1.0

    # restrict to V_112 and read off the Z3 spectrum
    Op = B.T @ Pi @ B
    ev = np.linalg.eigvals(Op)
    om = np.exp(2j * np.pi / 3)
    n1 = int(np.sum(np.abs(ev - 1) < 1e-6))
    nw = int(np.sum(np.abs(ev - om) < 1e-6))
    nwb = int(np.sum(np.abs(ev - om.conjugate()) < 1e-6))
    print(f"A2-Coxeter Z3 split of V_112: dim(1)={n1}, dim(omega)={nw}, dim(omega-bar)={nwb}")
    print(f"   => {n1}+{nw}+{nwb}  (Koide 58+27+27 expected; Laplacian mode would give 40+36+36)\n")

    # ---- 3. geometric probe: natural states give the democratic ratio, not sqrt(2) ----
    PV = B @ B.T
    G1, Gg, Gg2 = PV, Pi @ PV, Pi @ Pi @ PV
    P1 = (G1 + Gg + Gg2) / 3
    Pw = (G1 + om.conjugate() * Gg + om * Gg2) / 3
    Pwb = (G1 + om * Gg + om.conjugate() * Gg2) / 3

    def c_ratio(psi):
        psi = PV @ psi
        a_inv = np.linalg.norm(P1 @ psi)
        a_cyc = np.sqrt(np.linalg.norm(Pw @ psi) ** 2 + np.linalg.norm(Pwb @ psi) ** 2)
        return a_cyc / a_inv

    def Q_of_c(c):
        return (1.0 + c ** 2 / 2.0) / 3.0   # c = A_cyc/A_inv ; c=sqrt2 -> Q=2/3

    cs = np.array([c_ratio(np.eye(N)[i]) for i in range(N)])
    rng = np.random.default_rng(0)
    cc = np.array([c_ratio(B @ rng.standard_normal(112)) for _ in range(2000)])
    print("Geometric probe of the explicit V_112:")
    print(f"   root indicators (mean over 240): c = {cs.mean():.4f} -> Q = {Q_of_c(cs.mean()):.4f}")
    print(f"   generic vectors (per-dimension): c = {cc.mean():.4f} -> Q = {Q_of_c(cc.mean()):.4f}")
    print(f"   target Koide: c = sqrt(2) = {np.sqrt(2):.4f} -> Q = 2/3 = {2/3:.4f}")
    print("   => geometry is democratic (Q ~ 1/2); Koide's c=sqrt2 is the MMD weighting,")
    print("      not a consequence of the V_112 geometry.")


if __name__ == "__main__":
    main()
