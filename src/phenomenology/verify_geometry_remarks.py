#!/usr/bin/env python3
"""
verify_geometry_remarks.py
==============================================================================
Numerical verifications supporting the geometric remarks added in PMMD v6.0:

  * rem:delta9-dimensionality-v6   (Sec. on the Stage-9 projection coefficient)
  * rem:lepton-berry-triple-v6     (Facet 5 / charged-lepton triple)
  * rem:beta-from-foam-v6          (RG matching / foam-discreteness corrections)

What it checks (each block is self-contained and prints PASS/INFO):

  1. E8 -> H4 cut-and-project (Moody-Patera): the 240 E8 roots project onto
     two concentric 600-cells whose radii stand in the golden ratio phi.
     This is the *rigorous* phi^1 content of the projection.

  2. phi-power scaling of the Stage-9 deficit: the only golden quantity FORCED
     by the projection is phi^1 (radius / per-root component ratios). The
     exponent 3 in delta_9 = 1/phi^3 is a *3-dimensional measure* choice
     (k-volume ratio scales as phi^k), not a consequence of the projection.

  3. Charged-lepton triple as a Berry triple: m_k ratios reproduced from the
     Koide parametrisation sqrt(m_k)=M(1+sqrt2 cos(theta+2 pi k/3)), theta=2/9,
     and read as Berry areas Omega_k of three loops sharing the C3 topology.
     (Honest: with a common n this identification is tautological; see remark.)

  4. H4-protected Lorentz suppression: the H4 (icosahedral 4D) Coxeter group
     has fundamental-invariant degrees {2,12,20,30}; the first anisotropic
     invariant beyond |p|^2 is degree 12, so rotational anisotropy is pushed
     to relative order (p l*)^10 -- vs (p l*)^2 for a periodic lattice.

Author: Gianluca Genovese (PMMD framework).  Dependencies: numpy.
==============================================================================
"""
import itertools
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------
# Block 1 & 2: E8 -> H4 Coxeter projection
# --------------------------------------------------------------------------
def e8_roots():
    """Return the 240 roots of E8 in R^8 (norm^2 = 2)."""
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8)
                    v[i], v[j] = si, sj
                    roots.append(v)
    for s in itertools.product([0.5, -0.5], repeat=8):
        if list(s).count(-0.5) % 2 == 0:
            roots.append(np.array(s))
    return np.array(roots)


def e8_coxeter_element():
    """Coxeter element of E8 as a product of the 8 simple reflections."""
    e = np.eye(8)
    simple = np.array([
        0.5 * (e[0] - e[1] - e[2] - e[3] - e[4] - e[5] - e[6] + e[7]),
        e[0] + e[1], e[1] - e[0], e[2] - e[1],
        e[3] - e[2], e[4] - e[3], e[5] - e[4], e[6] - e[5],
    ])

    def refl_matrix(al):
        M = np.eye(8)
        for k in range(8):
            M[:, k] = e[k] - 2 * (e[k] @ al) / (al @ al) * al
        return M

    w = np.eye(8)
    for al in simple:
        w = refl_matrix(al) @ w
    return w


def h4_projector(exponents=(1, 11)):
    """Orthogonal projector onto the 4D H4 Coxeter plane.

    One eigenvector per conjugate pair (exponents 1 and 11; their conjugates
    29, 19 span the same real planes)."""
    w = e8_coxeter_element()
    val, vec = np.linalg.eig(w)
    exps = np.round(np.angle(val) / (2 * np.pi) * 30).astype(int) % 30
    basis = []
    for t in exponents:
        k = [i for i in range(8) if exps[i] == t][0]
        v = vec[:, k] / np.linalg.norm(vec[:, k])
        basis += [np.real(v), np.imag(v)]
    Q, _ = np.linalg.qr(np.array(basis).T)
    return Q[:, :4] @ Q[:, :4].T


def block1_2():
    print("[1] E8 -> H4 projection: two 600-cells in ratio phi")
    roots = e8_roots()
    P = h4_projector()
    rp = np.array([np.linalg.norm(P @ r) for r in roots])
    shells = sorted(set(np.round(rp, 3)))
    ratio = max(shells) / min(s for s in shells if s > 0.1)
    print(f"    physical shells = {shells}")
    print(f"    radius ratio    = {ratio:.4f}   (phi = {PHI:.4f})")
    ok = abs(ratio - PHI) < 1e-2
    print(f"    {'PASS' if ok else 'FAIL'}: two-600-cell golden ratio confirmed (phi^1 is rigorous)\n")

    print("[2] phi-power scaling: only phi^1 is forced; exponent 3 is a 3-volume choice")
    for k in (1, 2, 3, 4):
        print(f"    {k}-volume ratio = phi^{k} = {PHI**k:.4f}  (inverse {PHI**-k:.4f})")
    print(f"    delta_9 = 1/phi^3 = {PHI**-3:.4f}  matches the inverse 3-volume.")
    print("    INFO: dimensionality 3 is the substantive input (spatial section),")
    print("          NOT forced by the projection (which gives phi^1).\n")
    return ok


# --------------------------------------------------------------------------
# Block 3: charged-lepton triple
# --------------------------------------------------------------------------
def block3():
    print("[3] Charged-lepton triple from Koide (theta=2/9, c=sqrt2) as Berry areas")
    theta, c = 2 / 9, np.sqrt(2)
    amp = np.array([1 + c * np.cos(theta + 2 * np.pi * k / 3) for k in range(3)])
    m = np.sort(amp ** 2)  # ascending -> e, mu, tau (up to overall scale)
    ratios = m / m[0]
    obs = np.array([1.0, 105.6583755 / 0.51099895, 1776.86 / 0.51099895])
    print(f"    framework m ratios = {np.round(ratios, 2)}")
    print(f"    observed  m ratios = {np.round(obs, 2)}")
    ok = np.allclose(ratios, obs, rtol=3e-3)
    print(f"    {'PASS' if ok else 'FAIL'}: lepton mass ratios reproduced")
    Omega = m / m.max() * (2 * np.pi / 3)  # anchor heaviest at C3 value 2pi/3
    print(f"    as Berry areas (Omega_tau = 2pi/3): Omega_k = {np.round(Omega, 5)} sr")
    print("    INFO: with a common loop length n, Omega_k ~ m_k is tautological")
    print("          (corroborative Stratum-3 reading, not a derivation).\n")
    return ok


# --------------------------------------------------------------------------
# Block 4: H4-protected Lorentz suppression estimate
# --------------------------------------------------------------------------
def block4():
    print("[4] H4-protected Lorentz suppression (rotational sector)")
    M_GUT, M_cut = 1e16, 5.5e17  # GeV;  M_cut = m_P/(2 phi^5)
    x = M_GUT / M_cut
    print(f"    M_GUT/M_cutoff = {x:.4f}   (M_cut = m_P/2phi^5 ~ 5.5e17 GeV)")
    print(f"    isotropic correction      ~ (p l*)^2  = {x**2:.2e}  (harmless rescaling)")
    print(f"    rotational anisotropy (H4) ~ (p l*)^10 = {x**10:.2e}")
    print("    H4 invariant degrees = {2,12,20,30}; first anisotropic invariant = 12.")
    print("    INFO: boost-Lorentz invariance is supplied by foam Poisson universality")
    print("          (bridge claim 1), NOT by H4; the two are complementary.\n")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("PMMD v6.0 geometric-remark verifications")
    print("=" * 70 + "\n")
    results = [block1_2(), block3(), block4()]
    print("=" * 70)
    print(f"Summary: {sum(bool(r) for r in results)}/{len(results)} blocks PASS/INFO complete")
    print("=" * 70)
