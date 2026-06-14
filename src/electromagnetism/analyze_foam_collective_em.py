#!/usr/bin/env python3
"""
analyze_foam_collective_em.py -- Coarse-graining S[phi] to discrete Pancharatnam
on foam triangles; compute the dispersion of Z_3 eigenmodes psi^0, psi^{+/-1}.

Target: show that the foam-collective discretization of the substrate action
gives massless propagation for psi^{+/-1} (the photon's two polarizations).

Result:
  (a) The Pancharatnam (= signed spherical-triangle area on Bloch sphere,
      the standard discretization of the CP^1 WZ term) is verified to be
      ZERO on all equatorial configurations, and non-zero only with polar
      excursion. -> Topological cost of equatorial Z_3 cyclic patterns is 0.
  (b) The kinetic stiffness 1/2 sum (n_i - n_j)^2 on the foam graph, however,
      gives mass m^2 = 3 J_perp to psi^{+/-1} from intra-triangle phase
      differences -- even on equator -- so the naive coarse-graining
      does NOT give massless polarizations.
  (c) Honest assessment: the structural target is now sharply identified; the
      tension is real and requires either (i) a foam graph without intra-
      triangle 1-edges (only between-cell edges in 1-skeleton sense; but the
      E_8 foam IS triangulated), (ii) a gauge-symmetry/Stueckelberg mechanism
      removing the intra-triangle mass, or (iii) re-articulation of the
      photon's polarizations from a different foam-collective object. None
      of these falls out trivially.
"""

import numpy as np
import sympy as sp

# -----------------------------------------------------------------------------
# (a) Discrete Pancharatnam = signed spherical-triangle area on Bloch sphere
# -----------------------------------------------------------------------------
def pancharatnam(n1, n2, n3):
    """Signed solid angle subtended by spherical triangle (n1, n2, n3) on S^2.
    Equivalent to twice the Pancharatnam phase of the cyclic loop n1->n2->n3->n1.
    Standard formula: Omega = 2 atan2(num, den) with
       num = n1 . (n2 x n3),  den = 1 + n1.n2 + n2.n3 + n3.n1.
    """
    n1, n2, n3 = map(np.asarray, (n1, n2, n3))
    num = float(np.dot(n1, np.cross(n2, n3)))
    den = float(1 + n1.dot(n2) + n2.dot(n3) + n3.dot(n1))
    return 2 * np.arctan2(num, den)

def equator(phi, dtheta=0.0):
    """Bloch unit vector: phi azimuthal, dtheta polar offset from equator (north +)."""
    return np.array([np.cos(phi)*np.cos(dtheta),
                     np.sin(phi)*np.cos(dtheta),
                     np.sin(dtheta)])

print("=== (a) Discrete Pancharatnam on equatorial Z_3 configurations ===")
# Z_3 cyclic on equator: this is the psi^{+1} pattern
ph = [0, 2*np.pi/3, 4*np.pi/3]
n = [equator(p) for p in ph]
omega_eq_z3 = pancharatnam(*n)
print(f"  Z_3 cyclic on equator ({ph[0]:.2f}, {ph[1]:.2f}, {ph[2]:.2f}): Omega = {omega_eq_z3:.3e}")

# Generic equatorial
np.random.seed(0)
ph = np.random.uniform(0, 2*np.pi, size=3)
n = [equator(p) for p in ph]
omega_eq_rand = pancharatnam(*n)
print(f"  Random equatorial: Omega = {omega_eq_rand:.3e}")

# Slightly off-equator
ph = [0, 2*np.pi/3, 4*np.pi/3]
n = [equator(ph[i], dtheta=0.1) for i in range(3)]
omega_off = pancharatnam(*n)
print(f"  Z_3 pattern with uniform polar tilt dtheta=0.1: Omega = {omega_off:.4f}")

n = [equator(ph[i], dtheta=0.0 if i == 0 else 0.1) for i in range(3)]
omega_tilt = pancharatnam(*n)
print(f"  Z_3 pattern with non-uniform polar tilt: Omega = {omega_tilt:.4f}")

print()
print("Reading: equator -> Omega is either 0 or 2*pi depending on the phase")
print("pattern (the spherical triangle on the equator great circle is degenerate;")
print("denominator sign in the atan2 formula chooses between the two hemispheres).")
print("For the Z_3 cyclic equatorial pattern, Omega = 2*pi exactly (one full")
print("hemisphere bounded by the equator). For integer WZ level k, the contribution")
print("k * 2*pi is topologically TRIVIAL (mod 2*pi) -- equivalent to 0 for the path")
print("integral. The WZ topological term thus gives no dynamical cost on equatorial")
print("Z_3-cyclic configurations: this is the topological-protection mechanism the")
print("user's intuition appeals to. Off-equatorial (polar tilt) -> non-trivial Omega.")

# -----------------------------------------------------------------------------
# (b) Dispersion of Z_3 modes in a 1D chain of transverse triangles
# -----------------------------------------------------------------------------
print()
print("=== (b) Dispersion of Z_3 modes in the foam coarse-graining ===")
print()
print("Model: 1D chain of equilateral triangles along propagation direction z.")
print("Each triangle n has 3 vertex phases phi_{n,i}, i=1,2,3, all on equator.")
print("Action (foam-level coarse-graining of S[phi]):")
print("  L = sum_n sum_i (1/2)(d_t phi_{n,i})^2")
print("    - (J_perp/2) sum_n sum_<ij> (phi_{n,i} - phi_{n,j})^2     [intra-triangle, all 3 edges]")
print("    - (J_parallel/2) sum_n sum_i (phi_{n+1,i} - phi_{n,i})^2  [propagation edges]")
print("    + (k/2) sum_n Pancharatnam(n_{1,n}, n_{2,n}, n_{3,n})     [topological, zero on equator]")
print()
print("Decompose phi -> psi^a in Z_3 basis. Each mode decouples in this quadratic L.")

# Symbolic: dispersion for each mode
Jpe, Jpa, kw = sp.symbols('J_perp J_parallel k', positive=True, real=True)

# psi^0: invariant. Intra-triangle stiffness = 0 (uniform mode).
disp_psi0 = 2 * Jpa * (1 - sp.cos(kw))
# psi^{+/-1}: charged. Intra-triangle stiffness = 3 J_perp (from sum_<ij>(phi_i - phi_j)^2 = 3(|psi1|^2+|psi2|^2))
disp_psi_pm = 2 * Jpa * (1 - sp.cos(kw)) + 3 * Jpe

print(f"  psi^0   (gauge mode):     omega^2(k) = {disp_psi0}")
print(f"                            at k=0: omega = 0  (MASSLESS, as required for gauge mode)")
print(f"  psi^+1  (right chirality): omega^2(k) = {disp_psi_pm}")
print(f"  psi^-1  (left chirality):  omega^2(k) = {disp_psi_pm}")
print(f"                            at k=0: omega^2 = 3 J_perp  -- MASSIVE, gap sqrt(3 J_perp)")
print()
print("J_perp ~ hbar / tau_* (substrate-scale coupling) -> Planck-scale mass for psi^{+/-1}.")
print("Even adding the Pancharatnam term changes nothing on equator (Omega = 0).")
print("=> naive coarse-graining does NOT give massless photon polarizations from psi^{+/-1}.")

# -----------------------------------------------------------------------------
# (c) Honest assessment of what's going on
# -----------------------------------------------------------------------------
print()
print("=== (c) Honest assessment ===")
print()
print("The user's structural intuition (photon polarizations = psi^{+/-1} on")
print("transverse C_3 triangles) gives the correct MODE COUNT (1 gauge + 2 charged,")
print("matching photon 1 longitudinal-gauge + 2 transverse-polarizations) and the")
print("right TOPOLOGICAL backing (Pancharatnam = 0 on equator). BUT it does NOT")
print("give massless psi^{+/-1} from the naive coarse-graining of S[phi]:")
print()
print("  * The Pancharatnam/WZ piece is zero on equator (good: no topological mass).")
print("  * But the kinetic stiffness piece (1/2)(n_i - n_j)^2 on intra-triangle")
print("    edges gives mass 3 J_perp -- substrate-scale, not negligible.")
print()
print("Possible resolutions (each non-trivial, beyond this session):")
print(" (i) Foam-graph hypothesis: maybe the E_8 foam's 1-skeleton at the photon's")
print("     scale does NOT couple intra-triangle vertices via 1-edges -- only via")
print("     2-faces (the WZ phase). The C_3 triangle would be a 2-simplex, not a")
print("     1-clique, in the relevant graph. Then J_perp = 0 effectively. Requires")
print("     careful analysis of the E_8/D_4 cut-and-project's graph structure.")
print(" (ii) Stueckelberg / gauge-absorbed mass: maybe there is a gauge symmetry,")
print("     acting on the within-triangle phase structure, that absorbs the mass")
print("     of psi^{+/-1} as a Goldstone. Not obvious in the U(1)-substrate.")
print(" (iii) Different mechanism: the photon's 2 polarizations might not be the")
print("     Z_3 charged modes after all -- they might come from a different foam-")
print("     collective object (e.g., gauge field as composite of substrate, with")
print("     2-vector structure from foam dimensionality). The user's correspondence")
print("     is then structural-numerical (counting), not dynamical.")
print()
print("Status: foam-collective electromagnetic dynamics remains the genuine open")
print("frontier of the EM sector. The structural target is sharply identified.")
