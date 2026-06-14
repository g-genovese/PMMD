#!/usr/bin/env python3
"""
analyze_foam_collective_em_tetrahedron.py -- Redo the foam-collective
coarse-graining with the proper 3D structural unit: the tetrahedron, not the
1D chain of triangles. 4 vertices per cell, 6 edges, S_4 / T_d symmetry.

Mode decomposition under S_4:
   4 vertices = 1D trivial rep (uniform) + 3D standard rep (vector)
This matches A_mu's 4 components (1 scalar + 3 spatial vector) -- the photon's
natural structure. Gauge fixing removes 2 (the uniform/scalar and the longi-
tudinal-in-propagation), leaving the 2 transverse polarizations.

Test: does the naive intra-tetrahedron stiffness give massless polarizations?
Answer: NO, same obstacle as the triangle case -- but now the structural map
to A_mu is exact.
"""
import sympy as sp
import numpy as np

print("=== Tetrahedron as the 3D structural unit ===")
print()
print("A regular tetrahedron has 4 vertices, 6 edges (all equivalent), and full")
print("symmetry group T_d (isomorphic to S_4, order 24). The natural mode")
print("decomposition of 4 vertex phases under S_4 is:")
print("    4 = 1 (trivial / uniform)  +  3 (standard / vector)")
print()
print("The 3D vector irrep transforms exactly like a Euclidean 3-vector.")
print("Map to A_mu (the photon's 4-vector):")
print("    psi^0  (uniform 1D)         <->  A_0  (scalar potential, gauge in Coulomb)")
print("    psi^a  (vector 3D, a=x,y,z) <->  A_i  (spatial 3-vector)")
print("Gauge fixing (Coulomb: nabla.A = 0) removes the longitudinal mode along")
print("propagation; what remains is the two transverse polarizations. Mode count:")
print("    4 - 2 (gauge) = 2 physical = the photon's 2 transverse polarizations.")
print("This matches the standard A_mu mode count EXACTLY; the 1D chain of triangles")
print("(1 gauge + 2 charged) was one mode short. The tetrahedron is the right unit.")

# -----------------------------------------------------------------------------
# Within-tetrahedron stiffness in mode basis
# -----------------------------------------------------------------------------
print()
print("=== Within-tetrahedron stiffness in mode basis ===")
print()
print("Phases (phi_1, phi_2, phi_3, phi_4) on 4 vertices. 6 edges, all equivalent")
print("for a regular tetrahedron. Stiffness sum:")
print("    W = sum_{i<j} (phi_i - phi_j)^2")
print()
print("Identity: sum_{i<j}(phi_i - phi_j)^2 = 4 sum_i phi_i^2 - (sum_i phi_i)^2.")
ph1, ph2, ph3, ph4 = sp.symbols('phi_1 phi_2 phi_3 phi_4', real=True)
W = sum((p_i - p_j)**2 for i, p_i in enumerate([ph1, ph2, ph3, ph4]) for p_j in [ph1, ph2, ph3, ph4][i+1:])
W_alt = 4*sum(p**2 for p in [ph1, ph2, ph3, ph4]) - sum([ph1, ph2, ph3, ph4])**2
diff = sp.simplify(sp.expand(W - W_alt))
print(f"Verification: sum (phi_i - phi_j)^2 - [4 sum phi^2 - (sum phi)^2] = {diff}")
print()
print("With orthonormal Z_3->S_4 modes (psi^0 = (1/2) sum_i phi_i, and 3 orthonormal")
print("psi^a a=1,2,3 in the 3D vector irrep), one has")
print("    sum_i phi_i = 2 psi^0,   sum_i phi_i^2 = |psi^0|^2 + |psi^vec|^2,")
print("so")
print("    W = 4 (|psi^0|^2 + |psi^vec|^2) - (2 psi^0)^2 = 4 |psi^vec|^2.")
print()
print("Reading: the within-tetrahedron stiffness gives a MASS term 4 J_perp |psi^vec|^2")
print("for the 3D vector modes, while the uniform psi^0 is exempt. Same structural")
print("pattern as the triangle case, now with the *correct* mode count.")

# -----------------------------------------------------------------------------
# Dispersion analysis
# -----------------------------------------------------------------------------
print()
print("=== Dispersion of the 4 modes in foam coarse-grained dynamics ===")
print()
Jpe, Jpa, kw = sp.symbols('J_perp J_parallel k', positive=True, real=True)
disp_psi0 = 2 * Jpa * (1 - sp.cos(kw))
disp_psi_vec = 2 * Jpa * (1 - sp.cos(kw)) + 4 * Jpe
print(f"  psi^0   (uniform, A_0 -- gauge mode in Coulomb):")
print(f"          omega^2(k) = {disp_psi0}")
print(f"          at k=0: omega = 0  (massless, as required for the gauge mode)")
print()
print(f"  psi^x, psi^y, psi^z  (3D vector modes, A_i):")
print(f"          omega^2(k) = {disp_psi_vec}")
print(f"          at k=0: omega^2 = 4 J_perp  -- MASSIVE, gap 2 sqrt(J_perp)")
print()
print("Gauge fixing (ex: Coulomb, nabla.A = 0 in foam coarse-graining):")
print("  - Removes the longitudinal mode (the psi^a aligned with propagation hat k).")
print("  - Leaves 2 transverse polarisations (the psi^a perpendicular to hat k).")
print()
print("BUT both surviving transverse modes still have mass 2 sqrt(J_perp) from the")
print("intra-tetrahedron stiffness. The same obstruction as the triangle case persists.")

# -----------------------------------------------------------------------------
# Honest assessment
# -----------------------------------------------------------------------------
print()
print("=== Honest assessment ===")
print()
print("Improved with tetrahedron vs triangle:")
print("  + Mode count now matches A_mu exactly (4 = 1 scalar + 3 vector).")
print("  + Gauge fixing structure is the standard QED one (Coulomb gauge).")
print("  + The 2 transverse polarisations emerge naturally as the residual physical")
print("    modes after gauge fixing.")
print()
print("Not yet resolved:")
print("  - Intra-cell kinetic stiffness still gives substrate-scale mass to all 3")
print("    vector modes (and hence to the 2 transverse polarisations after gauge")
print("    fixing). The photon's masslessness requires this mass term to be ABSENT.")
print()
print("In standard QED gauge invariance EXPLICITLY FORBIDS a mass term m^2 A_mu A^mu")
print("for A_mu. So for the foam coarse-graining to give a massless photon, the")
print("effective action must be exactly gauge-invariant -- i.e., terms like the")
print("|psi^vec|^2 mass term must be absent or cancelled.")
print()
print("The remaining structural target is therefore: derive the foam-collective EM")
print("action from S[phi] *with manifest U(1) gauge invariance preserved*, so that")
print("the substrate-scale mass term we found here is automatically excluded.")
print("Candidate mechanisms:")
print(" (A) The coarse-graining of S[phi] yields the kinetic stiffness in the form of")
print("     a gauge-invariant field strength F_{mu nu} = d_mu A_nu - d_nu A_mu, not as")
print("     a direct mass term A_mu A^mu. This converts intra-cell stiffness into")
print("     gradient-of-gauge-field terms, automatically massless.")
print(" (B) The intra-cell sum is constrained (e.g., by closure of the cell as a")
print("     simplex, by the foam's specific E_8/D_4 cell structure) in a way that")
print("     enforces gauge invariance.")
print()
print("Both (A) and (B) require analysing the foam's actual structure carefully,")
print("not a generic lattice. The tetrahedron is the right MODE-COUNT unit; the")
print("actual foam coarse-graining (preserving U(1) gauge invariance) remains open.")
