"""
option_B_V112_koide.py
=======================
OPTION B: Verify the algebraic foundations of the Koide identity in PMMD v5.1:
  - V_112 exists as a 112-dim irrep of W(E_8)  (Theorem thm:V112-uniqueness)
  - theta = 2/9 = C_2(3) / (2 h^v)             (Equation eq:theta-algebraic)
  - V_112 has a Z_3-cyclic structure with character matching three generations

Strategy: compute permutation characters of W(E_8) on root subsets at specific
order-3 elements, then check what irrep content matches the claimed structure.
"""

import numpy as np
import time
from itertools import combinations
from collections import defaultdict
from e8_group_theory import (
    generate_e8_roots, simple_roots_e8, reflection_matrix,
    simple_reflections, WE8Element
)


# =====================================================================
# Step 1: Algebraic identity for theta = 2/9
# =====================================================================

def verify_theta_algebraic_identity():
    """
    The paper claims: theta = 2/9 = C_2(3) / (2 h^v)
    where the SU(N) interpretation gives:
      C_2(fundamental of SU(N)) = (N^2 - 1) / (2N)
      h^v(SU(N)) = N
    For N=3: C_2(3) = 4/3, h^v(SU(3)) = 3 -> 2 h^v = 6
    Hence theta = (4/3) / 6 = 2/9.
    
    This is a trivial algebraic identity, but worth confirming explicitly.
    """
    print("=" * 72)
    print("STEP 1: Algebraic identity theta = C_2(3) / (2 h^v) = 2/9")
    print("=" * 72)
    
    from fractions import Fraction
    for N in [2, 3, 4, 5]:
        C2 = Fraction(N*N - 1, 2*N)
        h_v = N  # dual Coxeter number of SU(N)
        theta_N = C2 / (2 * h_v)
        print(f"  SU({N}): C_2(N) = {C2}, h^v = {h_v}, theta = {theta_N} = {float(theta_N):.6f}")
    
    print(f"\n  For N=3: theta = 2/9 -> Koide angle predicted at 2/9 = "
          f"{float(Fraction(2,9)):.6f}")
    
    # Empirical Koide angle from paper line 3320:
    # |theta_obs - 2/9| / sigma_theta = 0.91 sigma; theta_obs ~ 0.222 +/- some
    # The framework's prediction is statistically supported.
    print(f"\n  Reference: paper line 3320 cites |theta_obs - 2/9| / sigma = 0.91 sigma")
    print(f"             (non-exclusion at current precision; ~10x tighter sigma_m_tau")
    print(f"              would discriminate from competing structural forms)")


# =====================================================================
# Step 2: Order-3 elements of W(E_8) and their action on root subsets
# =====================================================================

def find_order_3_element(simple_matrices):
    """An order-3 element of W(E_8) is the product of two simple reflections
    s_i s_j where alpha_i and alpha_j are connected in the Dynkin diagram
    (Cartan matrix entry A_{ij} = -1).
    
    For Bourbaki E_8: alpha_1 is connected to alpha_3 (only).
    So s_1 * s_3 has order 3.
    """
    # s_1 s_3 has order 3
    g = simple_matrices[0] @ simple_matrices[2]
    return g


def verify_z3_element(g_matrix):
    """Verify the element has order 3."""
    I = np.eye(8)
    M = g_matrix.copy()
    for k in range(1, 7):
        if np.allclose(M, I, atol=1e-9):
            return k
        M = g_matrix @ M
    return None


def permutation_character_on_root_subset(g_matrix, root_subset):
    """Count fixed points of g acting on a given subset of roots.
    
    For a permutation representation, character = #(fixed points).
    """
    transformed = (g_matrix @ root_subset.T).T
    # Compare each row of transformed to corresponding row of root_subset
    # Element-wise close + all-axis True
    fixed_count = 0
    root_keys = {tuple(np.round(2*r).astype(int)): i for i, r in enumerate(root_subset)}
    for i, r in enumerate(transformed):
        key = tuple(np.round(2*r).astype(int))
        if key in root_keys and root_keys[key] == i:
            fixed_count += 1
    return fixed_count


def analyze_z3_on_root_subsets():
    """Compute the character of the perm reps of W(E_8) on various root subsets
    for an order-3 element."""
    print("\n" + "=" * 72)
    print("STEP 2: Order-3 element action on E_8 root subsets")
    print("=" * 72)
    
    roots = generate_e8_roots()
    simple_e8 = simple_roots_e8()
    simple_matrices = [reflection_matrix(simple_e8[i]) for i in range(8)]
    
    # Multiple choices of order-3 elements
    # Different conjugacy classes of order 3 in W(E_8) give different characters
    candidates = {
        "s_1 s_3 (adjacent simple, connected via Dynkin)": simple_matrices[0] @ simple_matrices[2],
        "s_3 s_4": simple_matrices[2] @ simple_matrices[3],
        "s_2 s_4 (triality leg)": simple_matrices[1] @ simple_matrices[3],
        "s_4 s_5": simple_matrices[3] @ simple_matrices[4],
        "s_1 s_3 s_4 s_2 s_4 s_3 (commutator-like)": (
            simple_matrices[0] @ simple_matrices[2] @ simple_matrices[3] @
            simple_matrices[1] @ simple_matrices[3] @ simple_matrices[2]
        ),
    }
    
    # Separate the 240 roots into type-A (112 vectors ±e_i±e_j) and type-B (128 half-integer)
    type_A_mask = []
    type_B_mask = []
    for i, r in enumerate(roots):
        if all(abs(x) in [0, 1] for x in r):
            type_A_mask.append(i)
        else:
            type_B_mask.append(i)
    type_A_mask = np.array(type_A_mask)
    type_B_mask = np.array(type_B_mask)
    assert len(type_A_mask) == 112, f"Type-A count: {len(type_A_mask)}, expected 112"
    assert len(type_B_mask) == 128, f"Type-B count: {len(type_B_mask)}, expected 128"
    
    type_A_roots = roots[type_A_mask]
    type_B_roots = roots[type_B_mask]
    
    for name, g in candidates.items():
        ord_g = verify_z3_element(g)
        char_240 = permutation_character_on_root_subset(g, roots)
        char_112_typeA = permutation_character_on_root_subset(g, type_A_roots)
        char_128_typeB = permutation_character_on_root_subset(g, type_B_roots)
        print(f"\n  Element: {name}")
        print(f"    Order: {ord_g}")
        print(f"    chi_perm(g) on 240 roots:      {char_240}")
        print(f"    chi_perm(g) on 112 type-A:     {char_112_typeA}")
        print(f"    chi_perm(g) on 128 type-B:     {char_128_typeB}")
        
        # If we have a Z_3 decomposition of 112 = d_0 + d_omega + d_omegabar
        # and g acts with eigenvalue ω on R_omega, etc., then the character on the
        # NATURAL action (NOT permutation) on V_112 would be d_0 + d_omega * omega + d_omegabar * omega_bar
        # For a real representation, d_omega = d_omegabar so:
        # chi(g) = d_0 + 2 * Re(d_omega * omega) = d_0 - d_omega (since 2*cos(2pi/3) = -1)
        # For (58, 27, 27): chi = 58 - 27 = 31


# =====================================================================
# Step 3: Decompose the permutation representation
# =====================================================================

def permutation_rep_decomposition_helper():
    """For the perm rep of W(E_8) on type-A roots (112 vectors), compute
    constraints on the irreducible decomposition.
    
    Generally: perm rep χ_perm has character χ_perm(g) = #fixed points.
    Decomposition into irreps: χ_perm = sum_k m_k χ_k where m_k are multiplicities.
    
    Standard fact: the perm rep contains the trivial rep with multiplicity = 
        number of orbits of the action.
    Since W(E_8) is transitive on type-A roots: multiplicity of trivial = 1.
    
    Also: m_k = <χ_perm, χ_k> = (1/|G|) sum_g χ_perm(g) chi_k(g_inv)
    This requires the character table, which we don't have here.
    """
    print("\n" + "=" * 72)
    print("STEP 3: Permutation rep structure on type-A roots")
    print("=" * 72)
    
    roots = generate_e8_roots()
    simple_e8 = simple_roots_e8()
    simple_matrices = [reflection_matrix(simple_e8[i]) for i in range(8)]
    
    type_A_mask = np.array([i for i, r in enumerate(roots)
                             if all(abs(x) in [0, 1] for x in r)])
    type_A_roots = roots[type_A_mask]
    
    print(f"\n  perm rep on 112 type-A vectors (single W(E_8) orbit):")
    print(f"    decomposes as: trivial (dim 1) + non-trivial pieces (dim 111)")
    print(f"    -> If V_112 = trivial + nontrivial_112 contains V_111, then")
    print(f"       perm_112 = trivial + V_111")
    print(f"    Alternatively: V_112 is NOT the perm rep on type-A, but")
    print(f"    a distinct irrep that we need to identify via character table.")
    
    # Identity character on 112 vectors: 112 (everything fixed)
    print(f"\n  chi_perm(identity) on type-A = 112 (all fixed)")
    
    # Now order-3 element s_1 s_3
    g = simple_matrices[0] @ simple_matrices[2]
    chi_g = permutation_character_on_root_subset(g, type_A_roots)
    print(f"  chi_perm(s_1 s_3) on type-A = {chi_g}")
    print(f"    -> the trivial subrep contributes 1; non-trivial part contributes {chi_g - 1}")


# =====================================================================
# Step 4: Cycle structure analysis
# =====================================================================

def cycle_structure_on_roots(g_matrix, roots):
    """Compute the cycle structure of g acting on the given roots."""
    n = len(roots)
    roots_keys = {tuple(np.round(2*r).astype(int)): i for i, r in enumerate(roots)}
    perm = np.empty(n, dtype=int)
    transformed = (g_matrix @ roots.T).T
    for i, r in enumerate(transformed):
        perm[i] = roots_keys[tuple(np.round(2*r).astype(int))]
    
    # Find cycles
    visited = np.zeros(n, dtype=bool)
    cycles = []
    for i in range(n):
        if visited[i]:
            continue
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        cycles.append(cycle_len)
    
    # Count cycles by length
    cycle_count = defaultdict(int)
    for c in cycles:
        cycle_count[c] += 1
    return dict(cycle_count)


def step4_cycle_analysis():
    print("\n" + "=" * 72)
    print("STEP 4: Cycle-structure analysis of order-3 elements")
    print("=" * 72)
    
    roots = generate_e8_roots()
    simple_e8 = simple_roots_e8()
    simple_matrices = [reflection_matrix(simple_e8[i]) for i in range(8)]
    
    type_A_mask = np.array([i for i, r in enumerate(roots)
                             if all(abs(x) in [0, 1] for x in r)])
    type_A_roots = roots[type_A_mask]
    
    type_B_mask = np.array([i for i, r in enumerate(roots)
                             if not all(abs(x) in [0, 1] for x in r)])
    type_B_roots = roots[type_B_mask]
    
    # All single Coxeter generators s_i s_{i+1} (different conjugacy classes potentially)
    print(f"\nFor each candidate order-3 element, cycle structure on 112 type-A roots:")
    print(f"  Format: (cycle_length: number_of_cycles)")
    print(f"  For order-3 elements, cycle lengths can only be 1 or 3:")
    print(f"    #fixed_points = #1-cycles")
    print(f"    #(roots in 3-cycles) = 112 - #fixed = 3 * (#3-cycles)")
    print()
    
    for ij in [(0,2), (1,3), (2,3), (3,4), (4,5), (5,6), (6,7)]:
        i, j = ij
        g = simple_matrices[i] @ simple_matrices[j]
        ord_g = verify_z3_element(g)
        cycles_A = cycle_structure_on_roots(g, type_A_roots)
        cycles_B = cycle_structure_on_roots(g, type_B_roots)
        is_z3 = ord_g == 3
        if is_z3:
            print(f"  s_{i+1} s_{j+1} (order {ord_g}):")
            print(f"    on type-A (112): {cycles_A}")
            print(f"    on type-B (128): {cycles_B}")


if __name__ == "__main__":
    verify_theta_algebraic_identity()
    analyze_z3_on_root_subsets()
    permutation_rep_decomposition_helper()
    step4_cycle_analysis()
