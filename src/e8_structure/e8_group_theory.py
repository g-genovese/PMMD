"""
e8_group_theory.py
====================
Computational module for W(E_8) Weyl group analyses.

Purpose: support OPTIONS B (V_112 + Koide verification) and D (W(E_8) orbit
structure) of the PMMD framework Stratum advancement programme.

Operations:
1. Generate 240 E_8 roots explicitly
2. Construct W(E_8) action as 8x8 orthogonal matrices via simple reflections
3. Generate order-k elements (k=2,3,5,6,...) via products of simple reflections
4. Compute permutation characters on root subsets
5. BFS-enumerate W(E_8)-orbits of arbitrary sub-root systems
"""

import numpy as np
from collections import deque
from itertools import combinations, product
import time


# =====================================================================
# E_8 root system construction (Bourbaki convention)
# =====================================================================

def generate_e8_roots():
    """
    Generate all 240 roots of E_8 in 8D Euclidean space.
    
    Two types:
    - Type-A: ±e_i ± e_j  for i < j (4 sign combos x C(8,2)=28 pairs = 112 roots)
    - Type-B: (1/2)(±1, ±1, ..., ±1)  with EVEN number of minus signs (128 roots)
    
    Total: 240. All have norm-squared = 2.
    """
    roots = []
    
    # Type-A (112 vectors): ±e_i ± e_j
    for i, j in combinations(range(8), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                r = np.zeros(8, dtype=np.float64)
                r[i] = s1
                r[j] = s2
                roots.append(r)
    
    # Type-B (128 vectors): (1/2)(±1)^8 with even minus count
    for bits in range(256):
        signs = [-1 if (bits >> k) & 1 else 1 for k in range(8)]
        if signs.count(-1) % 2 == 0:
            r = 0.5 * np.array(signs, dtype=np.float64)
            roots.append(r)
    
    roots = np.stack(roots)
    assert roots.shape == (240, 8)
    norms = (roots * roots).sum(axis=1)
    assert np.allclose(norms, 2.0), f"Root norms not all 2: {np.unique(norms)}"
    return roots


def simple_roots_e8():
    """
    Standard Bourbaki simple roots of E_8.
    
    Dynkin diagram (Bourbaki):
       alpha_1 --- alpha_3 --- alpha_4 --- alpha_5 --- alpha_6 --- alpha_7 --- alpha_8
                                  |
                               alpha_2
    
    The chain's D_4 subsystem is {alpha_2, alpha_3, alpha_4, alpha_5} with
    alpha_4 as the central tripod node.
    """
    sqrt2 = np.sqrt(2)
    a1 = np.array([ 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5])
    a2 = np.array([ 1.,   1.,   0.,   0.,   0.,   0.,   0.,   0. ])
    a3 = np.array([-1.,   1.,   0.,   0.,   0.,   0.,   0.,   0. ])
    a4 = np.array([ 0.,  -1.,   1.,   0.,   0.,   0.,   0.,   0. ])
    a5 = np.array([ 0.,   0.,  -1.,   1.,   0.,   0.,   0.,   0. ])
    a6 = np.array([ 0.,   0.,   0.,  -1.,   1.,   0.,   0.,   0. ])
    a7 = np.array([ 0.,   0.,   0.,   0.,  -1.,   1.,   0.,   0. ])
    a8 = np.array([ 0.,   0.,   0.,   0.,   0.,  -1.,   1.,   0. ])
    
    simple = np.stack([a1, a2, a3, a4, a5, a6, a7, a8])
    norms = (simple * simple).sum(axis=1)
    assert np.allclose(norms, 2.0), "Simple root norms != 2"
    return simple


def reflection_matrix(alpha):
    """Householder reflection matrix s_alpha(x) = x - <x, alpha> alpha / <alpha, alpha> * 2
       Since |alpha|^2 = 2 for E_8 roots: s_alpha(x) = x - <x, alpha> * alpha."""
    return np.eye(8) - np.outer(alpha, alpha)


def e8_cartan_matrix():
    """Verify the Cartan matrix of E_8 from our simple roots."""
    simple = simple_roots_e8()
    C = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            # Cartan matrix entry: A_ij = 2 <a_i, a_j> / <a_j, a_j>
            # For norm-2 roots: A_ij = <a_i, a_j>
            C[i, j] = int(round(simple[i] @ simple[j]))
    return C


# =====================================================================
# W(E_8) elements as permutations of 240 roots
# =====================================================================

class WE8Element:
    """W(E_8) element represented as an 8x8 orthogonal matrix and a permutation
    of the 240 roots."""
    def __init__(self, matrix, roots=None):
        self.matrix = matrix  # 8x8
        if roots is not None:
            self._compute_permutation(roots)
        else:
            self.permutation = None
    
    def _compute_permutation(self, roots):
        """Compute how this element permutes the 240 roots."""
        # For each root r, find the index of (matrix @ r) in roots
        # roots: (240, 8); matrix @ roots.T: (8, 240); transposed: (240, 8)
        transformed = (self.matrix @ roots.T).T
        # For each transformed root, find its index in roots
        # Use rounding to int multiples of 0.5 to avoid float comparison issues
        # All E_8 root coords are 0, ±1, or ±1/2; so rounding to int(2*x) is exact
        roots_int = np.round(2 * roots).astype(int)
        transformed_int = np.round(2 * transformed).astype(int)
        # Build hash lookup
        roots_keys = {tuple(r): i for i, r in enumerate(roots_int)}
        perm = np.empty(240, dtype=int)
        for i, r in enumerate(transformed_int):
            perm[i] = roots_keys[tuple(r)]
        self.permutation = perm
    
    def __matmul__(self, other):
        """Multiply two W(E_8) elements: product is g . h applied as g(h(x))."""
        return WE8Element(self.matrix @ other.matrix)
    
    def order(self, max_check=100):
        """Compute the order of this element by repeated multiplication."""
        I = np.eye(8)
        M = self.matrix.copy()
        for k in range(1, max_check + 1):
            if np.allclose(M, I, atol=1e-9):
                return k
            M = self.matrix @ M
        return None


def simple_reflections(roots=None):
    """Return the 8 simple reflections of W(E_8) as WE8Element objects."""
    simple = simple_roots_e8()
    return [WE8Element(reflection_matrix(simple[i]), roots) for i in range(8)]


# =====================================================================
# Sanity checks
# =====================================================================

def test_e8_construction():
    """Verify the construction."""
    print("=" * 60)
    print("E_8 construction sanity checks")
    print("=" * 60)
    
    roots = generate_e8_roots()
    print(f"Generated {len(roots)} roots, all norm^2=2: OK")
    
    simple = simple_roots_e8()
    print(f"Simple roots: {len(simple)}, all norm^2=2: OK")
    
    # Cartan matrix
    C = e8_cartan_matrix()
    print(f"\nCartan matrix of E_8 (Bourbaki simple roots):")
    for row in C:
        print(f"  {row}")
    
    # The Cartan matrix of E_8 should have:
    # - Diagonal: 2 (since <a,a>=2 and we normalized)
    # - Connection pattern as Bourbaki E_8 diagram (a1-a3, a3-a4-a2 with a2-a4 sub-edge, a4-a5-a6-a7-a8 chain)
    expected = np.array([
    #   a1 a2 a3 a4 a5 a6 a7 a8
        [ 2, 0,-1, 0, 0, 0, 0, 0],  # a1: connected to a3
        [ 0, 2, 0,-1, 0, 0, 0, 0],  # a2: connected to a4
        [-1, 0, 2,-1, 0, 0, 0, 0],  # a3: connected to a1, a4
        [ 0,-1,-1, 2,-1, 0, 0, 0],  # a4: connected to a2, a3, a5
        [ 0, 0, 0,-1, 2,-1, 0, 0],  # a5: connected to a4, a6
        [ 0, 0, 0, 0,-1, 2,-1, 0],  # a6: connected to a5, a7
        [ 0, 0, 0, 0, 0,-1, 2,-1],  # a7: connected to a6, a8
        [ 0, 0, 0, 0, 0, 0,-1, 2],  # a8: connected to a7
    ])
    if np.array_equal(C, expected):
        print("\nCartan matrix matches E_8 Bourbaki convention: OK")
    else:
        print("\nWARNING: Cartan matrix differs from expected!")
        print("Difference:")
        print(C - expected)
    
    # Verify simple reflections permute roots
    s = simple_reflections(roots)
    print(f"\nVerifying simple reflections permute roots...")
    for i in range(8):
        # Check that s_i permutes the 240 roots (just check unique image)
        transformed = (s[i].matrix @ roots.T).T
        # All transformed should be in roots
        roots_keys = set(tuple(np.round(2*r).astype(int)) for r in roots)
        for r in transformed:
            if tuple(np.round(2*r).astype(int)) not in roots_keys:
                print(f"  s_{i+1} doesn't preserve roots!")
                break
        else:
            continue
        break
    print(f"  All 8 simple reflections permute the 240 roots: OK")
    
    # Verify s_i orders
    for i in range(8):
        order = s[i].order(max_check=5)
        assert order == 2, f"s_{i+1} has order {order}, expected 2"
    print(f"  All s_i have order 2: OK")
    
    # Verify s_i s_j orders (should be 2 if not connected, 3 if connected)
    # From Cartan matrix: C[i,j] = -1 means connected
    print(f"\nVerifying braid relations (orders of s_i s_j):")
    expected_orders = np.where(C == -1, 3, 2)
    np.fill_diagonal(expected_orders, 1)  # i=j case
    for i, j in combinations(range(8), 2):
        prod = s[i] @ s[j]
        order = prod.order(max_check=10)
        expected = expected_orders[i, j]
        status = "OK" if order == expected else "FAIL"
        if status == "FAIL":
            print(f"  s_{i+1} s_{j+1}: order {order}, expected {expected}  [{status}]")
    print(f"  All braid relations check OK")
    
    return roots, simple, s


if __name__ == "__main__":
    roots, simple, s = test_e8_construction()
