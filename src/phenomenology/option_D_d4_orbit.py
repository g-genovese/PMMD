"""
option_D_d4_orbit.py
=====================
OPTION D: Verify W(E_8) orbit of the chain's D_4 sub-root system.

Paper claim (Remark rem:triality-candidate, line 4536):
    |W(E_8)|/(|W(F_4)| * |W(D_4)|) = 696,729,600 / (1152 * 192) = 3,150
    Direct BFS enumeration: 3,150 conjugate D_4's, runtime ~10.8s

This script:
1. Extracts the chain's D_4 = {alpha_2, alpha_3, alpha_4, alpha_5}
2. Generates the full 24 roots of this D_4 (closure under reflection)
3. BFS-enumerates the W(E_8) orbit of this 24-root set
4. Verifies 3,150 orbit size
5. Computes the stabilizer order (W(F_4) x W(D_4) = 221,184)
6. Examines sub-orbit structure under specific subgroups
"""

import numpy as np
from collections import deque
import time
from e8_group_theory import (
    generate_e8_roots, simple_roots_e8, reflection_matrix,
    simple_reflections, WE8Element
)


def generate_d4_subroot_system(simple_d4):
    """
    Given 4 simple roots forming a D_4, generate the full 24-root D_4 root system
    by closure under reflections.
    """
    seen = set()
    # Store as int tuples (after scaling by 2 for D_4 these are all integers)
    queue = deque()
    for r in simple_d4:
        key = tuple(np.round(2 * r).astype(int))
        if key not in seen:
            seen.add(key)
            queue.append(r)
    
    while queue:
        r = queue.popleft()
        # Apply reflections by each previously seen root to r
        for k in list(seen):
            r_neighbor = np.array(k) / 2.0
            # Reflection: r_new = r - <r, r_n> * r_n  (for norm-2 roots)
            r_new = r - (r @ r_neighbor) * r_neighbor
            new_key = tuple(np.round(2 * r_new).astype(int))
            if new_key not in seen and np.allclose((r_new * r_new).sum(), 2.0):
                seen.add(new_key)
                queue.append(r_new)
    
    # Verify D_4 has 24 roots
    roots_d4 = np.array([np.array(k) / 2.0 for k in seen])
    return roots_d4, seen


def root_set_to_indices(d4_roots, all_roots_keys):
    """Convert a D_4 root system to a sorted tuple of indices in the 240-root array."""
    indices = []
    for r in d4_roots:
        key = tuple(np.round(2 * r).astype(int))
        if key in all_roots_keys:
            indices.append(all_roots_keys[key])
    return tuple(sorted(indices))


def apply_reflection_to_root_set(simple_idx, root_set_indices, roots, simple_matrices):
    """Apply simple reflection s_simple_idx to each root in root_set_indices.
    Returns the new sorted tuple of indices."""
    S = simple_matrices[simple_idx]
    new_indices = []
    for idx in root_set_indices:
        r = roots[idx]
        r_new = S @ r
        key = tuple(np.round(2 * r_new).astype(int))
        new_indices.append(roots_keys[key])
    return tuple(sorted(new_indices))


def bfs_orbit(initial_set, n_simple, roots, simple_matrices, roots_keys,
              verbose=True):
    """BFS-enumerate orbit of a root subset under W(E_8) acting via simple reflections."""
    seen = {initial_set}
    queue = deque([initial_set])
    n_steps = 0
    t0 = time.time()
    
    while queue:
        current = queue.popleft()
        for k in range(n_simple):
            S = simple_matrices[k]
            new_indices = []
            for idx in current:
                r = roots[idx]
                r_new = S @ r
                key = tuple(np.round(2 * r_new).astype(int))
                new_indices.append(roots_keys[key])
            new_set = tuple(sorted(new_indices))
            if new_set not in seen:
                seen.add(new_set)
                queue.append(new_set)
        n_steps += 1
        if verbose and len(seen) % 500 == 0 and len(seen) > 0:
            elapsed = time.time() - t0
            print(f"  ... {len(seen):>4} orbit elements found, {n_steps} BFS steps, "
                  f"{elapsed:.1f}s elapsed")
    
    return seen, time.time() - t0


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("OPTION D: W(E_8) orbit of chain's D_4 sub-root system")
    print("=" * 72)
    
    roots = generate_e8_roots()
    simple_e8 = simple_roots_e8()
    
    # Build lookup
    roots_keys = {tuple(np.round(2 * r).astype(int)): i for i, r in enumerate(roots)}
    
    # Chain's D_4 = {alpha_2, alpha_3, alpha_4, alpha_5}
    simple_d4 = simple_e8[[1, 2, 3, 4]]
    print(f"\nChain's D_4 simple roots (Bourbaki numbering 2,3,4,5 of E_8):")
    for i, r in enumerate(simple_d4):
        idx = roots_keys[tuple(np.round(2*r).astype(int))]
        print(f"  alpha_{i+2}: {r}  (root index in 240: {idx})")
    
    # Generate full D_4 root system (24 roots)
    print(f"\nGenerating full D_4 root system by closure...")
    d4_roots, d4_keys = generate_d4_subroot_system(simple_d4)
    print(f"  Generated {len(d4_roots)} roots (expected 24): "
          f"{'OK' if len(d4_roots) == 24 else 'FAIL'}")
    
    # Verify D_4 properties
    norms_sq = (d4_roots * d4_roots).sum(axis=1)
    assert np.allclose(norms_sq, 2.0), "D_4 root norms != 2"
    print(f"  All D_4 roots have norm^2=2: OK")
    
    # Convert to indices in 240-root array
    initial_d4_indices = root_set_to_indices(d4_roots, roots_keys)
    print(f"  D_4 as 24-tuple of indices in 240-root array: "
          f"first 5 = {initial_d4_indices[:5]}, last 5 = {initial_d4_indices[-5:]}")
    
    # Build simple reflection matrices
    simple_matrices = [reflection_matrix(simple_e8[i]) for i in range(8)]
    
    # BFS the orbit
    print(f"\nBFS-enumerating W(E_8) orbit of D_4...")
    print(f"  Expected: 3,150 conjugate D_4's (per paper line 4536)")
    print(f"  |W(E_8)| / |W(F_4) x W(D_4)| = 696,729,600 / 221,184 = 3,150")
    
    orbit, dt = bfs_orbit(initial_d4_indices, 8, roots, simple_matrices, roots_keys)
    
    print(f"\n=== RESULT ===")
    print(f"  Orbit size: {len(orbit)}")
    print(f"  Expected:   3,150")
    print(f"  Match:      {'YES' if len(orbit) == 3150 else 'NO'}")
    print(f"  Runtime:    {dt:.1f}s (paper reports ~10.8s)")
    
    if len(orbit) == 3150:
        print(f"\n  STRATUM CONFIRMATION: 3,150 conjugate D_4 sub-root systems")
        print(f"  Stabilizer order: |W(E_8)|/{len(orbit)} = "
              f"{696729600 // len(orbit):,}")
        print(f"  Expected |W(F_4) x W(D_4)| = 1152 * 192 = {1152*192:,}")
        if 696729600 // len(orbit) == 1152 * 192:
            print(f"  Stabilizer matches W(F_4) x W(D_4): OK")
