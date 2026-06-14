#!/usr/bin/env python3
"""
analyze_pc_cycle_sum.py -- next-loop correction to 1/p_c via 4-cycles in the
E8 foam graph (240-coordinated nearest-neighbour lattice).

Context: framework has 1/p_c = (z-1) - T_e = 183 (z=240, T_e=56) covering 84% of
the gap to the measured 1/p_c = 172.55. Residual gap ~10.45 in 1/p_c is supposed
to come from higher cycles. We compute the 4-cycle count per foam-edge.

Definitions:
  foam graph: vertices = E8 lattice points, edge iff distance sqrt(2). The 240
              nearest neighbours of vertex 0 are the 240 E8 roots.
  T_e (triangles per edge (0, alpha)): # common foam-neighbours of 0 and alpha
       = #{beta a root : beta . alpha = 1} = 56.
  C_4 (4-cycles per edge (0, alpha)): # paths 0-beta-gamma-alpha (length 3, with
       beta, gamma distinct, neither equal to 0 or alpha) closing back. We
       enumerate, then report candidate corrections.
"""
import numpy as np, itertools
from collections import Counter

def e8_roots():
    R = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8); v[i], v[j] = si, sj
                R.append(v)
    for s in itertools.product((0.5, -0.5), repeat=8):
        if sum(1 for x in s if x < 0) % 2 == 0:
            R.append(np.array(s))
    return np.array(R)

def key(v, scale=2): return tuple(int(round(scale*x)) for x in v)

R = e8_roots(); N = len(R)
keymap = {key(r): i for i, r in enumerate(R)}
assert N == 240
print(f"E8 roots: {N}; z = 240 (foam coordination)")

# pick a reference edge (0, alpha0) where alpha0 = first root
alpha0 = R[0]
# T_e sanity: roots with alpha0 . beta = 1
T_e = sum(1 for r in R if abs(r @ alpha0 - 1) < 1e-9)
print(f"T_e = # common foam-neighbours of 0 and alpha0 = {T_e}  (expect 56)")

# 4-cycles through edge (0, alpha0):
#   path 0 -> beta -> gamma -> alpha0 -> 0 (back), with beta, gamma distinct,
#   beta != alpha0, gamma != 0, and {0, alpha0, beta, gamma} all distinct foam vertices.
#   beta is a foam-neighbour of 0, hence a ROOT.
#   gamma is a foam-neighbour of alpha0: gamma - alpha0 is a root, so gamma = alpha0 + delta.
#   gamma is a foam-neighbour of beta: gamma - beta is a root.
#   We don't require gamma to be a root (gamma may be 2nd-shell).

# Enumerate: for each beta in roots (beta != alpha0), for each delta in roots (delta != 0,
# delta != -alpha0 so gamma != 0; and we'll handle gamma != beta), check gamma - beta is a root.
C4 = 0
shell_counts = Counter()                 # 'root' / 'non-root' gamma
for ib, beta in enumerate(R):
    if np.allclose(beta, alpha0): continue
    for delta in R:
        gamma = alpha0 + delta
        if np.allclose(gamma, 0): continue          # gamma != 0
        if np.allclose(gamma, beta): continue       # gamma != beta
        diff = gamma - beta
        if abs(diff @ diff - 2.0) > 1e-9: continue  # gamma-beta must be a root (norm^2=2)
        C4 += 1
        # is gamma itself a root (1st-shell of 0)?
        if abs(gamma @ gamma - 2.0) < 1e-9 and key(gamma) in keymap:
            shell_counts['gamma_root'] += 1
        else:
            shell_counts['gamma_2ndshell'] += 1
print(f"\n4-cycles per foam-edge: C_4 = {C4}")
print(f"  decomposition: {dict(shell_counts)}")

# Some 4-cycles are degenerate "triangle+back" if gamma is on the triangle path; we've
# excluded only literal coincidences. A 4-cycle can also visit the same vertex twice
# if e.g. delta = -alpha0+ ... -- already filtered.

# The percolation expansion in cycle counts for 1/p_c (Karplus form) is
#   1/p_c ~ (z-1) - T_e - C_4 - ...
# but with proper combinatorial weights. Report candidate corrections and gap:
target = 172.55                          # measured
print(f"\n--- candidate corrections beyond triangles (target 1/p_c = {target}) ---")
print(f"  Bethe                                : 1/p_c = z-1 = {239}")
print(f"  + triangle (T_e=56)                  : 1/p_c = {239-56}   (gap {239-56-target:+.2f})")
print(f"  + 4-cycle naive (-C_4)               : 1/p_c = {239-56-C4}   (gap {239-56-C4-target:+.2f})")
# Half-weighting (each 4-cycle counted from both directions of the edge): naive double-count
print(f"  + 4-cycle / 2 (orientation factor)   : 1/p_c = {239-56-C4//2}   (gap {239-56-C4//2-target:+.2f})")
# Try dividing by other natural factors
for fac in (3, 4, 6, 8):
    v = 239 - 56 - C4//fac
    print(f"  + 4-cycle / {fac:>2}                       : 1/p_c = {v}   (gap {v-target:+.2f})")
