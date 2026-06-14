#!/usr/bin/env python3
"""
analyze_121_pure58.py -- why does the pure-58 class give 121 A_2 subsystems?
Tests: (a) restricted-W(E8) orbit partition (120+1?), (b) identity of the special
A_2, (c) does 120 = #positive roots of E8 via a natural bijection, (d) 11x11?
Reuses cycle_classifier_v1 machinery.
"""
import numpy as np, itertools
from collections import Counter
import cycle_classifier_v1 as cc

R = cc.e8_roots()
keys = {cc.root_key(r): i for i, r in enumerate(R)}
grades = cc.harmonic_grades(R)
V112 = grades[3]; V_const = grades[0]
C = cc.a2_coxeter_order3()
perm = np.zeros((len(R), len(R)))
for i, r in enumerate(R):
    perm[keys[cc.root_key(C @ r)], i] = 1.0
A = V112.T @ (perm @ V112)
om = np.exp(2j*np.pi/3); I = np.eye(A.shape[0]); A2 = A @ A
P_1 = (I + A + A2)/3.0
P_om = (I + np.conj(om)*A + om*A2)/3.0
P_omb = (I + om*A + np.conj(om)*A2)/3.0

tris = cc.find_triangles(R)
pure = []
for (i, j, k) in tris:
    v = np.zeros(240); v[i]=v[j]=v[k]=1.0
    c = V112.T @ v
    n1 = np.linalg.norm(P_1@c)**2; no = np.linalg.norm(P_om@c)**2; nob = np.linalg.norm(P_omb@c)**2
    if abs(n1-1.5)<1e-3 and abs(no)<1e-3 and abs(nob)<1e-3:
        pure.append((i, j, k))

def a2_from_triangle(i, j, k):
    idxs = set([i, j, k])
    for idx in (i, j, k):
        idxs.add(keys[cc.root_key(-R[idx])])
    return frozenset(idxs)

pure_a2s = set(a2_from_triangle(*t) for t in pure)
print(f"pure-58 triangles: {len(pure)} ; A_2 subsystems: {len(pure_a2s)}  (=11^2={11**2})")

# ---- (a) restricted-W(E8) orbit partition ----
def reflect_a2(a2set, alpha_idx):
    a = R[alpha_idx]; out=[]
    for idx in a2set:
        r = R[idx]; rr = r - 2.0*(r@a)/(a@a)*a
        j = keys.get(cc.root_key(rr))
        if j is None: return None
        out.append(j)
    return frozenset(out)

# partition pure_a2s into orbits under reflections that STAY in pure_a2s
unassigned = set(pure_a2s); orbits=[]
while unassigned:
    seed = next(iter(unassigned)); orbit={seed}; frontier=[seed]
    while frontier:
        cur = frontier.pop()
        for ai in range(len(R)):
            img = reflect_a2(cur, ai)
            if img is not None and img in pure_a2s and img not in orbit:
                orbit.add(img); frontier.append(img)
    orbits.append(orbit); unassigned -= orbit
print(f"restricted-W(E8) orbits within pure-58: sizes {sorted(len(o) for o in orbits)}")

# ---- (b) identify the special (smallest-orbit / singleton) A_2 ----
orbits_sorted = sorted(orbits, key=len)
special = orbits_sorted[0]
print(f"\nsmallest orbit size = {len(special)}")
for a2 in special:
    roots_in = sorted(tuple(int(round(2*x)) for x in R[idx]) for idx in a2)
    # which coordinates are involved (support)
    support = sorted(set(c for idx in a2 for c in range(8) if abs(R[idx][c])>1e-6))
    print(f"  A_2 support coords = {support}  (family plane {{0,1,2}} ?)")
    for idx in sorted(a2):
        print(f"    {R[idx]}")
    break

# ---- (c) does the 120-orbit biject to E8 positive roots? ----
big = max(orbits, key=len)
print(f"\nlargest orbit size = {len(big)}  (#E8 positive roots = 120)")
# map each A_2 in the big orbit to its 'A_2 plane normal direction':
# the A_2 plane is 2D; its orthogonal complement is 6D. Characterize each A_2 by
# the pair of simple roots / by the sum of its 3 positive roots direction.
plane_keys = set()
for a2 in big:
    # positive roots = those with first nonzero entry > 0
    prs = []
    for idx in a2:
        r = R[idx]
        nz = next((x for x in r if abs(x)>1e-6), 0)
        if nz > 0: prs.append(r)
    # the A_2 plane is spanned by these; use a canonical key = sorted projector entries
    M = np.array(prs)
    P = M.T @ np.linalg.pinv(M @ M.T) @ M   # projector onto the 2-plane
    plane_keys.add(tuple(np.round(P.flatten(), 3)))
print(f"distinct A_2 2-planes in the big orbit: {len(plane_keys)}")

# ---- (d) 11x11 test: look for a pair of invariants each taking 11 values ----
# invariant 1: index of the A_2 plane among coordinate structure; try the multiset
# of |support| and pair patterns. Simpler: count how many A_2's share each support set.
support_counter = Counter()
for a2 in pure_a2s:
    support = frozenset(c for idx in a2 for c in range(8) if abs(R[idx][c])>1e-6)
    support_counter[len(support)] += 1
print(f"\npure-58 A_2 support-size distribution: {dict(support_counter)}")
print(f"interpretation: 121 = {len(big)} (one W(E8) orbit, =#E8 positive roots) "
      f"+ {len(special)} (special). 11^2 is then a numerical coincidence "
      f"unless an 11x11 product structure appears above.")
