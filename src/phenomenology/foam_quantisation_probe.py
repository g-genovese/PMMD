#!/usr/bin/env python3
"""First concrete step of quantising the foam field: the harmonic (free-field)
fluctuation spectrum = graph Laplacian on the local E8 foam (240 roots, z=240).
Eigenmodes = foam deformation modes; degeneracies = W(E8)-irrep dims. (PMMD v6.0)"""
import itertools, numpy as np
def e8_roots():
    R=[]
    for i in range(8):
        for j in range(i+1,8):
            for si in(1,-1):
                for sj in(1,-1):
                    v=np.zeros(8); v[i],v[j]=si,sj; R.append(v)
    for s in itertools.product([.5,-.5],repeat=8):
        if list(s).count(-.5)%2==0: R.append(np.array(s))
    return np.array(R)
R=e8_roots(); n=len(R); G=R@R.T
# foam adjacency: roots joined by an edge of the 4_21 polytope <=> inner product = 1
A=(np.abs(G-1)<1e-9).astype(float); deg=A.sum(1)
print(f"roots={n}  degree (z-local)={int(deg[0])}  (uniform: {np.allclose(deg,deg[0])})")
L=np.diag(deg)-A                      # graph Laplacian = free fluctuation operator
ev=np.linalg.eigvalsh(L)
# group eigenvalues into degenerate shells
shells=[]; ev_r=np.round(ev,6)
for e in sorted(set(ev_r)):
    shells.append((e,int((ev_r==e).sum())))
print("\nLaplacian spectrum (mode energy, degeneracy) -- the foam deformation modes:")
for e,d in shells: print(f"   omega^2 = {e:8.3f}   degeneracy = {d}")
print(f"\n   zero mode (uniform rescaling = the '5'/SM-Higgs direction): deg {shells[0][1]} at omega^2=0")
tot=sum(d for _,d in shells); print(f"   total modes = {tot}")
print("\n[interpretation] The lowest non-zero shell is the reflection/translation sector;")
print("higher shells are anisotropic shape modes. Degeneracies are W(E8)-irrep dimensions;")
print("mapping them to SU(5) Higgs reps (5,45,50) needs the W(E8)->...->SU(5) branching.")
print("This is the HARMONIC part only: the full quantisation (interacting modes, loop")
print("wavefunctions, overlaps -> Yukawa weights & CP phases) is beyond this free spectrum.")
