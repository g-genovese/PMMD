#!/usr/bin/env python3
"""First overlap: Yukawa = <Higgs zero-mode | fermion loop>. The Higgs is the
uniform zero-mode (the 5). We test what loop profiles give which overlaps. (PMMD v6.0)"""
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
R=e8_roots(); n=len(R)
zero=np.ones(n)/np.sqrt(n)          # Higgs zero-mode: uniform over the local foam

def overlap(psi):
    psi=psi/np.linalg.norm(psi); return abs(zero@psi)

print("Yukawa magnitude  y ~ |<zero-mode | loop>|   (zero-mode = uniform SM Higgs)")
print(f"  loop on 1 cell                 : y = {overlap(np.eye(n)[0]):.4f}")
for k in (4,16,56,120,240):
    p=np.zeros(n); p[:k]=1.0
    print(f"  loop spread on {k:3d} cells (in phase): y = {overlap(p):.4f}")
print("  -> in-phase loops span y ~ 0.065 ... 1.0  (covers TOP and heavy quarks only)")
print()
# extended loop with phases (cancellation): use a non-trivial foam mode as the loop
L=np.diag(((np.abs(R@R.T-1)<1e-9).astype(float)).sum(1))-(np.abs(R@R.T-1)<1e-9).astype(float)
val,vec=np.linalg.eigh(L)
print("extended loops = excited foam modes (carry phases / sign changes):")
for lab,target in [('deg-8 mode (omega^2=28)',28),('deg-35 (48)',48),('deg-112 (58)',58),('deg-84 (60)',60)]:
    v0=vec[:,np.argmin(np.abs(val-target))]
    print(f"  {lab:24s}: y = |<zero|mode>| = {overlap(v0):.2e}  (orthogonal -> ~0)")
print()
print("HONEST READ:")
print(" * In-phase localised loops give y in [0.065, 1]: this is the HEAVY sector")
print("   (top y~1 = most uniform loop). The mechanism gives the top naturally.")
print(" * Light fermions (y_e ~ 3e-6) need EXTENDED loops with phase cancellation -")
print("   overlaps with the uniform Higgs become exponentially small (and complex).")
print("   |overlap| = Yukawa magnitude, arg(overlap) = CP phase: UNIFIED in one object.")
print(" * But the ACTUAL loop profiles psi_f are the FERMIONIC foam sector - a separate")
print("   object from the bosonic spectrum above. No number without them. That is the wall.")
