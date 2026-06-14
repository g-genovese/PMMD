#!/usr/bin/env python3
"""Fermionic sector: 3 generations = 3 Z3-related loop states; uniform Higgs gives
a circulant mass matrix. Test: does a hierarchy emerge, and from what? (PMMD v6.0)"""
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
def coxeter():
    e=np.eye(8)
    sm=[0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),e[0]+e[1],e[1]-e[0],e[2]-e[1],e[3]-e[2],e[4]-e[3],e[5]-e[4],e[6]-e[5]]
    def refl(al):
        M=np.eye(8)
        for k in range(8): M[:,k]=e[k]-2*(e[k]@al)/(al@al)*al
        return M
    w=np.eye(8)
    for al in sm: w=refl(al)@w
    return w
R=e8_roots(); n=len(R); zero=np.ones(n)/np.sqrt(n)
w3=np.linalg.matrix_power(coxeter(),10)              # order-3 (generation Z3)
idx=np.array([np.argmin(((R-w3@R[j])**2).sum(1)) for j in range(n)])

print("3 generations = a loop |g0> and its Z3 images |g1>=Pi|g0>, |g2>=Pi^2|g0>.")
print("Uniform Higgs => mass matrix M_jk ~ <g_j|g_k> (circulant).")
print("Circulant eigenvalues => ratios (1+2c):(1-c):(1-c), c=<g0|g1>.\n")
print(f"{'loop spread':>16}{'overlap c':>12}{'m3:m2:m1':>22}")
def gen_images(p):
    p=p/np.linalg.norm(p); p1=p[idx]; p2=p1[idx]; return p,p1,p2
for k in (1,8,30,120,240):
    p=np.zeros(n); p[:k]=1.0
    g0,g1,g2=gen_images(p)
    c=g0@g1
    M=np.array([[g0@g0,g0@g1,g0@g2],[g1@g0,g1@g1,g1@g2],[g2@g0,g2@g1,g2@g2]])
    ev=np.sort(np.abs(np.linalg.eigvalsh(M)))[::-1]
    r=ev/ev[0] if ev[0]>1e-12 else ev
    print(f"{k:>13d}    {c:>11.3f}   {r[0]:6.3f}:{r[1]:6.3f}:{r[2]:6.3f}")
print()
# what c reproduces an up-type-like ratio m_t:m_c ~ 136 ?
print("target: m3/m2 = (1+2c)/(1-c).  For top/charm ~136 -> c =", round((136-1)/(136+2),4))
print("        for tau/mu ~ 17        -> c =", round((17-1)/(17+2),4))
print()
print("HONEST READ:")
print(" * Structure is forced: Z3 family symmetry + uniform Higgs => CIRCULANT mass")
print("   matrix => ratios (1+2c):(1-c):(1-c). The 3rd generation is heaviest;")
print("   the 1st & 2nd are DEGENERATE at this order (split only by Z3-breaking).")
print(" * The ENTIRE generational hierarchy collapses to ONE number: the inter-")
print("   generation loop overlap c. c->1 (aligned/uniform loops) => strong hierarchy")
print("   (heavy,0,0); c=0 (localised) => degenerate (no hierarchy).")
print(" * c is set by the loop PROFILES (the fermionic foam wavefunctions) = the wall.")
print("   And the 1-2 splitting needs Z3-breaking = the next layer.")
