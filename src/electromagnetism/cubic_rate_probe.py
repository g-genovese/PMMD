#!/usr/bin/env python3
"""
cubic_rate_probe.py
===================
Attempt at the Yukawa cubic rate y_f = <H | psi_f psi_f-bar>.

THE PROBLEM, precisely:
  In the relational network the Yukawa is the rate at which three modes
  reconfigure: the winding mode H (degree-0 harmonic = constant on the 240
  roots = the scale mode), and a matter cycle psi_f with its conjugate.
  The static structural proxy is the triple overlap on the root graph:
      y_f^(0) = sum_x  H(x) * psi_f(x) * psi_f(x)
  with H = const (homogeneous winding condensate at the VEV).

KEY STRUCTURAL FACTS to test:
  (a) LEADING term with constant H: y_f^(0) proportional to ||psi_f||^2,
      hence UNIFORM across all matter cycles -> all fermions get mass ~ v at
      leading order. (Top ~ v is then the 'unsuppressed' natural case; the
      hierarchy is the SUPPRESSION of the lighter fermions, a subleading effect.)
  (b) The generational hierarchy lives in the Z_3 orbit (3 generations of one
      cycle). At the level of the homogeneous mode the 3 are degenerate
      (Z_3-symmetric). Splitting requires Z_3 BREAKING. We compute a
      Z_3-graded Berry-like quantity per generation in a fixed Z_3-breaking
      projection and Z_3-Fourier decompose it (connection to the Koide form
      sqrt(m_k) = a0 + 2 a1 cos(2 pi k /3 + delta)).

WHAT IS MISSING (honest): the actual dynamical RATE (the cubic vertex
coefficient) requires the quantized network Hamiltonian. The static overlap
gives uniform leading mass; the exponential hierarchy (10^6 span) needs the
dynamical/RG suppression that is not computable here. We quantify the leading
result and the *structure* of the subleading Z_3 grading only.
"""

import numpy as np, itertools
from itertools import combinations_with_replacement as cwr, combinations
from collections import defaultdict

def e8_roots():
    roots=[]
    for i,j in itertools.combinations(range(8),2):
        for si in (1,-1):
            for sj in (1,-1):
                v=np.zeros(8); v[i],v[j]=si,sj; roots.append(v)
    for s in itertools.product((0.5,-0.5),repeat=8):
        if sum(1 for x in s if x<0)%2==0: roots.append(np.array(s))
    return np.array(roots)
def root_key(v,scale=2): return tuple(int(round(scale*x)) for x in v)
def root_type(v): return 'I' if np.max(np.abs(v))>0.75 else 'S'
def eval_monomials(R,deg):
    N=R.shape[0]
    if deg==0: return np.ones((N,1))
    cols=[]
    for combo in cwr(range(8),deg):
        col=np.ones(N)
        for idx in combo: col*=R[:,idx]
        cols.append(col)
    return np.array(cols).T
def harmonic_grades(R):
    cum=None; g=[]
    for d in range(5):
        Md=eval_monomials(R,d)
        if cum is not None: Md=Md-cum@(cum.T@Md)
        U,s,_=np.linalg.svd(Md,full_matrices=False)
        V=U[:,s>1e-9*s[0]]; g.append(V); cum=V if cum is None else np.hstack([cum,V])
    return g
def reflect_matrix(a):
    a=np.asarray(a,float); return np.eye(8)-2*np.outer(a,a)/(a@a)
def a2_coxeter_order3():
    a=np.zeros(8);a[0]=1;a[1]=-1; b=np.zeros(8);b[1]=1;b[2]=-1
    return reflect_matrix(a)@reflect_matrix(b)
def spherical_triangle_area(a,b,c):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); c=c/np.linalg.norm(c)
    num=np.dot(a,np.cross(b,c)); den=1+a@b+b@c+c@a
    return 2*np.arctan2(num,den)

def get_setup():
    R=e8_roots(); keys={root_key(r):i for i,r in enumerate(R)}
    grades=harmonic_grades(R); V112=grades[3]; Vconst=grades[0]
    C=a2_coxeter_order3()
    perm_idx=np.zeros(len(R),dtype=int); perm=np.zeros((len(R),len(R)))
    for i,r in enumerate(R):
        j=keys[root_key(C@r)]; perm_idx[i]=j; perm[j,i]=1
    A=V112.T@(perm@V112); om=np.exp(2j*np.pi/3); omb=np.conj(om); I8=np.eye(A.shape[0])
    P1=(I8+A+A@A)/3; Po=(I8+omb*A+om*(A@A))/3; Pob=(I8+om*A+omb*(A@A))/3
    matter=[]
    for i in range(len(R)):
        for j in range(i+1,len(R)):
            k=keys.get(root_key(-(R[i]+R[j])))
            if k is not None and k>j:
                v=np.zeros(240); v[i]=v[j]=v[k]=1; c=V112.T@v
                sig=(round(float(np.linalg.norm(P1@c)**2),3),
                     round(float(np.linalg.norm(Po@c)**2),3),
                     round(float(np.linalg.norm(Pob@c)**2),3))
                if sig==(0.167,0.667,0.667): matter.append((i,j,k))
    def a2_of(i,j,k):
        s={i,j,k}
        for idx in (i,j,k): s.add(keys[root_key(-R[idx])])
        return frozenset(s)
    a2set=set(a2_of(*t) for t in matter)
    def applyC(a2): return frozenset(perm_idx[i] for i in a2)
    seen=set(); orbits=[]
    for a2 in a2set:
        if a2 in seen: continue
        orb=[]; cur=a2
        while cur not in seen: seen.add(cur); orb.append(cur); cur=applyC(cur)
        orbits.append(orb)
    return R,keys,V112,Vconst,perm_idx,orbits

def pos_roots(R,a2):
    for trip in combinations(sorted(a2),3):
        if np.allclose(R[trip[0]]+R[trip[1]]+R[trip[2]],0): return list(trip)
    return None

def main():
    R,keys,V112,Vconst,perm_idx,orbits=get_setup()

    # ===== (a) Leading overlap with the homogeneous winding mode =====
    print("===== (a) Leading Yukawa overlap (homogeneous winding mode) =====")
    Hconst = Vconst[:,0]   # normalised constant function on 240 roots
    y0=[]
    for orb in orbits:
        p=pos_roots(R,orb[0])
        psi=np.zeros(240)
        for idx in p: psi[idx]=1.0
        psiV = V112@(V112.T@psi)        # project the cycle into the matter space
        # triple overlap with constant H:  sum_x H(x) psi(x)^2
        yval = float(np.sum(Hconst * psiV * psiV))
        y0.append(yval)
    y0=np.array(y0)
    print(f"  leading overlaps: min={y0.min():.5f} max={y0.max():.5f} "
          f"spread={y0.max()-y0.min():.2e}")
    print(f"  -> uniform to {np.std(y0)/np.mean(y0):.1e} relative: all fermions")
    print(f"     couple equally to the homogeneous winding mode (mass ~ v).")
    print(f"     The top (overlap~1) is the natural case; lighter fermions are")
    print(f"     the SUPPRESSED (subleading) cases. Hierarchy = suppression.\n")

    # ===== (b) Z_3-graded subleading structure =====
    print("===== (b) Z_3-graded Berry structure across the 3 generations =====")
    # For each Z_3 orbit (3 generations), project each generation onto a fixed
    # Z_3-breaking 3-space and compute the spherical triangle area.
    proj = [(0,3,4),(0,3,5),(0,4,5),(1,3,4)]   # one family coord + 2 SO(10)
    for sub in proj:
        print(f"\n  Projection {sub} (family-breaking):")
        fourier_mags=[]
        for orb in orbits[:6]:   # sample first 6 orbits
            gen_areas=[]
            for a2 in orb:        # the 3 generations
                p=pos_roots(R,a2)
                verts=[]; ok=True
                for idx in p:
                    v3=np.array([R[idx][c] for c in sub]); n=np.linalg.norm(v3)
                    if n<1e-9: ok=False; break
                    verts.append(v3/n)
                gen_areas.append(abs(spherical_triangle_area(*verts)) if ok else 0.0)
            gen_areas=np.array(gen_areas)
            # Z_3 Fourier: a0 (uniform), a1 (modulation magnitude)
            om=np.exp(2j*np.pi/3)
            f0=abs(gen_areas.sum())/3
            f1=abs(sum(gen_areas[k]*om**k for k in range(3)))/3
            fourier_mags.append((round(f0,3),round(f1,3)))
        # report the spread of generation areas
        nonzero=[fm for fm in fourier_mags if fm[0]>1e-6]
        if nonzero:
            print(f"    (a0_uniform, a1_modulation) per orbit: {nonzero[:6]}")
            mod_ratio=[fm[1]/fm[0] if fm[0]>1e-9 else 0 for fm in nonzero]
            print(f"    modulation/uniform ratios: {[round(m,3) for m in mod_ratio]}")
        else:
            print(f"    all-degenerate or projection kills a vertex.")

    print("\n===== Honest status =====")
    print("Leading term: rigorous, uniform -> all fermions ~ v; top natural.")
    print("Subleading (hierarchy): Z_3-graded Berry areas are O(1) modulations,")
    print("NOT the exponential 10^6 hierarchy. The exponential suppression needs")
    print("the dynamical rate (quantized network Hamiltonian) + RG flow, which is")
    print("not computable from the static root geometry alone.")

if __name__=="__main__":
    main()
