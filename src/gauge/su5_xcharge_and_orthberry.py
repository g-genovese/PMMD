#!/usr/bin/env python3
"""
su5_xcharge_and_orthberry.py
============================
(i) SU(5) sub-split of the 16 via the U(1)_X charge of SU(5) in Spin(10).

  Embedding: family A_2 in coords {0,1,2}; E_6 U(1) = (e0+e1+e2)/sqrt(3);
  SO(10) Cartan = coords {3,4,5,6,7}; SU(5) U(1)_X = sum of SO(10) Cartan.
  X-charge of a weight = sum of its last-5 coordinates.

  For the 16 spinor of SO(10) the X-charges are
     +5/2 (x1, SU(5) singlet 1),  +1/2 (x10, SU(5) 10),  -3/2 (x5, SU(5) 5bar).
  Each (I,S,S) cycle has 2 spinor roots with opposite X (one in 16, one in 16bar),
  so |X| in {1/2, 3/2, 5/2} maps to {10, 5bar, 1}.  Expected multiplicities: 10,5,1.

(ii) Berry-like areas of cycle projections onto Spin(10) sub-Cartan 3-planes
  (orthogonal to the family direction) to see which distinguish the 16.
"""

import numpy as np
import itertools
from itertools import combinations_with_replacement as cwr, combinations
from collections import Counter

def e8_roots():
    roots = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1,-1):
            for sj in (1,-1):
                v = np.zeros(8); v[i],v[j] = si,sj; roots.append(v)
    for s in itertools.product((0.5,-0.5), repeat=8):
        if sum(1 for x in s if x<0)%2==0: roots.append(np.array(s))
    return np.array(roots)

def root_key(v, scale=2): return tuple(int(round(scale*x)) for x in v)
def root_type(v): return 'I' if np.max(np.abs(v))>0.75 else 'S'

def eval_monomials(R, deg):
    N=R.shape[0]
    if deg==0: return np.ones((N,1))
    cols=[]
    for combo in cwr(range(8),deg):
        col=np.ones(N)
        for idx in combo: col*=R[:,idx]
        cols.append(col)
    return np.array(cols).T

def harmonic_grades(R):
    cum=None; grades=[]
    for d in range(5):
        Md=eval_monomials(R,d)
        if cum is not None: Md=Md-cum@(cum.T@Md)
        U,s,_=np.linalg.svd(Md,full_matrices=False)
        V=U[:,s>1e-9*s[0]]; grades.append(V)
        cum=V if cum is None else np.hstack([cum,V])
    return grades

def reflect_matrix(a):
    a=np.asarray(a,float); return np.eye(8)-2.0*np.outer(a,a)/(a@a)
def a2_coxeter_order3():
    a=np.zeros(8); a[0]=1; a[1]=-1
    b=np.zeros(8); b[1]=1; b[2]=-1
    return reflect_matrix(a)@reflect_matrix(b)

def spherical_triangle_area(a,b,c):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); c=c/np.linalg.norm(c)
    num=np.dot(a,np.cross(b,c)); den=1+a@b+b@c+c@a
    return 2*np.arctan2(num,den)

def get_orbits():
    R=e8_roots(); keys={root_key(r):i for i,r in enumerate(R)}
    grades=harmonic_grades(R); V112=grades[3]
    C=a2_coxeter_order3()
    perm_idx=np.zeros(len(R),dtype=int); perm=np.zeros((len(R),len(R)))
    for i,r in enumerate(R):
        j=keys[root_key(C@r)]; perm_idx[i]=j; perm[j,i]=1.0
    A=V112.T@(perm@V112)
    om=np.exp(2j*np.pi/3); omb=np.conj(om); I8=np.eye(A.shape[0])
    P1=(I8+A+A@A)/3; Po=(I8+omb*A+om*(A@A))/3; Pob=(I8+om*A+omb*(A@A))/3
    matter=[]
    for i in range(len(R)):
        for j in range(i+1,len(R)):
            k=keys.get(root_key(-(R[i]+R[j])))
            if k is not None and k>j:
                v=np.zeros(240); v[i]=v[j]=v[k]=1.0; c=V112.T@v
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
    return R,keys,orbits

def pos_roots(R,a2):
    for trip in combinations(sorted(a2),3):
        if np.allclose(R[trip[0]]+R[trip[1]]+R[trip[2]],0): return list(trip)
    return None

def main():
    R,keys,orbits=get_orbits()
    iss=[]; iii=[]
    for orb in orbits:
        p=pos_roots(R,orb[0])
        t=tuple(sorted(root_type(R[i]) for i in p))
        if t==('I','S','S'): iss.append((orb,p))
        elif t==('I','I','I'): iii.append((orb,p))
    print(f"(I,S,S): {len(iss)}   (I,I,I): {len(iii)}\n")

    # ====== (i) SU(5) X-charge of the 16 ======
    print("===== (i) SU(5) U(1)_X charge of the 16 (I,S,S) cycles =====")
    print("X = sum of last-5 coords (SO(10) Cartan); |X| in {1/2,3/2,5/2} -> {10,5bar,1}\n")
    xmag_counts=Counter()
    details=[]
    for orb,p in iss:
        spinor_x=[]
        for idx in p:
            if root_type(R[idx])=='S':
                x=sum(R[idx][3:8])   # last 5 coords
                spinor_x.append(x)
        # both spinors should have opposite X; |X| is the invariant
        xmag=round(abs(spinor_x[0]),3)
        # consistency check
        consistent = abs(abs(spinor_x[0])-abs(spinor_x[1]))<1e-6
        xmag_counts[xmag]+=1
        details.append((xmag, spinor_x, consistent))
    print(f"|X| distribution over the 16 cycles: {dict(xmag_counts)}")
    print(f"All cycles have |X(s1)|=|X(s2)|: {all(d[2] for d in details)}")
    su5_map={0.5:'10', 1.5:'5bar', 2.5:'1'}
    print("\nSU(5) assignment:")
    for xmag in sorted(xmag_counts):
        print(f"  |X|={xmag} -> SU(5) {su5_map.get(xmag,'?')}: {xmag_counts[xmag]} cycles")
    expect = (xmag_counts.get(0.5,0), xmag_counts.get(1.5,0), xmag_counts.get(2.5,0))
    print(f"\n  (10, 5bar, 1) multiplicities = {expect}  (expect (10,5,1))")

    # ====== (ii) Orthogonal Berry-phase projections ======
    print("\n===== (ii) Berry-like areas on Spin(10) sub-Cartan 3-planes =====")
    # try several 3-subsets of {3,4,5,6,7}
    subsets = [(3,4,5),(3,4,6),(3,4,7),(5,6,7),(4,5,6),(3,5,7)]
    for sub in subsets:
        areas=Counter()
        for orb,p in iss:
            verts=[]
            ok=True
            for idx in p:
                v3=np.array([R[idx][c] for c in sub])
                n=np.linalg.norm(v3)
                if n<1e-9: ok=False; break
                verts.append(v3/n)
            if ok:
                a=round(abs(spherical_triangle_area(*verts)),3)
            else:
                a=-1.0
            areas[a]+=1
        print(f"  proj {sub}: area distribution {dict(areas)}")

    # ====== Cross-check: does |X| correlate with a specific orthogonal-Berry? ======
    print("\n===== Cross-check: |X| class vs full SO(10) Cartan structure =====")
    for orb,p in iss:
        spinor_x=[sum(R[idx][3:8]) for idx in p if root_type(R[idx])=='S']
        xmag=round(abs(spinor_x[0]),3)
        # number of nonzero last-5 entries in each spinor (all 5 are +-1/2 so always 5)
        # better: count of minus signs in last 5 of the '16-side' spinor (X>0 in {0.5,2.5}, or X<0=-1.5)
    print("  (X-charge magnitude is the SU(5) discriminant; see (i).)")

if __name__=="__main__":
    main()
