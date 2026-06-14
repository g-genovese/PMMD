#!/usr/bin/env python3
"""
sm_identification.py
====================
Final stage of the cycle catalogue: assign each of the 16 (I,S,S) matter
cycles its Standard-Model quantum numbers under SU(3)_c x SU(2)_L x U(1)_Y.

Embedding (continuing su5_xcharge_and_orthberry.py):
  family A_2 : coords {0,1,2}
  SO(10) Cartan : coords {3,4,5,6,7}
  SU(5) fundamental index : the 5 coords {3,4,5,6,7}
  -> SU(3)_color acts on {3,4,5}, SU(2)_weak on {6,7}
  -> hypercharge Y = -1/3 (a3+a4+a5) + 1/2 (a6+a7)   (Georgi-Glashow)

For each cycle pick the '16-side' spinor root (X-charge in {+5/2,+1/2,-3/2}),
read off:
  color type  : pattern of (a3,a4,a5)  (singlet if all same, triplet if mixed)
  weak type   : pattern of (a6,a7)      (singlet if same, doublet if mixed)
  hypercharge : Y as above

Expected Standard-Model content of the 16:
  Q (3,2,+1/6) x6 ; u^c (3bar,1,-2/3) x3 ; e^c (1,1,+1) x1   [the 10]
  d^c (3bar,1,+1/3) x3 ; L (1,2,-1/2) x2                       [the 5bar]
  nu^c (1,1,0) x1                                              [the 1]
"""

import numpy as np, itertools
from itertools import combinations_with_replacement as cwr, combinations
from collections import Counter, defaultdict

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

def get_orbits():
    R=e8_roots(); keys={root_key(r):i for i,r in enumerate(R)}
    V112=harmonic_grades(R)[3]; C=a2_coxeter_order3()
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
    return R,keys,orbits
def pos_roots(R,a2):
    for trip in combinations(sorted(a2),3):
        if np.allclose(R[trip[0]]+R[trip[1]]+R[trip[2]],0): return list(trip)
    return None

def main():
    R,keys,orbits=get_orbits()
    iss=[]
    for orb in orbits:
        p=pos_roots(R,orb[0])
        if tuple(sorted(root_type(R[i]) for i in p))==('I','S','S'): iss.append((orb,p))
    print(f"(I,S,S) matter cycles: {len(iss)}\n")

    rows=[]
    for orb,p in iss:
        # spinor roots and their X charges
        spinors=[R[i] for i in p if root_type(R[i])=='S']
        xs=[sum(s[3:8]) for s in spinors]
        # 16-side = the spinor whose X is in {+2.5,+0.5,-1.5}
        side=None
        for s,x in zip(spinors,xs):
            if round(x,3) in (2.5,0.5,-1.5): side=s; xside=x; break
        if side is None:  # fallback
            side=spinors[0]; xside=xs[0]
        a=side
        col=a[3:6]      # color coords {3,4,5}
        wk =a[6:8]      # weak coords {6,7}
        ncol_minus=int(sum(1 for x in col if x<0))
        nwk_minus =int(sum(1 for x in wk if x<0))
        Y = -1/3*sum(col) + 1/2*sum(wk)
        color_type = 'singlet' if ncol_minus in (0,3) else 'triplet'
        weak_type  = 'singlet' if nwk_minus in (0,2) else 'doublet'
        rows.append({'X':round(xside,2),'Y':round(Y,3),
                     'color':color_type,'weak':weak_type,
                     'ncol_minus':ncol_minus,'nwk_minus':nwk_minus,
                     'root':a})

    # ---- Group by |X| (SU(5) rep), then by (color,weak,Y) ----
    print("===== SM quantum numbers of the 16 matter cycles =====\n")
    by_x=defaultdict(list)
    for r in rows: by_x[abs(r['X'])].append(r)

    su5={2.5:'1 (singlet)',0.5:'10',1.5:'5bar'}
    for xmag in sorted(by_x):
        group=by_x[xmag]
        print(f"--- |X|={xmag}  (SU(5) {su5[xmag]}), {len(group)} cycles ---")
        # tabulate by (color,weak,Y)
        cls=Counter((r['color'],r['weak'],r['Y']) for r in group)
        for (c,w,Y),n in sorted(cls.items(), key=lambda x:(-x[1])):
            print(f"   color={c:8s} weak={w:8s} Y={Y:+.3f} : {n} states")
        print()

    # ---- SM particle assignment ----
    print("===== Standard-Model assignment =====")
    # Q:(triplet,doublet) u^c:(triplet,singlet) e^c:(singlet,singlet) in the 10
    # d^c:(triplet,singlet) L:(singlet,doublet) in the 5bar; nu^c:(singlet,singlet) in 1
    def assign(r):
        x=abs(r['X']); c=r['color']; w=r['weak']
        if x==0.5:
            if c=='triplet' and w=='doublet': return 'Q  (3,2,+1/6)'
            if c=='triplet' and w=='singlet': return 'u^c(3bar,1,-2/3)'
            if c=='singlet' and w=='singlet': return 'e^c(1,1,+1)'
        if x==1.5:
            if c=='triplet' and w=='singlet': return 'd^c(3bar,1,+1/3)'
            if c=='singlet' and w=='doublet': return 'L  (1,2,-1/2)'
        if x==2.5:
            return 'nu^c(1,1,0)'
        return '??'
    counts=Counter(assign(r) for r in rows)
    for part,n in sorted(counts.items()):
        print(f"  {part:20s}: {n}")
    print(f"\n  Total: {sum(counts.values())} (expect 16)")
    print(f"  Expected: Q x6, u^c x3, e^c x1, d^c x3, L x2, nu^c x1")

if __name__=="__main__":
    main()
