#!/usr/bin/env python3
"""Honest probe: do the foam's natural geometric orbit-types give the
generation mass hierarchy, or only O(1)/phi scales? (PMMD v6.0 exploration)"""
import itertools, numpy as np
phi=(1+5**.5)/2

# ---- E8 roots ----
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

# ---- Coxeter element & H4 physical/internal split (Moody-Patera) ----
def cox():
    e=np.eye(8)
    simple=np.array([0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),
                     e[0]+e[1],e[1]-e[0],e[2]-e[1],e[3]-e[2],e[4]-e[3],e[5]-e[4],e[6]-e[5]])
    def refl(al):
        M=np.eye(8)
        for k in range(8): M[:,k]=e[k]-2*(e[k]@al)/(al@al)*al
        return M
    w=np.eye(8)
    for al in simple: w=refl(al)@w
    return w
def proj(exps):
    w=cox(); val,vec=np.linalg.eig(w)
    ex=np.round(np.angle(val)/(2*np.pi)*30).astype(int)%30
    b=[]
    for t in exps:
        k=[i for i in range(8) if ex[i]==t][0]
        v=vec[:,k]/np.linalg.norm(vec[:,k]); b+=[np.real(v),np.imag(v)]
    Q,_=np.linalg.qr(np.array(b).T); return Q[:,:4]@Q[:,:4].T

print("="*64)
print("[1] E8 -> H4 physical-space shells (the two 600-cells)")
roots=e8_roots()
Ppar=proj((1,11)); Pperp=proj((7,13))   # physical / internal Coxeter planes
rpar=np.array([np.linalg.norm(Ppar@r) for r in roots])
shells=sorted(set(np.round(rpar,3)))
shells=[s for s in shells if s>0.05]
print(f"    physical shells = {shells}, ratio = {max(shells)/min(shells):.4f} (phi={phi:.4f})")

print("\n[2] The three icosahedral orbit-types (5/3/2-fold axes: 12/20/30 pts)")
# icosahedron circumradius 1
V=[]
for a,b in itertools.product([1,-1],[phi,-phi]):
    V+= [(0,a,b),(a,b,0),(b,0,a)]
V=np.unique(np.array(V),axis=0); V=V/np.linalg.norm(V[0])   # 12 verts, R=1
# edges: pairs at min distance
from scipy.spatial.distance import pdist,squareform
D=squareform(pdist(V)); el=np.min(D[D>1e-6])
edges=[(i,j) for i in range(12) for j in range(i+1,12) if abs(D[i,j]-el)<1e-6]
emid=np.array([ (V[i]+V[j])/2 for i,j in edges])           # 30 edge midpoints
# faces: triples mutually at edge length
faces=[(i,j,k) for i,j,k in itertools.combinations(range(12),3)
       if abs(D[i,j]-el)<1e-6 and abs(D[i,k]-el)<1e-6 and abs(D[j,k]-el)<1e-6]
fcen=np.array([ (V[i]+V[j]+V[k])/3 for i,j,k in faces])     # 20 face centers
rv,re,rf=1.0, np.linalg.norm(emid[0]), np.linalg.norm(fcen[0])
print(f"    vertices(12) R={rv:.4f} | edges(30) R={re:.4f} | faces(20) R={rf:.4f}")
print(f"    ratios to smallest: {rv/rf:.3f} : {re/rf:.3f} : {rv/rf:.3f}  -> all O(1)")

print("\n[3] HONEST TEST: can a one-parameter localisation m~exp(k*r) give 1:207:3477?")
mass=np.array([1.0,206.77,3477.2]); lr=np.log(mass)
r=np.sort([rf,re,rv])              # three geometric radii, ascending
# fit single k by least squares on ln m = k*r + c
A=np.vstack([r,np.ones(3)]).T; (k,c),res,_,_=np.linalg.lstsq(A,lr,rcond=None)
pred=np.exp(A@np.array([k,c]))
print(f"    geometric radii (sorted) = {np.round(r,4)}")
print(f"    gap ratio (r3-r1)/(r2-r1) = {(r[2]-r[0])/(r[1]-r[0]):.3f}   (lepton needs ln(3477)/ln(207)={np.log(3477)/np.log(207):.3f})")
print(f"    best single-k fit: k={k:.2f}; predicted ratios = {np.round(pred/pred[0],1)}  vs 1:207:3477")
print(f"    residual = {res[0] if len(res) else 0:.3f}  -> {'works' if (len(res) and res[0]<0.2) else 'does NOT fit with one param'}")
print("="*64)
