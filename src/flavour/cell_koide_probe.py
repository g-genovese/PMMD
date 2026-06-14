#!/usr/bin/env python3
"""Honest probe: does the E8 cell / orbit-type geometry (physical+internal) already
produce the Koide structure, or only the Z3 backbone? (PMMD exploration)"""
import itertools, numpy as np
phi=(1+5**.5)/2; pi=np.pi

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
def cox():
    e=np.eye(8)
    s=np.array([0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),e[0]+e[1],e[1]-e[0],e[2]-e[1],e[3]-e[2],e[4]-e[3],e[5]-e[4],e[6]-e[5]])
    def refl(al):
        M=np.eye(8)
        for k in range(8):M[:,k]=e[k]-2*(e[k]@al)/(al@al)*al
        return M
    w=np.eye(8)
    for al in s:w=refl(al)@w
    return w
def proj(exps):
    w=cox();val,vec=np.linalg.eig(w)
    ex=np.round(np.angle(val)/(2*pi)*30).astype(int)%30
    b=[]
    for t in exps:
        k=[i for i in range(8) if ex[i]==t][0]
        v=vec[:,k]/np.linalg.norm(vec[:,k]);b+=[np.real(v),np.imag(v)]
    Q,_=np.linalg.qr(np.array(b).T);return Q[:,:4]@Q[:,:4].T

roots=e8_roots()
Ppar=proj((1,11)); Pperp=proj((7,13))
rpar =np.array([np.linalg.norm(Ppar@r)  for r in roots])
rperp=np.array([np.linalg.norm(Pperp@r) for r in roots])
print("E8 -> H4: physical shells",sorted(set(np.round(rpar,3))),"internal shells",sorted(set(np.round(rperp,3))))

def Kq(u):  # Koide Q from amplitudes u=sqrt(m); Q=2/3 <=> vector at 45deg from democratic
    u=np.array(u,float); return (u**2).sum()/u.sum()**2
def ang(u): # angle of amplitude vector from democratic axis
    u=np.array(u,float); return np.degrees(np.arccos(u.sum()/(np.sqrt(3)*np.linalg.norm(u))))

print("\n[1] Three icosahedral orbit-types (the Z3 generation backbone)")
# canonical icosahedral special orbits radii (computed earlier): faces(20),edges(30),verts(12)
r3=np.array([0.7947,0.8507,1.0])
print(f"    orbit radii = {r3}; are they Z3 (120deg) related? the A5 face-stabiliser IS Z3 -> yes structurally")

print("\n[2] Does a NATURAL geometric amplitude give Koide Q=2/3? (45deg test)")
for lbl,u in [('sqrt(m)~orbit radius',r3),('sqrt(m)~orbit size {12,20,30}',[12,20,30]),
              ('sqrt(m)~1/radius',1/r3),('sqrt(m)~internal radius shells',sorted(set(np.round(rperp,3)))[:3])]:
    print(f"    {lbl:34s}: Q={Kq(u):.3f}  angle={ang(u):.1f}deg  ({'=2/3!' if abs(Kq(u)-2/3)<.02 else 'not 2/3'})")
print(f"    [Koide target: Q=2/3 <=> 45.0deg ; observed leptons 45.0, up~51.3, down~47.4]")

print("\n[3] What the geometry gives vs what Koide needs:")
print("    - Z3 / 120-degree backbone (3 generations): YES, intrinsic to the orbit-types.")
print("    - the sqrt2 amplitude & theta=2/9 phase (=> 45deg, Q=2/3): NOT from bare radii.")
print("      These are the L2-norm normalisation + SU(3) Casimir = representation weighting.")
print("\nHONEST VERDICT: bare cell geometry yields the Z3 three-fold backbone but NOT the")
print("Koide amplitude. The pattern IS 'from the bottom' -- but the relevant bottom is the")
print("E8 REPRESENTATION content (the weighted overlaps), not the cell radii. The sectoral")
print("quark deformation must likewise come from colour/representation weighting, not shape.")
