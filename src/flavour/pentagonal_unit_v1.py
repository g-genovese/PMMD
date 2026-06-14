"""PMMD - the pentagonal transport unit and the PMNS grid charges (session 2026-06-13).

RESULT (closes named target 1, supports target 2):
  g = w^{-6}, with w the chain-anchored Coxeter element (McKay order, case-(G)),
  is a lattice symmetry of order 5 acting as the COHERENT +2pi/5 rotation on
  E_par (both H4 planes, exponents {1,29},{11,19}) and as -+ 2*(2pi/5) on the two
  E_perp planes (m=13: -144 deg; m=7: +144 deg).
  => available pentagonal transport charges: {0 (invariants), +1 (E_par), +-2 (E_perp)}.
  Operator channel map: 1,diag(v) -> contraction/invariant (0);
  J (democratic/E6-axis, mode-0) -> E_par (+1);  v(x)v (condensate bilinear) ->
  E_perp m=13 (-2), selected BOTH as the Galois conjugate (sigma: m -> 7m mod 30,
  the sqrt5-flip) of the chain-entry-dominant parallel plane m=11, AND by the
  weight axis's own perp content (|z13(w)|>|z7(w)|).
  The CP-mirror grid (0,-1,0,+2) is the sigma/orientation conjugate, consistent
  with delta <-> 360-delta.
STATUS: quantum + charge menu Stratum 1-2 (exponent arithmetic); channel map
  Stratum 2 (explicit foam-path holonomy = remaining Stratum-3 item).
Target 2 support: per-insertion phase pattern (0,0,x,2x) is NOT realizable as a
  mode rephasing (residual 3e-2): the VEV-phase reading is unique. Geometric
  witness for the golden grading: long-class root shadow ratio perp/par = phi^-2
  EXACT (alpha3 family-root class).
Target 3 (Clebsch): NOT closed; curated scan matches are look-elsewhere
  compatible (best: u/v ~ |z7(w)|/|z13(w)| at -0.13%, flag-only).
"""
import numpy as np, itertools
phi=(1+np.sqrt(5))/2
e=np.eye(8)
a={1:0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),2:e[0]+e[1],3:e[1]-e[0],
   4:e[2]-e[1],5:e[3]-e[2],6:e[4]-e[3],7:e[5]-e[4],8:e[6]-e[5]}
def refl(v): return np.eye(8)-2*np.outer(v,v)/(v@v)
w=np.eye(8)
for i in [8,7,6,5,4,3,1,2]: w=w@refl(a[i])
ev,V=np.linalg.eig(w); ang=np.angle(ev); ms=np.round(np.abs(ang)*30/(2*np.pi)).astype(int)
planes={}
for m in (1,11,7,13):
    idx=[k for k in range(8) if ms[k]==m and ang[k]>0][0]
    u=V[:,idx]; f=np.real(u); f/=np.linalg.norm(f); g0=np.imag(u); g0-=(g0@f)*f; g0/=np.linalg.norm(g0)
    planes[m]=(f,g0)
def z(m,x): f,g0=planes[m]; return (x@f)+1j*(x@g0)
G=np.linalg.matrix_power(w,30-6)   # g = w^{-6}
print("g=w^-6: ordine 5:",np.allclose(np.linalg.matrix_power(G,5),np.eye(8)))
e0=np.eye(8)
test=a[3]
for m in (1,11,7,13):
    sh=np.degrees(np.angle(z(m,G@test))-np.angle(z(m,test)))%360
    print("  shift piano m=%2d: %8.3f deg"%(m,sh))
def emb(v3): x=np.zeros(8); x[:3]=v3; return x/np.linalg.norm(x)
nhat=emb([1,1,1]); what=emb([2,-1,-1])
print("entrata catena alpha5: |z1|=%.4f |z11|=%.4f  (dominanza m=11)"%(abs(z(1,a[5]/np.sqrt(2))),abs(z(11,a[5]/np.sqrt(2)))))
print("Galois sigma: m->7m mod 30: 11 -> %d  (canale perp selezionato)"%((7*11)%30))
print("peso perp di w_hat: |z7|=%.5f |z13|=%.5f  (lean verso m=13: %s)"%(abs(z(7,what)),abs(z(13,what)),abs(z(13,what))>abs(z(7,what))))
# Galois pairing (teorema, aritmetica esponenti): sigma(zeta30)=zeta30^7 =>
# autopiano {m,30-m} -> {7m mod 30, 30-7m}: {1,29}->{7,23}; {11,19}->{17,13} = piano m=13.
print("sigma: piano{1,29}->piano{7,23};  piano{11,19}->piano{17,13}  (= m=13)  [aritmetica]")
# conjugacy sqrt5 verificata a livello di settore: |Ppar x|^2 <-> |Pperp x|^2
Ppar=sum(np.outer(planes[m][0],planes[m][0])+np.outer(planes[m][1],planes[m][1]) for m in (1,11))
Pperp=np.eye(8)-Ppar
def fit5(v):
    for dd in (1,2,3,5,6,10,15,30,60):
        for pp in range(-4*dd,4*dd+1):
            qq=np.round((v-pp/dd)/np.sqrt(5)*dd)
            if abs(pp/dd+qq/dd*np.sqrt(5)-v)<1e-9: return int(pp),int(qq),dd
    return None
for nm,x in (('n_hat',nhat),('w_hat',what)):
    f1=fit5(x@Ppar@x); f2=fit5(x@Pperp@x)
    ok=(f1 is not None and f2 is not None and f1[2]==f2[2] and f1[0]==f2[0] and f1[1]==-f2[1])
    print("  %s: |Ppar|^2=(%d%+d*sqrt5)/%d  |Pperp|^2=(%d%+d*sqrt5)/%d  coniugati-sqrt5: %s"%((nm,)+f1+f2+(ok,)))
# inflazione: sigma(M) = -phi^{-1} Ppar + phi Pperp = -M^{-1}; integrale su base radici
M=phi*Ppar-(1/phi)*Pperp; sM=-(1/phi)*Ppar+phi*Pperp
print("  sigma(M) = -M^{-1}: %s"%np.allclose(sM,-np.linalg.inv(M)))
B=np.array([a[i] for i in range(1,9)]).T
A=np.linalg.solve(B,sM@B)
print("  sigma(M) integrale su base radici: max|A-round(A)| = %.2e"%np.max(np.abs(A-np.round(A))))
np.savez('pentagonal_unit_v1.npz',w=w,G=G)
print("salvato pentagonal_unit_v1.npz")
