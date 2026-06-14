"""PMMD - chain-anchored Coxeter split and icosian shadows (session 2026-06-13).
Targets: (1) microscopic origin of the 2pi/5 transport charges q=(0,+1,0,-2);
(2) the eps^2 = phi^-4 weight in arg<Phi> = beta_nu*eps^2; (3) Clebsch moduli.
Construction anchored to the paper (sec:foam-projection): Coxeter element built
in the case-(G) McKay order (alpha8,...,alpha2), E_par = exponents {1,11,19,29}.
"""
import numpy as np
np.set_printoptions(suppress=True,linewidth=150)
phi=(1+np.sqrt(5))/2
e=np.eye(8)
a={1:0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),2:e[0]+e[1],3:e[1]-e[0],
   4:e[2]-e[1],5:e[3]-e[2],6:e[4]-e[3],7:e[5]-e[4],8:e[6]-e[5]}
C=np.array([[2*(a[i]@a[j])/(a[j]@a[j]) for j in range(1,9)] for i in range(1,9)])
assert abs(np.linalg.det(C)-1)<1e-9, "Cartan det"
def refl(v): return np.eye(8)-2*np.outer(v,v)/(v@v)
order=[8,7,6,5,4,3,1,2]                      # McKay (case-(G))
w=np.eye(8)
for i in order: w=w@refl(a[i])               # w = R8 R7 R6 R5 R4 R3 R1 R2
ev,V=np.linalg.eig(w)
ang=np.angle(ev); ms=np.round(np.abs(ang)*30/(2*np.pi)).astype(int)
print("esponenti trovati:",sorted(set(ms)))
PAR={1,11,19,29}
Ppar=np.zeros((8,8))
planes={}                                     # m -> (f,g) base ortonormale del piano
for m in sorted(set(ms)):
    idx=[k for k in range(8) if ms[k]==m and ang[k]>0][0]
    u=V[:,idx]; f=np.real(u); g=np.imag(u)
    # ortonormalizza (f,g)
    f=f/np.linalg.norm(f); g=g-(g@f)*f; g=g/np.linalg.norm(g)
    P=np.outer(f,f)+np.outer(g,g); planes[m]=(f,g)
    if m in PAR: Ppar+=P
Pperp=np.eye(8)-Ppar
M=phi*Ppar-(1/phi)*Pperp
B=np.array([a[i] for i in range(1,9)]).T
A=np.linalg.solve(B,M@B)
print("inflazione M integrale su base radici: max|A-round(A)| = %.2e ; det M = %.6f"%(np.max(np.abs(A-np.round(A))),np.linalg.det(M)))
# radici: split aureo dei moduli
roots=[]
for i in range(8):
    for j in range(i+1,8):
        for si in(1,-1):
            for sj in(1,-1): roots.append(si*e[i]+sj*e[j])
for k in range(128):
    s=[1 if (k>>b)&1 else -1 for b in range(7)]; s.append(np.prod(s))  # prodotto +1
    if np.prod(s)==1: roots.append(0.5*np.array(s,float))
roots=np.array([r for r in roots if abs(r@r-2)<1e-9])
np2=np.round(np.einsum('ij,ij->i',roots@Ppar,roots),9)
vals,counts=np.unique(np2,return_counts=True)
print("|P_par r|^2 sulle 240 radici:",dict(zip(vals,counts)),"  rapporto=%.6f vs phi^2=%.6f"%(vals.max()/vals.min(),phi**2))
# direzioni di famiglia (coords {0,1,2})
def emb(v3): x=np.zeros(8); x[:3]=v3; return x/np.linalg.norm(x)
nhat=emb([1,1,1]); what=emb([2,-1,-1]); w2=emb([0,1,-1])
names=["n_hat(E6 axis)","w_hat(weight)","w2(2nd plane ax)","alpha5","alpha4(fam)","alpha3(fam)","alpha8(chain st)"]
vecs=[nhat,what,w2,a[5]/np.sqrt(2),a[4]/np.sqrt(2),a[3]/np.sqrt(2),a[8]/np.sqrt(2)]
print("\n--- NORME (Bersagli 2/3): |P_par x|^2, |P_perp x|^2, ratio, fit p+q*sqrt5 ---")
def fit_gold(val):
    best=None
    for d in range(1,61):
        p=np.round(val*d); q=np.round((val-p/d)*d/np.sqrt(5))
        for qq in (q-1,q,q+1):
            cand=(p/d)+ (qq/d)*np.sqrt(5)
            # ricerca diretta p,q
        # ricerca brute p,q in [-3d,3d]
    for d in (1,2,3,4,5,6,8,10,12,15,20,24,30,40,60):
        for p in range(-3*d,3*d+1):
            rem=(val-p/d)/np.sqrt(5)
            q=np.round(rem*d)/d
            if abs(p/d+q*np.sqrt(5)-val)<1e-9 and (best is None): best=(p,q,d)
    return best
for nm,x in zip(names,vecs):
    pp=x@Ppar@x; qq=x@Pperp@x
    fb=fit_gold(pp)
    fs="(%s+%s*sqrt5)/%s"%(fb[0],int(fb[1]*fb[2]),fb[2]) if fb else "?"
    print("%-18s |par|^2=%.9f |perp|^2=%.9f  perp/par=%.6f   par=%s"%(nm,pp,qq,qq/pp,fs))
print("riferimenti: phi^-1=%.6f phi^-2=%.6f phi^-3=%.6f phi^-4=%.6f"%(1/phi,phi**-2,phi**-3,phi**-4))
# angoli pentagonali nei piani m=1 e m=11
print("\n--- ANGOLI (Bersaglio 1): coordinate complesse nei piani H4 ---")
for m in (1,11):
    f,g=planes[m]
    z={nm:( (x@f)+1j*(x@g) ) for nm,x in zip(names,vecs)}
    print("piano m=%d: |z| e arg (gradi):"%m)
    for nm in names:
        print("   %-18s |z|=%.6f  arg=%9.3f"%(nm,abs(z[nm]),np.degrees(np.angle(z[nm]))))
    ref=np.angle(z["alpha8(chain st)"])
    print("   differenze rispetto a alpha8 (mod 36):")
    for nm in names[:-1]:
        d=np.degrees(np.angle(z[nm])-ref)%360
        print("     %-18s d=%9.3f   mod36=%7.3f"%(nm,d,d%36))
np.savez('icosian_shadows_v1.npz',w=w,Ppar=Ppar,roots=roots)
print("\nsalvato icosian_shadows_v1.npz")
