"""PMMD case-(G) -> PMNS phase structure (session 2026-06-12, part 2).

*** STATUS NOTE (added later same day): the phase analyses in this script (single-psi scan,
    psi=-9pi/10 minimum, free-phase decomposition) were performed on the Sigma<=0.066
    COMPROMISED family of the flat-Dirac architecture (Dm2_31 halved there) and are
    SUPERSEDED. The geometry block below (McKay frame det=+1, family-plane shadow sequence,
    reflection-chirality pairing derivation, Hypothesis-(R) exclusion) STANDS.
    Final structure: architecture v3 (conjugate Dirac branch, grid phases (0,+2pi/5,x,-4pi/5+2x),
    x = beta_nu*eps^2): see v3_pinned_closure.npz / v3_costx_curve.npz. ***
GEOMETRY (derived, see transcript):
  - McKay canonical frame (alpha8,alpha7,alpha6,alpha5,alpha4,alpha3,alpha1,alpha2): det = +1 (case-(G) sign, Stratum-3 item closed).
  - Family-plane shadow sequence: first three steps OFF-plane; entry at alpha5 along WEIGHT axis v=(1,1,-2)/sqrt6
    => preserved transposition (01) [mode class p=1], plane sense CLOCKWISE => alpha_eff = -2pi/3.
    This DERIVES the reflection-chirality pairing found empirically (p=1 <-> alpha=-120).
  - Hypothesis (R) (all-real core, phases only from dressing): EXCLUDED (cost 15-25, delta->0).
  - Operator decomposition of the Z2 core (B-L forbids bare term):
      core = c0*Id + c1*J + c2*diag(v) + c3*v(x)v,  phases = transport per condensate insertion.
  - Single-psi ansatz (0,0,psi,2psi): mixing exact, Dm2 off by ~2; minimum at psi=-162 = -9pi/10 (icosian grid).
  - This script: free complex c_i (= full Z2 core, 7 physical) -> exact fit; decompose phases; read delta_CP.
"""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.linalg import expm, svd
from scipy.optimize import least_squares
rng=np.random.default_rng(57)
phi=(1+np.sqrt(5))/2; EPS=phi**-2
w=np.exp(2j*np.pi/3)
U=np.array([[1,1,1],[1,w,w*w],[1,w*w,w]],dtype=complex)/np.sqrt(3)
K=np.array([[0,1,0],[0,0,1],[1,0,0]],float)
PI=np.array([[1,0,0],[0,0,1],[0,1,0]],float)
DM21,DM31=7.41e-5,2.511e-3
v=np.array([1,1,-2])/np.sqrt(6)
OPS=[np.eye(3),np.ones((3,3))/3,np.diag(v),np.outer(v,v)]
E=expm(EPS*np.exp(-2j*np.pi/3)*K)
def takagi(M):
    Uu,S,_=svd(M); Q=Uu.conj().T@M@Uu.conj()
    return Uu@np.diag(np.exp(0.5j*np.angle(np.diag(Q)))),S
def full_obs(Mt):
    V,S=takagi(Mt); Up=np.conj(V)
    o=np.argsort(S); i3=o[-1]; pa,pb=o[0],o[1]
    i1,i2=(pa,pb) if abs(Up[1,pa])**2>=abs(Up[1,pb])**2 else (pb,pa)
    cols=[i1,i2,i3]; m=[S[c] for c in cols]; Ue,Um=Up[1,cols],Up[2,cols]
    s13=abs(Ue[2]); c13s=1-s13**2
    return dict(s13=s13,s23sq=abs(Um[2])**2/c13s,s12sq=abs(Ue[1])**2/c13s,
        dm21=m[1]**2-m[0]**2,dm31=m[2]**2-m[0]**2,msw=S[i2]>S[i1],m=m,Ue=Ue,Um=Um)
def dcp_of(r):
    t12,t13,t23=np.arcsin(np.sqrt(min(r['s12sq'],1))),np.arcsin(min(r['s13'],1)),np.arcsin(np.sqrt(min(r['s23sq'],1)))
    Qm=r['Ue'][0]*r['Um'][2]*np.conj(r['Ue'][2])*np.conj(r['Um'][0])
    s12,c12,s23,c23,s13,c13=np.sin(t12),np.cos(t12),np.sin(t23),np.cos(t23),np.sin(t13),np.cos(t13)
    best,bd=9e9,0
    for d in np.linspace(0,2*np.pi,1441):
        q=(c12*c13)*(s23*c13)*np.conj(s13*np.exp(-1j*d))*np.conj(-s12*c23-c12*s23*s13*np.exp(1j*d))
        vv=abs(np.angle(q/Qm))
        if vv<best: best,bd=vv,d
    return np.degrees(bd)
def light(p):
    c=p[:4]+1j*p[4:8]
    Mfl=sum(c[i]*OPS[i] for i in range(4))
    return -PI@np.linalg.inv(E@(U.T@Mfl@U)@E.T)@PI
W=3.0
def resid(p):
    try:
        r=full_obs(light(p)); pen=0.0 if r['msw'] else 3.0
        spen=max(0.0,(sum(r['m'])-0.066)/0.002)
        return [W*np.log(max(abs(r['dm21']),1e-30)/DM21),W*np.log(max(r['dm31'],1e-30)/DM31),
                (r['s13']-0.1485)/0.0019,(r['s23sq']-0.51)/0.03,(r['s12sq']-0.304)/0.012,pen,spen]
    except Exception: return [1e3]*7
if __name__=='__main__':
    best=None
    for t in range(150):
        p0=np.concatenate([rng.normal(scale=1.0,size=4),rng.normal(scale=1.0,size=4)])*np.exp(rng.normal(scale=2.0))
        try:
            s=least_squares(resid,p0,method='trf',max_nfev=1400)
            if best is None or s.cost<best.cost: best=s
        except Exception: pass
    r=full_obs(light(best.x)); d=dcp_of(r)
    c=best.x[:4]+1j*best.x[4:8]; c=c*np.exp(-1j*np.angle(c[0]))
    mbb=abs(sum(r['Ue'][i]**2*r['m'][i] for i in range(3)))
    print(f"cost={best.cost:.3e}")
    print(f"s13={r['s13']:.5f} s23^2={r['s23sq']:.4f} s12^2={r['s12sq']:.4f} MSW={r['msw']}")
    print(f"Dm21={r['dm21']:.4e} Dm31={r['dm31']:.4e}  Sigma={sum(r['m']):.4f} mbb={mbb:.5f} masse={np.round(r['m'],5)}")
    print(f"delta_CP={d:.1f} deg")
    print("decomposizione (fase globale tolta su c0):")
    for nm,ci in zip(['Id','J','diag(v)','v(x)v'],c):
        print(f"  {nm:8s}: |c|={abs(ci):.5f}  arg={np.degrees(np.angle(ci)):+8.2f} deg")
    p1=np.degrees(np.angle(c[2])); p2=np.degrees(np.angle(c[3])); p0_=np.degrees(np.angle(c[1]))
    print(f"controlli struttura: psi1={p1:.2f} (vs -162=-9pi/10: D={p1+162:+.2f});  (psi2-2psi1)mod360={(p2-2*p1)%360:.2f} (vs 0/120/240);  psi0(J)={p0_:.2f}")
