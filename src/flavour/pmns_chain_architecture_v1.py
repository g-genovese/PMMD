"""PMMD - PMNS sector from the chain: closed architecture v1 (session 2026-06-12).

*** CORRECTIONS LEDGER (added 2026-06-12, later session) ***
(1) The claim below "succeeds at cost ~0.23 with ... Sigma ~ 0.056-0.066+" was WRONG as stated:
    at Sigma<=0.066 the cost 0.23 was entirely a HALVED Dm2_31 (1.29e-3 vs 2.511e-3), hidden
    because Dm2 was not printed. The exact-oscillation manifold of the FLAT-Dirac architecture
    has Sigma floor ~0.092-0.095 eV (disfavoured by DESI-class bounds).
(2) "Hierarchical variants numerically EXCLUDED" held only for the Z3-symmetric core. With the
    Z2(Weyl) core, the CONJUGATE branch (beta_nu = -2/9, cosine level) fits all oscillation
    data exactly with Sigma floor 0.060-0.064 eV (DESI-compatible) and is the selected branch.
(3) The delta window "[60,120] U [240,300]" was computed on the compromised family of (1).
*** SUPERSEDED BY architecture v3: see grid_phase_closure.npz, v3_pinned_closure.npz,
    v3_costx_curve.npz, conj_branch_sigma_floor.npz and the session transcript. v3:
    conjugate Dirac (beta_nu=-2/9, cos level) x Z2 core with operator phases
    (0, +2pi/5, x, -4pi/5+2x), x = arg<Phi> = beta_nu*eps^2 = -(2/9)phi^-4, dressing
    eps=phi^-2, alpha=+2pi/3; four real moduli; all oscillation observables <=0.5 sigma;
    OUTPUTS Sigma=0.0648 eV, delta_CP=299.9 deg (PDG convention, audited), m_bb=0.18 meV. ***
Ingredients (all chain-named):
 1. Charged sector: family-Z3 circulant, Fourier U_w, Koide assignment (tau,e,mu)=(k0,k1,k2), beta=2/9.
 2. Neutrino Dirac: FLAT (nu_R = the three modes of the family-plane singleton A2 {0,1,2}; paper line ~8248).
    Hierarchical variants (beta_nu = +-2/9, cos and cos^2 grading) numerically EXCLUDED.
 3. M_R core: Z2(Weyl-reflection)-symmetric (B-L condensate direction on the family plane breaks Z3 -> Z2).
    Z3-symmetric core excluded by T2 (theta23 = 0). Discrete reflection-chirality pairing: p=1 incompatible with alpha=+2pi/3.
 4. Spurion dressing: E = exp(eps e^{i alpha} K), eps = phi^-2 (bivector cut&project grading), alpha = +-2pi/3 (Omega_prim).
    eps=0 control FAILS (cost ~250): the golden spurion is essential.
 5. Seesaw inversion -> light sector. Fit of the Z2-core (4 complex - 1 phase = 7 real) to
    (dm21, dm31, s13, s23^2, s12^2, MSW, Sigma<=edge) succeeds at cost ~0.23 with:
    s13=0.14850, s23^2=0.509, s12^2=0.304, NO ordering emitted, Sigma ~ 0.056-0.066+, m_bb ~ 10 meV,
    delta_CP in ~[60,120] U CP-conj [240,300] along the single residual flat direction
    = condensate angle within its reflection class (the case-(G) orientation datum) -> LAST open number.
Earlier minimal model (direct dressing, Z3 core): hits theta13 (C=1.017), s12^2=0.294, delta=215.5;
fails theta23/MSW -> superseded by this architecture; kept as documented intermediate stage.
"""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.linalg import expm, svd
from scipy.optimize import least_squares
phi=(1+np.sqrt(5))/2; w=np.exp(2j*np.pi/3)
K=np.array([[0,1,0],[0,0,1],[1,0,0]],float)
PI=np.array([[1,0,0],[0,0,1],[0,1,0]],float)
DM21,DM31=7.41e-5,2.511e-3
def takagi(M):
    Uu,S,_=svd(M); Q=Uu.conj().T@M@Uu.conj()
    return Uu@np.diag(np.exp(0.5j*np.angle(np.diag(Q)))),S
def _mk(a,b):
    E=np.zeros((3,3),complex); E[a,b]=E[b,a]=1; return E
def z2_basis(p=0):
    s=PI@np.diag([1,w**p,w**(2*p)])
    idx=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    A=np.zeros((9,6),complex)
    for j,(a,b) in enumerate(idx): A[:,j]=(s.T@_mk(a,b)@s-_mk(a,b)).ravel()
    _,sv,Vh=svd(A); null=Vh.conj().T[:,sv<1e-10]
    return [sum(null[j,c]*_mk(*idx[j]) for j in range(6)) for c in range(null.shape[1])]
def full_obs(Mt):
    V,S=takagi(Mt); Up=np.conj(V)
    o=np.argsort(S); i3=o[-1]; pa,pb=o[0],o[1]
    i1,i2=(pa,pb) if abs(Up[1,pa])**2>=abs(Up[1,pb])**2 else (pb,pa)
    cols=[i1,i2,i3]; m=[S[c] for c in cols]; Ue,Um=Up[1,cols],Up[2,cols]
    s13=abs(Ue[2]); c13s=1-s13**2
    return dict(s13=s13,s23sq=abs(Um[2])**2/c13s,s12sq=abs(Ue[1])**2/c13s,
        dm21=m[1]**2-m[0]**2,dm31=m[2]**2-m[0]**2,msw=S[i2]>S[i1],m=m,Ue=Ue,Um=Um)
def fit_architecture(p=0,alpha=2*np.pi/3,eps=phi**-2,edge=0.066,starts=60,seed=11):
    rng=np.random.default_rng(seed); B=z2_basis(p); nb=len(B); E=expm(eps*np.exp(1j*alpha)*K)
    def light(c):
        core=sum((c[2*i]+1j*c[2*i+1])*B[i] for i in range(nb))
        return -PI@np.linalg.inv(E@core@E.T)@PI
    def resid(c):
        try:
            r=full_obs(light(c)); pen=0.0 if r['msw'] else 3.0
            spen=max(0.0,(sum(r['m'])-edge)/0.002)
            out=[np.log(max(abs(r['dm21']),1e-30)/DM21),np.log(max(r['dm31'],1e-30)/DM31),
                 (r['s13']-0.1485)/0.0019,(r['s23sq']-0.51)/0.03,(r['s12sq']-0.304)/0.012,pen,spen]
        except Exception: out=[1e3]*7
        return out+list(1e-3*np.asarray(c))
    best=None
    for t in range(starts):
        c0=rng.normal(scale=1.0,size=2*nb)*np.exp(rng.normal(scale=1.8))
        try:
            s=least_squares(resid,c0,method='trf',max_nfev=1500)
            if best is None or s.cost<best.cost: best=s
        except Exception: pass
    return best,light
if __name__=='__main__':
    for label,eps in [('architettura (eps=phi^-2)',phi**-2),('controllo eps=0',0.0)]:
        b,light=fit_architecture(eps=eps)
        r=full_obs(light(b.x))
        print(f"{label}: cost={b.cost:.3e}  s13={r['s13']:.5f} s23^2={r['s23sq']:.4f} s12^2={r['s12sq']:.4f} MSW={r['msw']} Sigma={sum(r['m']):.4f}")
