"""PMMD - pentagonal holonomy on the flavour bundle V112 (session 2026-06-13).
V112 = degree-3 harmonics on the 240 E8 roots (the collective flavour carrier).
Act with g=w^{-6} (icosahedral Z5) and c2=A2-Coxeter (family Z3); read charges.
"""
import numpy as np, itertools, collections
np.set_printoptions(suppress=True,linewidth=160,precision=4)
phi=(1+np.sqrt(5))/2
e=np.eye(8)
a={1:0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),2:e[0]+e[1],3:e[1]-e[0],
   4:e[2]-e[1],5:e[3]-e[2],6:e[4]-e[3],7:e[5]-e[4],8:e[6]-e[5]}
def refl(v): return np.eye(8)-2*np.outer(v,v)/(v@v)
# radici
R=[]
for i in range(8):
    for j in range(i+1,8):
        for si in(1,-1):
            for sj in(1,-1): R.append(si*e[i]+sj*e[j])
for k in range(256):
    s=[1 if (k>>b)&1 else -1 for b in range(8)]
    if np.prod(s)==1: R.append(0.5*np.array(s,float))
R=np.array([r for r in R if abs(r@r-2)<1e-9]); N=len(R)
idx={tuple(np.round(r,6)):n for n,r in enumerate(R)}
def perm_of(M): return np.array([idx[tuple(np.round(M@r,6))] for r in R])
def Pmat(perm):
    P=np.zeros((N,N))
    for i,j in enumerate(perm): P[j,i]=1.0
    return P
w=np.eye(8)
for i in [8,7,6,5,4,3,1,2]: w=w@refl(a[i])
g=np.linalg.matrix_power(w,30-6)          # icosahedral, order 5
c2=refl(a[3])@refl(a[4])                   # family A2-Coxeter, order 3
Pg=Pmat(perm_of(g)); Pc2=Pmat(perm_of(c2))
# === costruisci V112 ===
def monos(deg):
    cols=[np.ones(N)] if deg==0 else [np.prod(R[:,list(c)],axis=1) for c in itertools.combinations_with_replacement(range(8),deg)]
    return np.array(cols).T
def onb(X,tol=1e-8):
    U,S,_=np.linalg.svd(X,full_matrices=False); return U[:,S>tol]
D2=np.hstack([monos(0),monos(1),monos(2)]); D3=np.hstack([D2,monos(3)])
Q2=onb(D2); Q3=onb(D3)
QV=onb(Q3-Q2@(Q2.T@Q3))
print("dim D2=%d  dim D3=%d  dim V112=%d"%(Q2.shape[1],Q3.shape[1],QV.shape[1]))
gV=QV.T@Pg@QV; c2V=QV.T@Pc2@QV
# === spettri di carica ===
evg=np.linalg.eigvals(gV); kg=np.round(np.angle(evg)/(2*np.pi/5)).astype(int)%5
kg=[(k if k<=2 else k-5) for k in kg]
print("g|V112  (Z5) spettro cariche:",dict(sorted(collections.Counter(kg).items())))
evc=np.linalg.eigvals(c2V); jc=np.round(np.angle(evc)/(2*np.pi/3)).astype(int)%3
print("c2|V112 (Z3) spettro cariche:",dict(sorted(collections.Counter(jc).items())),"  (atteso 58/27/27)")
print("commutano? ||[c2,g]||_V112 = %.3e"%np.linalg.norm(c2V@gV-gV@c2V))
# === il gruppo generato: <g,c2> contiene una simmetria di famiglia compatibile? ===
# cerca elemento di ordine 5 che PRESERVA il piano di famiglia (coords 0,1,2)
P012=np.diag([1,1,1,0,0,0,0,0])
print("g preserva il 3-spazio famiglia? ||(1-P)gP|| = %.3f"%np.linalg.norm((np.eye(8)-P012)@g@P012))
# === lift della tripletta di famiglia in V112 e loro carica g ===
w3=np.exp(2j*np.pi/3)
modes={'mode0(n)':np.array([1,1,1,0,0,0,0,0])/np.sqrt(3),
       'mode1':np.array([1,w3,w3**2,0,0,0,0,0])/np.sqrt(3),
       'mode2':np.array([1,w3**2,w3,0,0,0,0,0])/np.sqrt(3)}
def lift_cubic(u):                          # P_{V112}[(u·r)^3]
    f=(R@u)**3; c=QV.T@f; return c/ (np.linalg.norm(c)+1e-30)
print("\n--- cariche g dei lift cubici di famiglia (3x3 azione proiettata) ---")
H=np.array([lift_cubic(u) for u in modes.values()]).T   # 112 x 3 (complesso)
# azione di g sui lift: g·h_u = h_{gu}; matrice 3x3 nella base dei lift
Hg=np.array([QV.T@((R@(g@u))**3) for u in modes.values()]).T
Hg=Hg/ (np.linalg.norm(Hg,axis=0)+1e-30)
# proietta g nello span dei 3 lift
S=H.conj().T@H                              # gram
G3=np.linalg.solve(S, H.conj().T@Hg)        # azione 3x3
evG=np.linalg.eigvals(G3)
print("autofasi azione-g sui lift (in 72deg):",np.round(np.angle(evG)/(2*np.pi/5),3),
      "  |modulo|:",np.round(np.abs(evG),3))
print("(se |modulo|<1: i lift NON sono g-chiusi -> tumbling generico, come il piano nudo)")
# === DOVE vive la carica: proietta i lift sugli autospazi-carica di g|V112 ===
wg,Vg=np.linalg.eig(gV)
chg=np.array([ (lambda k:(k if k<=2 else k-5))(int(np.round(np.angle(x)/(2*np.pi/5)))%5) for x in wg])
print("\n--- distribuzione di carica g dei lift cubici su V112 ---")
for nm,u in modes.items():
    c=lift_cubic(u); coeff=np.linalg.solve(Vg,c) if np.linalg.matrix_rank(Vg)==Vg.shape[0] else np.linalg.lstsq(Vg,c,rcond=None)[0]
    wsum={}
    for q,amp in zip(chg,np.abs(coeff)**2):
        wsum[q]=wsum.get(q,0)+amp
    tot=sum(wsum.values())
    print("  %-10s peso per carica:"%nm,{q:round(v/tot,3) for q,v in sorted(wsum.items())})
np.savez('v112_bundle_holonomy_v1.npz',QV=QV,gV=gV,c2V=c2V)
print("\nsalvato v112_bundle_holonomy_v1.npz")
