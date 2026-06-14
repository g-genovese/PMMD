"""PMMD - explicit foam-path holonomy for the PMNS transport charges (2026-06-13).
Goal: realise q=(0,+1,0,-2) as the holonomy of a closed loop on the foam graph,
not as an assignment. The loop is the order-5 orbit of g=w^{-6} (pentagonal unit).
"""
import numpy as np
np.set_printoptions(suppress=True,linewidth=160,precision=5)
phi=(1+np.sqrt(5))/2
e=np.eye(8)
a={1:0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),2:e[0]+e[1],3:e[1]-e[0],
   4:e[2]-e[1],5:e[3]-e[2],6:e[4]-e[3],7:e[5]-e[4],8:e[6]-e[5]}
def refl(v): return np.eye(8)-2*np.outer(v,v)/(v@v)
w=np.eye(8)
for i in [8,7,6,5,4,3,1,2]: w=w@refl(a[i])
g=np.linalg.matrix_power(w,30-6)          # g = w^{-6}, ordine 5
print("[1] g=w^-6: ordine5=%s det=%.4f integrale(base radici)=%.1e"%(
    np.allclose(np.linalg.matrix_power(g,5),np.eye(8)), np.linalg.det(g),
    np.max(np.abs(np.linalg.solve(np.array([a[i] for i in range(1,9)]).T, g@np.array([a[i] for i in range(1,9)]).T)-np.round(np.linalg.solve(np.array([a[i] for i in range(1,9)]).T, g@np.array([a[i] for i in range(1,9)]).T))))))

# ---- costruisci le 240 radici ----
R=[]
for i in range(8):
    for j in range(i+1,8):
        for si in(1,-1):
            for sj in(1,-1): R.append(si*e[i]+sj*e[j])
for k in range(256):
    s=[1 if (k>>b)&1 else -1 for b in range(8)]
    if np.prod(s)==1: R.append(0.5*np.array(s,float))
R=np.array([r for r in R if abs(r@r-2)<1e-9])
print("    radici:",len(R))
# g come permutazione
idx={tuple(np.round(r,6)):n for n,r in enumerate(R)}
perm=np.array([idx[tuple(np.round(g@r,6))] for r in R])
# struttura orbite
seen=set(); orb_sizes=[]
for st in range(len(R)):
    if st in seen: continue
    o=[]; x=st
    while x not in seen: seen.add(x); o.append(x); x=perm[x]
    orb_sizes.append(len(o))
import collections
print("[2] orbite di g sulle 240 radici:",dict(collections.Counter(orb_sizes)))

# ---- cerca un pentagono nel GRAFO foam (spigoli = inner product +1, 60 gradi) ----
found=None
for st in range(len(R)):
    o=[st]; x=st
    for _ in range(4): x=perm[x]; o.append(x)
    if perm[o[-1]]!=st: continue
    ips=[R[o[k]]@R[o[(k+1)%5]] for k in range(5)]
    if found is None: found=(o,ips)
    if all(abs(ip-1)<1e-9 for ip in ips):
        found=(o,ips); break
o,ips=found
print("[3] orbita-5 campione, inner product consecutivi:",np.round(ips,4),
      "  pentagono-60deg:%s"%all(abs(ip-1)<1e-9 for ip in ips))
print("    (se non 60deg: il loop vive nel grafo dei quasi-vicini; struttura comunque chiusa a 5)")

# ---- [4] trasporto del PIANO DI FAMIGLIA F lungo l'orbita {g^k F}: olonomia di Berry non-abeliana ----
F=np.zeros((8,3)); F[0,0]=F[1,1]=F[2,2]=1.0          # base ortonormale di F (coords 0,1,2)
frames=[F]
for k in range(1,5): frames.append(g@frames[-1])
# trasporto discreto: U_k = frame_{k+1}^T frame_k, poi raddrizza (polar) -> SU(3)
def polar_u(A):
    U,_,Vt=np.linalg.svd(A); return U@Vt
H=np.eye(3)
for k in range(5):
    Fk=frames[k]; Fk1=frames[(k+1)%5]
    M=Fk1.T@Fk                                       # 3x3 overlap
    H=polar_u(M)@H
evH=np.linalg.eigvals(H)
phases=np.sort(np.round(np.degrees(np.angle(evH))%360,3))
print("[4] olonomia di Berry del 3-piano di famiglia attorno al loop pentagonale:")
print("    autofasi (gradi):",phases,"  in unita' di 72deg:",np.round(np.angle(evH)/(2*np.pi/5)%5,3))

# ---- [5] cariche di coniugazione dei 4 canali sotto il trasporto di famiglia R ----
# R = parte unitaria del blocco di famiglia di g (come ruota il frame di famiglia in un passo)
Rblock=g[:3,:3]; UR=polar_u(Rblock)
evR,VR=np.linalg.eig(UR)
phR=np.angle(evR)
nhat=np.array([1,1,1.])/np.sqrt(3); v=np.array([2,-1,-1.])/np.sqrt(6)
ops={'1':np.eye(3),'J=nn^T':np.outer(nhat,nhat),'diag(v)':np.diag(v),'v(x)v':np.outer(v,v)}
print("[5] autofasi del trasporto di famiglia UR (in 72deg):",np.round(phR/(2*np.pi/5)%5,3))
print("    carica dominante per canale (coniugazione  O -> UR O UR^dag):")
for nm,O in ops.items():
    Ot=VR.conj().T@O@VR                              # nel frame che diagonalizza UR
    # elemento (j,k) porta carica (phR[j]-phR[k]) / (72deg)
    best=None
    for j in range(3):
        for kk in range(3):
            ch=((phR[j]-phR[kk])/(2*np.pi/5))
            mag=abs(Ot[j,kk])
            if mag>1e-6 and (best is None or mag>best[0]): best=(mag,round(ch)%5,ch)
    q=best[1]; q=q-5 if q>2 else q
    print("    %-10s carica = %+d   (peso %.3f, raw %.3f)"%(nm,q,best[0],best[2]))
np.savez('foam_path_holonomy_v1.npz',w=w,g=g,perm=perm,H=H,UR=UR)
print("salvato foam_path_holonomy_v1.npz")

# ============================================================
# CONCLUSIONE ONESTA (sessione 2026-06-13):
#  POSITIVO: g=w^{-6} ha 48 orbite-5 pulite sulle 240 radici; ogni orbita e' un
#    PENTAGONO esplicito di radici (inner product consecutivi = -1, angolo 120 deg).
#    Questo E' il cammino chiuso esplicito sul foam. Il quanto e il MENU di cariche
#    {0,+-1,+-2} (|q|=1 su E_par, |q|=2 su E_perp) sono dimostrati (Prop. pentagonale).
#  NEGATIVO/APERTO: l'olonomia di Berry NAIVE del piano di famiglia attorno al loop
#    da' fasi (105.24, 180, 254.76) deg = NON multipli puliti di 72 deg, e le cariche
#    di coniugazione naive NON riproducono (0,+1,0,-2). Motivo strutturale: il piano
#    A2 di famiglia giace a un angolo GENERICO rispetto allo split E_par/E_perp
#    (|P_par n|^2=(15+2v5)/30, |P_par v|^2=(15+v5)/30: nessuno e' 0 o 1), quindi le
#    cariche per-canale NON sono semplici olonomie geometriche del piano di famiglia.
#  => L'assegnazione per-canale resta Stratum 2 (struttura Galois+ombre), e
#     l'olonomia di foam-path esplicita che la realizza resta il bersaglio Stratum-3.
# ============================================================
