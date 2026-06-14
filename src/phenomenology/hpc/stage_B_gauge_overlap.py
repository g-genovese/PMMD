#!/usr/bin/env python3
"""STAGE B (physical) - fermion zero modes on the baby-skyrmion via the EMERGENT CP^1 GAUGE FIELD.

Key physics (this replaces the Yukawa-to-(n.sigma) coupling, which gives index 0 = no chiral
zero modes): the O(3)/CP^1 soliton carries a composite U(1) gauge field a_mu = -i z^dag d_mu z,
with z the CP^1 spinor (n = z^dag sigma z). The skyrmion is a magnetic flux, flux = 2 pi Q.
A charge-1 Dirac fermion in this background has, by Atiyah-Singer, index = Q -> |Q| chiral zero
modes of one chirality = the |Q| generations. We realise this on the lattice with a gauge-
covariant Wilson-Dirac operator carrying the emergent U(1) links, made exactly chiral via the
overlap (Ginsparg-Wilson) construction. The sign function is dense here (small/medium L); for
L >~ 128 and 3D it must become matrix-free (Chebyshev/Zolotarev) -> the HPC Stage 6 on Server C.

Outputs: the zero-mode count and chirality (must match Q), and the Koide ratio Q_K = sum m /
(sum sqrt m)^2 of the zero-mode mass matrix under a Higgs profile (Q_K = 2/3 <=> |c| = 1/sqrt2).

  usage: python3 stage_B_gauge_overlap.py --soliton sol.npy --m0 1.0
"""
import argparse, numpy as np, scipy.sparse as sp, scipy.linalg as sla
s1=np.array([[0,1],[1,0]],complex); s2=np.array([[0,-1j],[1j,0]],complex)
s3=np.array([[1,0],[0,-1]],complex); I2=np.eye(2,dtype=complex)

def berg_luscher_Q(n):
    n=np.array(n); n=n/np.linalg.norm(n,axis=-1,keepdims=True)
    a=n[:-1,:-1]; b=n[1:,:-1]; c=n[1:,1:]; d=n[:-1,1:]
    t=lambda p,q,r:2*np.arctan2(np.sum(p*np.cross(q,r),-1),1+np.sum(p*q,-1)+np.sum(q*r,-1)+np.sum(r*p,-1))
    return float((t(a,b,c).sum()+t(a,c,d).sum())/(4*np.pi))

def cp1_links(n):
    """z from n (robust per-site patch to avoid the poles), then U(1) links U_mu=z^dag(x)z(x+mu)/|.|."""
    L=n.shape[0]; nz=n[...,2]; nx=n[...,0]; ny=n[...,1]; z=np.zeros((L,L,2),complex)
    up=nz>=0
    z[up,0]=np.sqrt((1+nz[up])/2); z[up,1]=(nx[up]+1j*ny[up])/np.sqrt(2*(1+nz[up]))
    dn=~up
    z[dn,0]=(nx[dn]-1j*ny[dn])/np.sqrt(2*(1-nz[dn])); z[dn,1]=np.sqrt((1-nz[dn])/2)
    def link(ax):
        ov=np.sum(np.conj(z)*np.roll(z,-1,axis=ax),axis=-1); return ov/np.abs(ov)
    return link(0), link(1)

def build_DW_gauge(Ux,Uy,r_w=1.0,m0=1.0):
    """Gauge-covariant Wilson-Dirac (charge-1) with the emergent U(1) links. dof = 2 (spinor)."""
    L=Ux.shape[0]; N=L*L; dof=2
    ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij'); s=(ii*L+jj).ravel()
    R=[]; C=[]; V=[]
    for p in range(2): R.append(s*dof+p); C.append(s*dof+p); V.append(np.full(N,2*r_w-m0,complex))
    for di,dj,G,U in [(1,0,s1,Ux.ravel()),(0,1,s2,Uy.ravel())]:
        nbf=(((ii+di)%L)*L+((jj+dj)%L)).ravel(); Bf=-0.5*(r_w*I2-G)         # x -> x+mu, link U(x)
        for p in range(2):
            for q in range(2):
                if Bf[p,q]!=0: R.append(s*dof+p); C.append(nbf*dof+q); V.append(Bf[p,q]*U)
        Uback=np.roll(np.conj(U.reshape(L,L)),(di,dj),axis=(0,1)).ravel()   # conj U at x-mu
        nbb=(((ii-di)%L)*L+((jj-dj)%L)).ravel(); Bb=-0.5*(r_w*I2+G)
        for p in range(2):
            for q in range(2):
                if Bb[p,q]!=0: R.append(s*dof+p); C.append(nbb*dof+q); V.append(Bb[p,q]*Uback)
    return sp.csr_matrix((np.concatenate(V),(np.concatenate(R),np.concatenate(C))),shape=(N*dof,N*dof))

def overlap_zero_modes(Ux,Uy,m0,nzero):
    """Overlap Dirac D_ov = 1 + g5 sign(g5 (D_W - m0)); return its lowest modes and g5 (dense sign)."""
    N=Ux.shape[0]**2; G5=sp.kron(sp.eye(N),s3).tocsr()
    DW=build_DW_gauge(Ux,Uy,m0=m0); Hw=(G5@DW).toarray(); Hw=0.5*(Hw+Hw.conj().T)
    w,U=sla.eigh(Hw); S=(U*np.sign(w))@U.conj().T
    Dov=np.eye(N*2)+G5.toarray()@S
    wv,Vr=np.linalg.eig(Dov); o=np.argsort(np.abs(wv))[:max(nzero,6)]
    return wv[o], Vr[:,o], G5.toarray()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--soliton",required=True); ap.add_argument("--m0",type=float,default=1.0)
    a=ap.parse_args()
    n=np.load(a.soliton); L=n.shape[0]; Q=berg_luscher_Q(n); nz=int(round(abs(Q)))
    print(f"loaded {a.soliton}  L={L}  Berg-Luscher Q={Q:.3f}  -> expect {nz} chiral zero modes")
    if L*L*2>40000: print(f"  WARNING: dense sign() is heavy at this L (dim {L*L*2}); use matrix-free for L>~128.")
    Ux,Uy=cp1_links(n)
    ev,V,G5=overlap_zero_modes(Ux,Uy,a.m0,nz)
    print("lowest |Dov eig|:",np.round(np.abs(ev),4))
    Z,_=np.linalg.qr(V[:,:nz]); g5p=Z.conj().T@G5@Z; chi=np.linalg.eigvalsh(0.5*(g5p+g5p.conj().T))
    print(f"zero-mode chirality (projected g5): {np.round(chi,2)}  index = {chi.sum():+.2f}  (target {-Q:+.0f} up to convention)")
    # Koide test: mass matrix M_ij = <psi_i| h |psi_j>, h = (1 - n_z) Higgs/coherent profile
    h=(1-n[...,2]).ravel(); Hh=sp.kron(sp.diags(h),np.eye(2)).tocsr()
    M=0.5*(Z.conj().T@(Hh@Z)+(Z.conj().T@(Hh@Z)).conj().T); lam=np.sort(np.abs(np.linalg.eigvalsh(M)))
    QK=lam.sum()/(np.sqrt(lam).sum()**2)
    print("zero-mode masses (Higgs profile):",np.round(lam,4))
    print(f"Koide Q_K = {QK:.4f}   (target 2/3 = {2/3:.4f};  Q_K=2/3 <=> |c|=1/sqrt2)")
    print("NOTE: generation COUNT and chirality are topological (index=Q, robust). The Koide VALUE")
    print("      depends on the mass operator; the naive Higgs profile is only one probe.")
