#!/usr/bin/env python3
"""STAGE B+C - fermionic zero modes in the soliton background, and the overlap |c|.

Builds a 2D lattice Dirac operator (Wilson term against doublers) Yukawa-coupled to the
Stage-A soliton field n(x):   D = sum_mu [gamma_mu central-diff + Wilson] (x) I_internal
                                  + g * I_spinor (x) (n.sigma)
By the index theorem the winding-B background carries ~B low (would-be zero) modes:
the B generations. Stage C forms the generation mass matrix from their overlaps with the
uniform (Higgs) mode and extracts |c| = off-diagonal/diagonal of the Z3 circulant; it
reports the Koide Q and compares |c| with 1/sqrt(2).

  *** PHYSICS NOTE (validate on the cluster) ***  the fermion-soliton coupling (here the
  standard Yukawa g n.sigma to an internal doublet) is the model-sensitive choice flagged
  in the paper (Remark on Stage 6). The continuum-extrapolated |c| (Stage D) is the result.

Usage:  python3 stage_BC_overlap.py --soliton soliton_L96_B3.npy --g 1.0 --nmodes 8
"""
import argparse, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

s1=np.array([[0,1],[1,0]],complex); s2=np.array([[0,-1j],[1j,0]],complex); s3=np.array([[1,0],[0,-1]],complex)
I2=np.eye(2,dtype=complex)
# 2D Euclidean gamma = (s1, s2); chirality gamma5 = s3 (spinor space)

def build_dirac(n, g, r_w=1.0):
    """Vectorised assembly (identical to the naive loop, ~20x faster at L=128, and the only
    feasible route at large L). dof = 2 spinor x 2 internal = 4. 5-point stencil:
    per-site Yukawa+Wilson diagonal block, constant hopping blocks."""
    L=n.shape[0]; N=L*L; dof=4
    Gx=np.kron(s1,I2); Gy=np.kron(s2,I2); W=np.kron(I2,I2)
    ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij'); s=(ii*L+jj).ravel()
    pp,qq=np.meshgrid(np.arange(dof),np.arange(dof),indexing='ij'); pp=pp.ravel(); qq=qq.ravel()
    R=[]; C=[]; V=[]
    # per-site diagonal block: g * I2(x)(n.sigma) + 2 r_w I4
    nsig=(n[...,0,None,None]*s1+n[...,1,None,None]*s2+n[...,2,None,None]*s3).reshape(N,2,2)
    Mdiag=g*np.einsum('ab,scd->sacbd',I2,nsig).reshape(N,4,4)+2*r_w*np.eye(4)[None]
    R.append((s[:,None]*dof+pp[None,:]).ravel()); C.append((s[:,None]*dof+qq[None,:]).ravel())
    V.append(Mdiag[:,pp,qq].ravel())
    # constant hopping blocks to the 4 neighbours
    for di,dj,G in [(1,0,Gx),(-1,0,Gx),(0,1,Gy),(0,-1,Gy)]:
        sgn=1 if (di+dj)>0 else -1; B=-0.5*(r_w*W-sgn*G)
        nb=(((ii+di)%L)*L+((jj+dj)%L)).ravel()
        for (p,q) in np.argwhere(B!=0):
            R.append(s*dof+p); C.append(nb*dof+q); V.append(np.full(N,B[p,q]))
    return sp.csr_matrix((np.concatenate(V),(np.concatenate(R),np.concatenate(C))),shape=(N*dof,N*dof))

def lowest_modes(D,k):
    # lowest singular values of D via the lowest eigenvalues of H = D^dag D.
    # Shift-invert just ABOVE zero: with genuine zero modes H is near-singular and sigma=0 is
    # fragile/slow; sigma=1e-6 keeps (H - sigma I) well-conditioned yet still targets modes near 0.
    H=(D.getH()@D)
    try:
        w,v=spla.eigsh(H,k=k,sigma=1e-6,which='LM',maxiter=5000)
    except Exception as ex:
        print("  shift-invert failed (%s); fallback which='SA'"%type(ex).__name__,flush=True)
        w,v=spla.eigsh(H,k=k,which='SA',maxiter=20000)
    order=np.argsort(w); return np.sqrt(np.abs(w[order])), v[:,order]

def overlaps_to_c(modes, n, g, L, B):
    """Complex Yukawa mass matrix M_ij = <psi_i| Y |psi_j>, Y = g (I_spinor (x) n.sigma).
    This keeps the phase (hence the Z3/circulant structure), unlike a |psi|^2 density Gram.
    |c| = mean|off-diagonal| / mean|diagonal| of the B-generation block."""
    dof=4; N=L*L
    nsig=(n[...,0,None,None]*s1+n[...,1,None,None]*s2+n[...,2,None,None]*s3).reshape(N,2,2)
    Y=g*np.einsum('ab,scd->sacbd',I2,nsig).reshape(N,4,4)           # per-site 4x4 Yukawa block
    psis=[modes[:,m].reshape(N,dof) for m in range(B)]
    M=np.zeros((B,B),complex)
    for i in range(B):
        for j in range(B):
            M[i,j]=np.sum(np.conj(psis[i])[:,:,None]*Y*psis[j][:,None,:])
    Md=np.abs(np.diag(M)).mean(); Mo=np.abs(M[~np.eye(B,dtype=bool)]).mean()
    return Mo/Md, M

def c_density_gram(modes, L, B):
    """Diagnostic: |c| from the |psi|^2 density Gram (phase-blind). Brackets the mass-matrix
    value from above. Reported only to show that Wilson modes do not pin |c|."""
    N=L*L; P=[]
    for m in range(B):
        d=np.sum(np.abs(modes[:,m].reshape(N,4))**2,1); P.append(d/np.sqrt(np.sum(d**2)))
    P=np.array(P); G=P@P.T
    return np.abs(G[~np.eye(B,dtype=bool)]).mean()/np.abs(np.diag(G)).mean()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--soliton",required=True); ap.add_argument("--g",type=float,default=1.0)
    ap.add_argument("--nmodes",type=int,default=8); ap.add_argument("--B",type=int,default=3)
    a=ap.parse_args()
    n=np.load(a.soliton); L=n.shape[0]
    print(f"loaded {a.soliton}  L={L}")
    D=build_dirac(n,a.g); print("Dirac built:",D.shape,"nnz",D.nnz)
    vals,modes=lowest_modes(D,a.nmodes)
    print("lowest |eig|:",np.round(vals,4))
    c,M=overlaps_to_c(modes,n,a.g,L,a.B)
    c_gram=c_density_gram(modes,L,a.B)
    print("Yukawa mass matrix M_ij (complex):\n",np.round(M,3))
    print(f"|c| (mass matrix)   = {c:.4f}")
    print(f"|c| (density Gram)  = {c_gram:.4f}   <- phase-blind upper bracket")
    print(f"target 1/sqrt2      = {1/np.sqrt(2):.4f}")
    print("NOTE: Wilson modes are massive (no exact zero modes) -> the two proxies bracket |c|")
    print("      widely; a genuine value needs an overlap/Ginsparg-Wilson Dirac (Stage 6).")
