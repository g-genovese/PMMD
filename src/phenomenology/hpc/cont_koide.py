import numpy as np, scipy.sparse as sp
from stage_A_soliton import relax
from stage_B_gauge_overlap import cp1_links, overlap_zero_modes, berg_luscher_Q
box=8.0; pts=[]
for L in [28,40,48,56,64]:
    best,info=relax(L,3,1.0,0.3,15000,0.01,box,400)
    if best is None: print(f"L={L}: soliton collapsed, skip"); continue
    n=best; Q=info[5]; Ux,Uy=cp1_links(n)
    ev,V,G5=overlap_zero_modes(Ux,Uy,1.0,3)
    nz_count=int(np.sum(np.abs(ev)<1e-2))
    Z,_=np.linalg.qr(V[:,:3]); h=(1-n[...,2]).ravel()
    Hh=sp.kron(sp.diags(h),np.eye(2)).tocsr()
    M=Z.conj().T@(Hh@Z); M=0.5*(M+M.conj().T); lam=np.sort(np.abs(np.linalg.eigvalsh(M)))
    QK=lam.sum()/(np.sqrt(lam).sum()**2); a=2*box/L
    pts.append((a,QK)); print(f"L={L} a={a:.3f} Q={Q:.2f} zero-modes={nz_count} Q_K={QK:.4f}")
pts=np.array(pts)
if len(pts)>=3:
    A=np.c_[np.ones(len(pts)),pts[:,0]**2]; (q0,k),*_=np.linalg.lstsq(A,pts[:,1],rcond=None)
    print(f"\ncontinuum Q_K0 = {q0:.4f}  (+ {k:+.3f} a^2)   target 2/3 = {2/3:.4f}")
