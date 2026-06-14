#!/usr/bin/env python3
"""(A) Does the 112-mode eigenspace of the foam Laplacian carry the Z3 structure
of V112 = 58 + 27 + 27 (Higgs-fixed + two E6 matter generations)? (PMMD v6.0)"""
import itertools, numpy as np
pi=np.pi
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
def coxeter():
    e=np.eye(8)
    simple=[0.5*(e[0]-e[1]-e[2]-e[3]-e[4]-e[5]-e[6]+e[7]),e[0]+e[1],e[1]-e[0],
            e[2]-e[1],e[3]-e[2],e[4]-e[3],e[5]-e[4],e[6]-e[5]]
    def refl(al):
        M=np.eye(8)
        for k in range(8): M[:,k]=e[k]-2*(e[k]@al)/(al@al)*al
        return M
    w=np.eye(8)
    for al in simple: w=refl(al)@w
    return w

R=e8_roots(); n=len(R); G=R@R.T
A=(np.abs(G-1)<1e-9).astype(float); L=np.diag(A.sum(1))-A
val,vec=np.linalg.eigh(L)
# the 112-eigenspace: eigenvalue ~58
mask=np.abs(val-58)<1e-3; V=vec[:,mask]
print(f"112-eigenspace dimension found: {V.shape[1]} (expect 112)")

w=coxeter(); w3=np.linalg.matrix_power(w,10)
print(f"Coxeter^10 order-3 check  ||w3^3 - I|| = {np.linalg.norm(w3@w3@w3-np.eye(8)):.2e}")
# permutation of the 240 roots by w3
idx=np.zeros(n,int)
for j in range(n):
    rj=w3@R[j]; k=np.argmin(((R-rj)**2).sum(1)); idx[j]=k
Pi=np.zeros((n,n)); Pi[idx,np.arange(n)]=1.0
print(f"valid root permutation: {np.allclose(Pi.sum(0),1) and np.allclose(Pi.sum(1),1)}")

M=V.T@Pi@V                      # order-3 action restricted to the 112-eigenspace
tr=np.trace(M)
n1=(112+2*tr)/3; no=(112-tr)/3  # Z3 multiplicities (mult(w)=mult(w^2)=no)
print(f"\ncharacter tr(M) on 112-eigenspace = {tr:.3f}")
print(f"Z3 decomposition of the 112-mode:")
print(f"   eigenvalue +1  (Z3-fixed): {n1:.2f}")
print(f"   eigenvalue  w           : {no:.2f}")
print(f"   eigenvalue  w^2         : {no:.2f}")
print(f"\n   V112 prediction: 58 + 27 + 27  (tr would be 58-27 = 31)")
print(f"   match 58+27+27 ? {abs(n1-58)<1.5 and abs(no-27)<1.5}")
# also report the actual eigenvalues of M
ev=np.linalg.eigvals(M); ph=np.round(np.angle(ev)/(2*pi)*3)%3
print(f"   actual multiplicities from eig: +1:{np.sum(np.abs(ph)<.1)}  w:{np.sum(np.abs(ph-1)<.1)}  w^2:{np.sum(np.abs(ph-2)<.1)}")
