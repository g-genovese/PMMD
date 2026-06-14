#!/usr/bin/env python3
"""
koide_mass_operator.py
======================
Attack on the open Koide VALUE (the framework's flavour frontier).

Background (Remark rem:gauge-overlap-generations-v6): the 3 generations are the
3 chiral zero modes of a gauge-covariant overlap-Dirac operator in the emergent
U(1) field of a charge-Q=3 CP^1 baby-skyrmion (index theorem, verified). The OPEN
problem is the Koide ratio of the zero-mode masses: a LOCAL HIGGS overlap operator
gives Q_K ~ 0.43-0.45 (between degenerate 1/3 and target 2/3). The paper's own
hypothesis: the correct operator is WINDING-SENSITIVE (rest-mass-as-Berry-winding),
not a local Higgs overlap.

This script:
  1. builds a Q=3 baby-skyrmion CP^1 field on a 2D lattice,
  2. forms the emergent U(1) links U_mu = z^dag(x) z(x+mu)/|...|, verifies flux=3,
  3. builds the 2D Wilson-Dirac operator with those links,
  4. extracts the 3 lowest (near-zero) chiral modes,
  5. computes the zero-mode mass matrix under several candidate mass operators:
       (a) local Higgs overlap  m_k = <psi_k| n_3(x) |psi_k>   [the 0.43 case]
       (b) winding density      m_k = <psi_k| B(x)   |psi_k>   [emergent flux density]
       (c) Berry/winding-rate   m_k = <psi_k| |a(x)|^2 |psi_k>  [connection magnitude]
  6. reports the Koide ratio Q_K = (sum m)/(sum sqrt m)^2 for each, target 2/3.

Honest expectation: this is a proof-of-concept at modest L. The question is whether
ANY winding-sensitive operator moves Q_K from ~0.43 toward 2/3. A positive trend
would support the framework's hypothesis; it will not by itself close the problem.
"""

import numpy as np
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import eigsh
import scipy.linalg as sla

# ---------------- CP^1 baby-skyrmion of winding Q ----------------
def baby_skyrmion(L, Q, w=None):
    """z(x) normalised 2-spinor, winding Q, profile f(r): pi->0."""
    if w is None: w = L/4.0
    cx, cy = L/2.0, L/2.0
    z = np.zeros((L, L, 2), dtype=complex)
    for x in range(L):
        for y in range(L):
            dx, dy = x-cx, y-cy
            r = np.hypot(dx, dy)
            th = np.arctan2(dy, dx)
            f = np.pi*np.exp(-r/w)          # f(0)=pi (north), f(inf)=0 (south)
            z[x,y,0] = np.cos(f/2)
            z[x,y,1] = np.sin(f/2)*np.exp(1j*Q*th)
            z[x,y] /= np.linalg.norm(z[x,y])
    return z

def emergent_links(z):
    """U_mu(x) = z^dag(x) z(x+mu) / |.|  (compact U(1))."""
    L = z.shape[0]
    U = np.zeros((L, L, 2), dtype=complex)   # 2 directions
    for mu, (dx,dy) in enumerate([(1,0),(0,1)]):
        for x in range(L):
            for y in range(L):
                zx = z[x,y]; zn = z[(x+dx)%L,(y+dy)%L]
                ov = np.vdot(zx, zn)
                U[x,y,mu] = ov/abs(ov) if abs(ov)>1e-12 else 1.0
    return U

def plaquette_flux(U):
    """Total winding = (1/2pi) sum of plaquette phases."""
    L = U.shape[0]; tot = 0.0
    for x in range(L):
        for y in range(L):
            p = (U[x,y,0]*U[(x+1)%L,y,1]*
                 np.conj(U[x,(y+1)%L,0])*np.conj(U[x,y,1]))
            tot += np.angle(p)
    return tot/(2*np.pi)

# ---------------- 2D Wilson-Dirac with emergent links ----------------
def wilson_dirac(U, m0):
    """2D Wilson-Dirac operator (2-spinor), gamma_1=sigma_x, gamma_2=sigma_y,
       gamma_5=sigma_z. Returns dense complex matrix of size 2L^2."""
    L = U.shape[0]; N = L*L
    sx = np.array([[0,1],[1,0]],dtype=complex)
    sy = np.array([[0,-1j],[1j,0]],dtype=complex)
    g = [sx, sy]
    I2 = np.eye(2,dtype=complex)
    D = np.zeros((2*N,2*N),dtype=complex)
    def idx(x,y): return (x%L)*L+(y%L)
    r = 1.0
    for x in range(L):
        for y in range(L):
            i = idx(x,y)
            D[2*i:2*i+2,2*i:2*i+2] += (m0 + 2*r)*I2
            for mu,(dx,dy) in enumerate([(1,0),(0,1)]):
                j = idx(x+dx,y+dy)
                Uf = U[x,y,mu]
                P_plus  = -0.5*(r*I2 - g[mu])     # forward
                P_minus = -0.5*(r*I2 + g[mu])     # backward
                D[2*i:2*i+2,2*j:2*j+2] += P_plus*Uf
                D[2*j:2*j+2,2*i:2*i+2] += P_minus*np.conj(Uf)
    return D

# ---------------- main ----------------
def main():
    L = 20; Q = 3; m0 = -1.0   # negative Wilson mass in (-2,0) for index
    print(f"Lattice L={L}, soliton winding Q={Q}, Wilson mass m0={m0}\n")
    z = baby_skyrmion(L, Q)
    U = emergent_links(z)
    flux = plaquette_flux(U)
    print(f"Emergent U(1) total flux (winding): {flux:.3f} (target {Q})")

    N = L*L
    g5 = np.zeros((2*N,2*N),dtype=complex)
    for i in range(N): g5[2*i,2*i]=1; g5[2*i+1,2*i+1]=-1

    D = wilson_dirac(U, m0)
    # ---- EXACT OVERLAP (Ginsparg-Wilson) construction ----
    # H_W = gamma5 D_W (Hermitian); sign(H_W) via eigendecomposition;
    # D_ov = 1 + gamma5 sign(H_W). Zero modes: gamma5 sign(H_W) v = -v.
    Hw = g5 @ D
    Hw = 0.5*(Hw + Hw.conj().T)
    w, V = sla.eigh(Hw)
    signHw = (V * np.sign(w)) @ V.conj().T
    D_ov = np.eye(2*N) + g5 @ signHw
    # zero modes of D_ov: smallest |eigenvalue| of D_ov^dag D_ov
    Dov2 = D_ov.conj().T @ D_ov
    evals, evecs = sla.eigh(Dov2)
    print(f"\nLowest 8 |D_ov|^2 eigenvalues: {np.round(evals[:8],5)}")
    print("(exact zero modes -> values ~0; index theorem predicts 3 for Q=3)\n")
    nlow = 6
    print("Overlap low modes: <psi|gamma5|psi> (exact chirality):")
    n_zero = 0
    for k in range(nlow):
        v = evecs[:,k]
        c = np.real(np.vdot(v, g5@v))
        tag = " <-- zero mode" if evals[k] < 1e-6 else ""
        if evals[k] < 1e-6: n_zero += 1
        print(f"  mode {k}: |D_ov|^2={evals[k]:.5f}  chirality={c:+.3f}{tag}")
    print(f"\nNumber of exact zero modes: {n_zero} (target |Q|={Q})")

    zero_modes = [evecs[:,k] for k in range(3)]

    # spatial densities
    def site_density(v):
        d = np.zeros(N)
        for i in range(N):
            d[i] = abs(v[2*i])**2 + abs(v[2*i+1])**2
        return d

    # Candidate mass-operator fields on sites
    n3 = np.zeros(N)   # Higgs profile = n_3 = |z0|^2-|z1|^2
    for x in range(L):
        for y in range(L):
            i=x*L+y; n3[i]= abs(z[x,y,0])**2 - abs(z[x,y,1])**2
    # winding/flux density per site (plaquette angle at site)
    Bdens=np.zeros(N)
    for x in range(L):
        for y in range(L):
            i=x*L+y
            p=(U[x,y,0]*U[(x+1)%L,y,1]*np.conj(U[x,(y+1)%L,0])*np.conj(U[x,y,1]))
            Bdens[i]=np.angle(p)
    # connection magnitude |a|^2 ~ (1-Re U) summed over directions
    a2=np.zeros(N)
    for x in range(L):
        for y in range(L):
            i=x*L+y
            a2[i]=(1-np.cos(np.angle(U[x,y,0])))+(1-np.cos(np.angle(U[x,y,1])))

    def koide(masses):
        m=np.abs(np.array(masses))
        s=np.sqrt(m)
        return m.sum()/ (s.sum()**2)

    def mass_matrix(field):
        # m_kl = <psi_k| field |psi_l>, then take eigenvalues (masses)
        M=np.zeros((3,3),dtype=complex)
        for k in range(3):
            dk=site_density_vec(zero_modes[k])
            for l in range(3):
                # overlap weighted by field
                M[k,l]=overlap_field(zero_modes[k],zero_modes[l],field)
        ev=np.linalg.eigvalsh(0.5*(M+M.conj().T))
        return np.abs(ev)

    def site_density_vec(v):
        return v
    def overlap_field(vk,vl,field):
        s=0j
        for i in range(N):
            f=field[i]
            s+= f*(np.conj(vk[2*i])*vl[2*i]+np.conj(vk[2*i+1])*vl[2*i+1])
        return s

    # gradient/energy density |partial z|^2 (sigma-model kinetic density)
    grad=np.zeros(N)
    for x in range(L):
        for y in range(L):
            i=x*L+y
            g=0.0
            for dx,dy in [(1,0),(0,1)]:
                dz=z[(x+dx)%L,(y+dy)%L]-z[x,y]
                g+=np.vdot(dz,dz).real
            grad[i]=g
    # radial winding rate: |a| (connection magnitude, not squared)
    amag=np.sqrt(a2)
    # Berry curvature magnitude |B| (absolute flux density)
    Babs=np.abs(Bdens)

    print("\n===== Koide ratio under candidate mass operators =====")
    print("(target Q_K = 2/3 = 0.6667; degenerate = 1/3 = 0.3333)\n")
    results=[]
    for name, field in [("(a) local Higgs n_3", n3),
                        ("(b) winding/flux density B", Bdens),
                        ("(c) connection |a|^2", a2),
                        ("(d) gradient |partial z|^2", grad),
                        ("(e) connection |a|", amag),
                        ("(f) |Berry curvature|", Babs)]:
        masses=mass_matrix(field)
        qk=koide(masses)
        results.append((name,qk))
        print(f"  {name:28s}: masses={np.round(masses,4)}  Q_K={qk:.4f}")
    best=max(results,key=lambda r:r[1])
    print(f"\n  Best (closest to 2/3): {best[0]} with Q_K={best[1]:.4f}")
    print(f"  Local-Higgs baseline: {results[0][1]:.4f} ; trend toward 2/3 = "
          f"{'YES' if best[1]>results[0][1]+0.05 else 'weak'}")

    print("\n===== Honest status =====")
    print("Proof-of-concept at L=20, 2D baby-skyrmion. The question is whether a")
    print("winding-sensitive operator (b,c) moves Q_K toward 2/3 vs the local")
    print("Higgs (a). A positive trend supports the framework's hypothesis; the")
    print("continuum limit and 3D Hopfion (HPC) remain for a definitive value.")

if __name__=="__main__":
    main()
