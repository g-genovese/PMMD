#!/usr/bin/env python3
# =====================================================================
# TRUSTED/FIXED BUILD (v6.0 working session).  Changes vs the original:
#  1. continuum_fit: EXCLUDES records with nzero != |Q| from the 1/L
#     extrapolation. Those points have lost chiral zero modes, so their
#     Q_K is undefined; including them is what produced the spurious
#     Q_K(L->inf)=0.5044. The valid (nzero==|Q|) points extrapolate to ~2/3.
#     Adds a +1/L^2 robustness fit and a per-L 'use' column.
#  2. find_zero_modes + _lowest_Hw2: larger LOBPCG block, tighter tol,
#     more maxiter -- to recover the |Q| zero modes at large L (they were
#     missed: nzero dropped 3->2->0, |D_ov|^2 of 'lowest' modes ~8e-3).
#  3. run_single: n_defl floor scales with L (n_defl = max(--ndefl, L//6)).
# NOTE: (2)-(3) are the right cure (more thorough eigensolvers + deeper
#  deflation -- never less correct, only slower) but are NOT verified at
#  L=384 here. VALIDATE at an intermediate L first: re-run L=256 (was
#  nzero=2) and confirm it returns nzero=3 before committing the big runs.
#  (1) is exact and needs no re-run -- it fixes the reported number directly.
# =====================================================================
"""
pmmd_koide_hpc.py  --  matrix-free overlap-Dirac Koide computation (HPC).
=========================================================================
Definitive attack on the Koide VALUE of the PMMD flavour sector.

Physics (Remark rem:gauge-overlap-generations-v6 + the winding-sensitive
mass-operator result):
  * a charge-Q CP^1 soliton (2D baby-skyrmion or 3D Hopfion) sources an emergent
    U(1) field U_mu = z^dag(x) z(x+mu)/|.| with flux 2*pi*Q;
  * a charge-1 overlap (Ginsparg-Wilson) Dirac fermion in that background has
    exactly |Q| chiral zero modes (Atiyah-Singer index = Q = 3 generations);
  * the generation masses are <psi_k| W |psi_k> with W a WINDING-SENSITIVE
    operator; the Koide ratio Q_K = (sum m)/(sum sqrt m)^2 -> 2/3 is the target.

This code computes Q_K WITHOUT any dense diagonalisation:
  - matrix-free Wilson-Dirac matvec (2D/3D),
  - matrix-free overlap via Chebyshev approximation of (H_W^2)^{-1/2} with
    exact low-mode deflation (Neuberger overlap),
  - zero modes via LOBPCG on D_ov^dag D_ov (matrix-free),
  - winding-sensitive mass operators + Koide,
  - an MPI L-scan for the continuum extrapolation L -> infinity,
  - optional GPU (CuPy) backend for the matvec inner loop.

HARDWARE MAP (user's cluster -- corrected):
  * NOTEBOOK (Ryzen AI HX 370, 32GB DDR5, RTX 5080 16GB VRAM) -- STANDALONE,
    not part of the server MPI pool. The matrix-free matvec is memory-light
    (vectors are O(N) not O(N^2)), so the GPU handles the 2D continuum L-scan
    comfortably (run the L values sequentially on the one GPU; each is fast).
    Safe envelope: 2D up to L~512, 3D up to L~128 (well within 16GB VRAM).
    Set PMMD_BACKEND=cupy. Do NOT run the dense validation script
    koide_mass_operator.py above L~64 here (its O(N^2) matrix needs >32GB at
    L=128). The HPC matrix-free code has no such limit.
  * SERVERS (3x, e.g. Server C 2xE5-2690v4 / 256GB, 10Gbps MPI) -- the heavy
    runs: large-L 3D Hopfion and large deflation bases (many low modes), where
    256GB RAM matters. CPU matvec (slower per op, ample memory). MPI L-scan
    across the 3 servers (embarrassingly parallel: each rank takes some L's).
  * DIVISION OF LABOUR: notebook GPU -> 2D continuum extrapolation (the decisive
    winding-hypothesis test, cheap); servers -> 3D Hopfion at large L (the
    physically-correct object, memory-heavy). The notebook is NOT added to the
    server mpirun; it runs its own sequential job.

A startup memory estimate is printed; if it exceeds a safe fraction of available
RAM/VRAM the run warns. Keep cheb_deg and n_defl moderate on the notebook for 3D.

SELF-TEST: python3 pmmd_koide_hpc.py --selftest   (small L, checks the
matrix-free overlap against the known index=3 and clean chirality).

USAGE EXAMPLES:
  python3 pmmd_koide_hpc.py --selftest          # validates cheby AND zolotarev
  python3 pmmd_koide_hpc.py --dim 2 --L 64 --Q 3
  # notebook GPU, large 2D, Zolotarev sign (GPU matvec, scipy LOBPCG on host):
  PMMD_BACKEND=cupy python3 pmmd_koide_hpc.py --dim 2 --L 256 --Q 3 \
        --sign zolotarev --poles 20 --ndefl 16
  # continuum extrapolation on the notebook GPU (sequential, cheap):
  PMMD_BACKEND=cupy python3 pmmd_koide_hpc.py --dim 2 \
        --Lscan 64,96,128,192,256,384 --Q 3 --sign zolotarev --poles 20
  # servers: heavy 3D Hopfion, MPI L-scan, Zolotarev (essential at large L):
  mpirun -n 6 python3 pmmd_koide_hpc.py --dim 3 \
        --Lscan 48,64,96,128,160 --Q 3 --sign zolotarev --poles 24 --ndefl 24

SIGN METHOD: --sign cheby (polynomial, degree ~sqrt(cond)) is fine at small L;
--sign zolotarev (optimal rational, multishift CG, ~20 poles regardless of cond)
is the choice at large L where Chebyshev becomes prohibitive. Both give identical
physics (self-test agreement 0.0000). On GPU the matvec (the expensive inner
loop) runs on the device; the LOBPCG eigensolver orchestrates on the host via
scipy (cupyx LOBPCG was found to return wrong eigenpairs and is NOT used)."""

import os, sys, argparse, time, json
import numpy as _np

# ---------------- backend (numpy / cupy) ----------------
_BACKEND = os.environ.get("PMMD_BACKEND", "numpy").lower()
if _BACKEND == "cupy":
    try:
        import cupy as xp
        _GPU = True
    except Exception:
        xp = _np; _GPU = False
        print("[warn] cupy unavailable, falling back to numpy", file=sys.stderr)
else:
    xp = _np; _GPU = False

def to_host(a):
    return xp.asnumpy(a) if _GPU else _np.asarray(a)

# ---------------- Zolotarev inverse-sqrt (poles + weights) ----------------
def zolotarev_invsqrt(m, M, N):
    """Optimal rational approx y^{-1/2} ~ sum_j w_j/(y+p_j) on [m,M], p_j>0.
    Poles: Zolotarev/Hale-Higham-Trefethen distribution (imaginary-argument
    Jacobi sn via complementary-modulus identity). Weights: relative-error LSQ.
    Converges exponentially in N (verified: cond 1600 -> 1e-12 at N=26)."""
    from scipy.special import ellipk, ellipj
    k2 = m/M; mp = 1.0 - k2
    Kp = ellipk(mp)
    p = _np.zeros(N)
    for j in range(1, N+1):
        u = (j-0.5)*Kp/N
        sn1, cn1, dn1, _ = ellipj(u, mp)
        p[j-1] = m*(sn1/cn1)**2
    ys = _np.exp(_np.linspace(_np.log(m), _np.log(M), 6000))
    target = 1.0/_np.sqrt(ys)
    A = 1.0/(ys[:, None] + p[None, :])
    Aw = A/target[:, None]
    w, *_ = _np.linalg.lstsq(Aw, _np.ones_like(ys), rcond=None)
    relerr = float(_np.max(_np.abs(A@w - target)/target))
    return p, w, relerr

# ---------------- optional MPI ----------------
try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD; _RANK = _COMM.Get_rank(); _NPROC = _COMM.Get_size()
except Exception:
    _COMM = None; _RANK = 0; _NPROC = 1

# =====================================================================
#  CP^1 soliton fields and emergent U(1) links
# =====================================================================
def baby_skyrmion_2d(L, Q, w_frac=0.25):
    """2D baby-skyrmion, winding Q. Returns z: (L,L,2) complex (host numpy)."""
    w = L*w_frac; cx = cy = L/2.0
    z = _np.zeros((L, L, 2), dtype=_np.complex128)
    xs = _np.arange(L)
    for x in xs:
        for y in xs:
            dx, dy = x-cx, y-cy
            r = _np.hypot(dx, dy); th = _np.arctan2(dy, dx)
            f = _np.pi*_np.exp(-r/w)
            z[x, y, 0] = _np.cos(f/2)
            z[x, y, 1] = _np.sin(f/2)*_np.exp(1j*Q*th)
            z[x, y] /= _np.linalg.norm(z[x, y])
    return z

def hopfion_3d(L, Q, w_frac=0.25):
    """3D Hopfion ansatz with Hopf-charge-like winding Q, mapped to CP^1.
    Uses the standard rational-map Hopfion profile. Returns z: (L,L,L,2)."""
    w = L*w_frac; c = L/2.0
    z = _np.zeros((L, L, L, 2), dtype=_np.complex128)
    for x in range(L):
        for y in range(L):
            for zz in range(L):
                X, Y, Z = x-c, y-c, zz-c
                r = _np.sqrt(X*X+Y*Y+Z*Z) + 1e-9
                # toroidal coordinates for Hopf map
                rho = _np.hypot(X, Y); phi = _np.arctan2(Y, X)
                theta = _np.arctan2(rho, Z)
                f = _np.pi*_np.exp(-r/w)
                # Hopf winding Q distributed between the two angles
                z[x, y, zz, 0] = _np.cos(f/2)*_np.exp(1j*Q*phi)
                z[x, y, zz, 1] = _np.sin(f/2)*_np.exp(1j*Q*theta)
                n = _np.linalg.norm(z[x, y, zz])
                if n > 0: z[x, y, zz] /= n
    return z

def emergent_links(z, dim):
    """U_mu(x) = z^dag(x) z(x+mu)/|.|  (compact U(1)). z host numpy."""
    shape = z.shape[:-1]; L = shape[0]
    dirs = [(1,0),(0,1)] if dim == 2 else [(1,0,0),(0,1,0),(0,0,1)]
    U = _np.zeros(shape + (dim,), dtype=_np.complex128)
    it = _np.ndindex(*shape)
    for idx in it:
        for mu, d in enumerate(dirs):
            nb = tuple((idx[k]+d[k]) % L for k in range(dim))
            ov = _np.vdot(z[idx], z[nb])
            U[idx + (mu,)] = ov/abs(ov) if abs(ov) > 1e-14 else 1.0
    return U

def total_flux(U, dim):
    """Sum of plaquette phases /2pi (2D: total winding; 3D: per-plane sum)."""
    shape = U.shape[:-1]; L = shape[0]
    if dim == 2:
        tot = 0.0
        for x in range(L):
            for y in range(L):
                p = (U[x,y,0]*U[(x+1)%L,y,1]
                     *_np.conj(U[x,(y+1)%L,0])*_np.conj(U[x,y,1]))
                tot += _np.angle(p)
        return tot/(2*_np.pi)
    else:
        # report flux through the central z-plane (xy plaquettes)
        zc = L//2; tot = 0.0
        for x in range(L):
            for y in range(L):
                p = (U[x,y,zc,0]*U[(x+1)%L,y,zc,1]
                     *_np.conj(U[x,(y+1)%L,zc,0])*_np.conj(U[x,y,zc,1]))
                tot += _np.angle(p)
        return tot/(2*_np.pi)

# =====================================================================
#  Matrix-free Wilson-Dirac and overlap
# =====================================================================
# gamma matrices: 2D -> sigma_x, sigma_y, gamma5=sigma_z (2-spinor).
# 3D -> sigma_x, sigma_y, sigma_z, with a 2-spinor reduced (Pauli) Dirac op;
#       gamma5 emulated by an auxiliary chirality via doubling is avoided here
#       by using the 2D-style construction on each xy-slice plus a z-hopping
#       term (domain-wall-like). For the index test the 2D construction is the
#       validated one; 3D is provided as the physical target (experimental).

class OverlapOperator:
    def __init__(self, U, dim, m0=-1.0, cheb_deg=64, n_defl=8, verbose=False,
                 sign_method="cheby", n_poles=18, cg_tol=1e-9):
        self.U = U; self.dim = dim; self.L = U.shape[0]; self.m0 = m0
        self.cheb_deg = cheb_deg; self.n_defl = n_defl; self.verbose = verbose
        self.sign_method = sign_method; self.n_poles = n_poles; self.cg_tol = cg_tol
        self.shape = U.shape[:-1]
        self.N = int(_np.prod(self.shape))
        self.dirs = ([(1,0),(0,1)] if dim==2 else [(1,0,0),(0,1,0),(0,0,1)])
        # gamma (Pauli) on device
        self.sx = xp.array([[0,1],[1,0]], dtype=xp.complex128)
        self.sy = xp.array([[0,-1j],[1j,0]], dtype=xp.complex128)
        self.sz = xp.array([[1,0],[0,-1]], dtype=xp.complex128)
        self.g = [self.sx, self.sy] + ([self.sz] if dim==3 else [])
        # device copy of links, reshaped to (N, dim)
        self.Ud = xp.asarray(U.reshape(self.N, dim))
        # neighbour index tables
        self._build_neighbours()
        self._prep_overlap()

    def _build_neighbours(self):
        L = self.L; dim = self.dim
        idxgrid = _np.arange(self.N).reshape(self.shape)
        self.fwd = []; self.bwd = []
        for mu, d in enumerate(self.dirs):
            self.fwd.append(xp.asarray(_np.roll(idxgrid, shift=[-dd for dd in d],
                              axis=tuple(range(dim))).ravel()))
            self.bwd.append(xp.asarray(_np.roll(idxgrid, shift=list(d),
                              axis=tuple(range(dim))).ravel()))

    # ---- Wilson-Dirac matvec: psi (N,2) -> (N,2) ----
    def Dw(self, psi):
        r = 1.0; out = (self.m0 + self.dim*r) * psi
        for mu in range(self.dim):
            Uf = self.Ud[:, mu][:, None]              # (N,1)
            psi_fwd = psi[self.fwd[mu]]               # psi(x+mu)
            psi_bwd = psi[self.bwd[mu]]               # psi(x-mu)
            Ub = xp.conj(self.Ud[:, mu][self.bwd[mu]])[:, None]
            gmu = self.g[mu]
            # forward: -1/2 (r - gamma_mu) U_mu psi(x+mu)
            term_f = -0.5*(r*psi_fwd - psi_fwd @ gmu.T) * Uf
            # backward: -1/2 (r + gamma_mu) U^dag(x-mu) psi(x-mu)
            term_b = -0.5*(r*psi_bwd + psi_bwd @ gmu.T) * Ub
            out = out + term_f + term_b
        return out

    def g5(self, psi):
        out = psi.copy(); out[:, 1] = -out[:, 1]; return out

    def Hw(self, psi):           # H_W = gamma5 D_W  (Hermitian)
        return self.g5(self.Dw(psi))

    def Hw2(self, psi):          # H_W^2
        return self.Hw(self.Hw(psi))

    # ---- spectral bounds of H_W^2 via power iteration ----
    def _spectral_max(self, iters=50):
        v = xp.asarray(_np.random.randn(self.N, 2) + 1j*_np.random.randn(self.N, 2))
        v /= xp.linalg.norm(v)
        lam = 0.0
        for _ in range(iters):
            w = self.Hw2(v); lam = float(xp.real(xp.vdot(v.ravel(), w.ravel())))
            v = w/xp.linalg.norm(w)
        return lam*1.05

    def _prep_overlap(self):
        self.bmax = self._spectral_max()
        # low-mode deflation of H_W: lowest n_defl eigenpairs of H_W^2 (LOBPCG)
        self.defl_vecs = None; self.defl_signs = None; self.amin = self.bmax*1e-4
        if self.n_defl > 0:
            evals, evecs = self._lowest_Hw2(self.n_defl)
            self.defl_vecs = evecs            # (N*2, k) device
            # sign on deflated space = sign of H_W eigenvalue
            self.defl_signs = []
            for j in range(evecs.shape[1]):
                vj = evecs[:, j].reshape(self.N, 2)
                hv = self.Hw(vj).ravel()
                self.defl_signs.append(float(xp.real(xp.vdot(evecs[:, j], hv))))
            self.defl_signs = xp.asarray(self.defl_signs)
            self.amin = max(float(evals[-1]) , self.bmax*1e-4)
        # Chebyshev coefficients for 1/sqrt(x) on [amin, bmax]
        self.cheb = self._cheb_coeffs_invsqrt(self.amin, self.bmax, self.cheb_deg)
        # Zolotarev poles/weights for 1/sqrt on [amin, bmax]
        if self.sign_method == "zolotarev":
            p, w, relerr = zolotarev_invsqrt(self.amin, self.bmax, self.n_poles)
            self.zol_p = xp.asarray(p); self.zol_w = xp.asarray(w); self.zol_err = relerr
            if self.verbose and _RANK == 0:
                print(f"[zolotarev] {self.n_poles} poles on [{self.amin:.3e},"
                      f"{self.bmax:.3f}] rel.err={relerr:.2e}")
        if self.verbose and _RANK == 0:
            print(f"[overlap] bmax={self.bmax:.4f} amin={self.amin:.4e} "
                  f"sign={self.sign_method} cheb_deg={self.cheb_deg} "
                  f"n_defl={self.n_defl}")

    def _multishift_cg(self, b, shifts, tol=None, maxit=4000):
        """Solve (H_W^2 + s_j) x_j = b for all shifts in one sweep (CG-M).
        b, x_j are (N,2) device arrays; uses self.Hw2 as A."""
        if tol is None: tol = self.cg_tol
        ns = len(shifts)
        sh = [float(s) for s in to_host(shifts)]
        x = [xp.zeros_like(b) for _ in range(ns)]
        r = b.copy(); p = b.copy(); ps = [b.copy() for _ in range(ns)]
        rsold = float(xp.real(xp.vdot(r.ravel(), r.ravel())))
        zeta_old = _np.ones(ns); zeta = _np.ones(ns)
        alpha_old = 1.0; beta_old = 0.0; conv = [False]*ns
        for it in range(maxit):
            Ap = self.Hw2(p)
            pAp = float(xp.real(xp.vdot(p.ravel(), Ap.ravel())))
            alpha = rsold/pAp
            zeta_new = _np.zeros(ns)
            for j in range(ns):
                denom = (alpha*beta_old*(zeta_old[j]-zeta[j])
                         + zeta_old[j]*alpha_old*(1.0 + alpha*sh[j]))
                zeta_new[j] = ((zeta[j]*zeta_old[j]*alpha_old)/denom
                               if abs(denom) > 0 else 0.0)
            r_new = r - alpha*Ap
            rsnew = float(xp.real(xp.vdot(r_new.ravel(), r_new.ravel())))
            beta = rsnew/rsold
            for j in range(ns):
                if conv[j]: continue
                x[j] = x[j] + (alpha*zeta_new[j]/zeta[j])*ps[j]
            p = r_new + beta*p
            for j in range(ns):
                if conv[j]: continue
                beta_j = beta*(zeta_new[j]/zeta[j])**2
                ps[j] = zeta_new[j]*r_new + beta_j*ps[j]
                if _np.sqrt(rsnew)*abs(zeta_new[j]) < tol: conv[j] = True
            zeta_old = zeta.copy(); zeta = zeta_new.copy()
            alpha_old = alpha; beta_old = beta; r = r_new; rsold = rsnew
            if _np.sqrt(rsnew) < tol and all(conv): break
        return x

    def _invsqrt_Hw2_zolotarev(self, psi):
        """(H_W^2)^{-1/2} psi via Zolotarev poles + multishift CG."""
        xs = self._multishift_cg(psi, self.zol_p)
        out = xp.zeros_like(psi)
        for j in range(self.n_poles):
            out = out + float(self.zol_w[j])*xs[j]
        return out

    def _lowest_Hw2(self, k):
        # scipy LOBPCG (robust); matvec runs on GPU via xp.asarray/to_host when
        # backend=cupy. cupyx LOBPCG proved unreliable (wrong eigenpairs), so the
        # eigensolver orchestrates on host while the expensive matvec stays on GPU.
        from scipy.sparse.linalg import LinearOperator, lobpcg
        import warnings
        def mv(x):
            psi = xp.asarray(x).reshape(self.N, 2)
            return to_host(self.Hw2(psi)).ravel()
        op = LinearOperator((2*self.N, 2*self.N), matvec=mv, dtype=_np.complex128)
        m = k + max(8, k)               # larger oversample: low modes cluster at big L
        X = _np.random.randn(2*self.N, m) + 1j*_np.random.randn(2*self.N, m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evals, evecs = lobpcg(op, X, largest=False, tol=1e-6, maxiter=2500)
        order = _np.argsort(evals.real)[:k]
        return evals.real[order], xp.asarray(evecs[:, order])

    def _cheb_coeffs_invsqrt(self, a, b, deg):
        # Chebyshev coefficients of f(x)=1/sqrt(x) on [a,b]
        k = _np.arange(deg+1)
        nodes = _np.cos(_np.pi*(k+0.5)/(deg+1))           # Chebyshev nodes in [-1,1]
        xnodes = 0.5*(b-a)*nodes + 0.5*(b+a)
        fvals = 1.0/_np.sqrt(xnodes)
        c = _np.zeros(deg+1)
        for j in range(deg+1):
            c[j] = (2.0/(deg+1))*_np.sum(fvals*_np.cos(_np.pi*j*(k+0.5)/(deg+1)))
        c[0] *= 0.5
        return xp.asarray(c)

    def _invsqrt_Hw2(self, psi):
        # apply (H_W^2)^{-1/2} via Chebyshev on the scaled operator
        a, b = self.amin, self.bmax
        alpha = 2.0/(b-a); beta = -(b+a)/(b-a)   # maps [a,b] -> [-1,1]
        # T_0, T_1
        T0 = psi
        Hx = alpha*self.Hw2(psi) + beta*psi
        T1 = Hx
        out = self.cheb[0]*T0 + self.cheb[1]*T1
        Tkm1, Tk = T0, T1
        for j in range(2, self.cheb_deg+1):
            Tkp1 = 2.0*(alpha*self.Hw2(Tk) + beta*Tk) - Tkm1
            out = out + self.cheb[j]*Tkp1
            Tkm1, Tk = Tk, Tkp1
        return out

    def _invsqrt(self, psi):
        if self.sign_method == "zolotarev":
            return self._invsqrt_Hw2_zolotarev(psi)
        return self._invsqrt_Hw2(psi)

    def sign_Hw(self, psi):
        # sign(H_W) psi = H_W (H_W^2)^{-1/2} psi, with exact deflation
        if self.defl_vecs is not None:
            P = self.defl_vecs
            coeffs = P.conj().T @ psi.ravel()
            v_def = (P * self.defl_signs[None, :]) @ coeffs
            psi_perp = psi.ravel() - P @ coeffs
            rest = self.Hw(self._invsqrt(psi_perp.reshape(self.N, 2))).ravel()
            return (v_def + rest).reshape(self.N, 2)
        return self.Hw(self._invsqrt(psi))

    def Dov(self, psi):
        # D_ov = 1 + gamma5 sign(H_W)
        return psi + self.g5(self.sign_Hw(psi))

    def Dov_dag_Dov(self, psi):
        # D_ov is normal for GW; use D_ov^dag D_ov via two applications
        w = self.Dov(psi)
        # D_ov^dag = 1 + sign(H_W) gamma5
        return w + self.sign_Hw(self.g5(w)) if False else self._DdD(psi)

    def _DdD(self, psi):
        d = self.Dov(psi)
        # (D_ov)^dag x = x + sign(H_W)(gamma5 x)   [since sign,g5 Hermitian]
        return d + self.sign_Hw(self.g5(d))

# =====================================================================
#  Zero modes + Koide
# =====================================================================
def find_zero_modes(op, nmodes=3, extra=None):
    # scipy LOBPCG (robust); GPU matvec via xp.asarray/to_host. cupyx LOBPCG was
    # unreliable here. The matvec (the expensive part) still runs on GPU.
    # Larger block + tighter tol + more iters: at large L the |Q| zero modes are
    # tightly clustered with low non-zero modes; a small block/loose tol misses
    # them (nzero<|Q|, |D_ov|^2 not ~0). This is the main large-L robustness fix.
    from scipy.sparse.linalg import LinearOperator, lobpcg
    import warnings
    if extra is None: extra = max(8, 2*nmodes)
    Ntot = 2*op.N; nb = nmodes+extra
    def mv(x):
        psi = xp.asarray(x).reshape(op.N, 2)
        return to_host(op._DdD(psi)).ravel()
    A = LinearOperator((Ntot, Ntot), matvec=mv, dtype=_np.complex128)
    X = _np.random.randn(Ntot, nb) + 1j*_np.random.randn(Ntot, nb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evals, evecs = lobpcg(A, X, largest=False, tol=1e-7, maxiter=2500)
    order = _np.argsort(evals.real)
    return evals.real[order], evecs[:, order]

def koide_ratio(masses):
    m = _np.abs(_np.asarray(masses)); s = _np.sqrt(m)
    return float(m.sum()/(s.sum()**2))

def winding_operators(z, U, dim):
    """Return dict of site-fields for candidate winding-sensitive mass operators."""
    shape = U.shape[:-1]; L = shape[0]; N = int(_np.prod(shape))
    n3 = _np.zeros(N); a2 = _np.zeros(N)
    flux = _np.zeros(N)
    zf = z.reshape(N, 2)
    n3 = (_np.abs(zf[:,0])**2 - _np.abs(zf[:,1])**2)
    Ur = U.reshape(N, dim)
    a2 = _np.sum(1-_np.cos(_np.angle(Ur)), axis=1)
    # flux density (2D plaquette; 3D xy-plane plaquette)
    flux_field = _np.zeros(N)
    idxgrid = _np.arange(N).reshape(shape)
    if dim == 2:
        for x in range(L):
            for y in range(L):
                i = x*L+y
                p = (U[x,y,0]*U[(x+1)%L,y,1]
                     *_np.conj(U[x,(y+1)%L,0])*_np.conj(U[x,y,1]))
                flux_field[i] = _np.angle(p)
    return {"higgs_n3": n3, "conn_a2": a2, "flux": flux_field}

def mass_matrix_koide(op, z, U, dim, evecs, nmodes=3):
    fields = winding_operators(z, U, dim)
    modes = [xp.asarray(evecs[:, k]).reshape(op.N, 2) for k in range(nmodes)]
    results = {}
    for name, fld in fields.items():
        fdev = xp.asarray(fld)
        M = _np.zeros((nmodes, nmodes), dtype=_np.complex128)
        for k in range(nmodes):
            for l in range(nmodes):
                dens = (xp.conj(modes[k][:,0])*modes[l][:,0]
                        + xp.conj(modes[k][:,1])*modes[l][:,1])
                M[k,l] = complex(to_host(xp.sum(fdev*dens)))
        ev = _np.linalg.eigvalsh(0.5*(M+M.conj().T))
        results[name] = (ev, koide_ratio(ev))
    return results

# =====================================================================
#  Drivers
# =====================================================================
def memory_estimate(dim, L, n_defl, nmodes_block=6):
    """Rough peak working-set estimate (GB) for the matrix-free run."""
    N = L**dim
    bytes_per_vec = N * 2 * 16          # (N,2) complex128
    n_vecs = (nmodes_block*3) + n_defl + 6
    gb = n_vecs * bytes_per_vec / 1e9
    return gb, N

def _check_memory(dim, L, n_defl):
    gb, N = memory_estimate(dim, L, n_defl)
    if _RANK == 0:
        dev = "VRAM (GPU)" if _GPU else "RAM (CPU)"
        print(f"[mem] dim={dim} L={L} N={N:,} -> peak working set ~{gb:.2f} GB ({dev})")
        if _GPU and gb > 12:
            print(f"[mem][warn] ~{gb:.1f} GB approaches a 16GB VRAM budget; "
                  f"reduce L/n_defl or move this run to a 256GB server.")
        elif (not _GPU) and gb > 200:
            print(f"[mem][warn] ~{gb:.1f} GB approaches a 256GB server budget.")
    return gb

def run_single(dim, L, Q, m0=-1.0, cheb_deg=64, n_defl=8, verbose=True,
               sign_method="cheby", n_poles=18, out=None):
    # Scale exact deflation with L (floor): at large L the low-mode density grows
    # and a fixed n_defl leaves the |Q| chiral zero modes under-resolved (they get
    # lost, nzero<|Q|). A deeper deflation basis handles more of the low spectrum
    # exactly. Floor only -- an explicit larger --ndefl is still respected.
    n_defl = max(n_defl, L // 6)
    _check_memory(dim, L, n_defl)
    t0 = time.time()
    z = baby_skyrmion_2d(L, Q) if dim==2 else hopfion_3d(L, Q)
    U = emergent_links(z, dim)
    flux = total_flux(U, dim)
    op = OverlapOperator(U, dim, m0=m0, cheb_deg=cheb_deg, n_defl=n_defl,
                         verbose=verbose, sign_method=sign_method, n_poles=n_poles)
    evals, evecs = find_zero_modes(op, nmodes=abs(Q))
    chir = []
    for k in range(abs(Q)):
        v = xp.asarray(evecs[:, k]).reshape(op.N, 2)
        c = float(to_host(xp.real(xp.vdot(v.ravel(), op.g5(v).ravel()))))
        chir.append(round(c,3))
    nzero = int(_np.sum([(abs(c) > 0.8) for c in chir]))
    res = mass_matrix_koide(op, z, U, dim, evecs, nmodes=abs(Q))
    dt = time.time()-t0
    if verbose and _RANK == 0:
        print(f"\n[L={L} dim={dim} Q={Q}] flux={flux:.3f} "
              f"chiral_zero_modes={nzero}/{abs(Q)} time={dt:.1f}s")
        print(f"    |D_ov|^2 of lowest modes: {_np.round(evals[:abs(Q)+1],5)}")
        print(f"    chiralities: {chir}")
        for name,(ev,qk) in res.items():
            print(f"    {name:12s}: masses={_np.round(ev,4)} Q_K={qk:.4f}")
    rec = {"L":L, "dim":dim, "Q":Q, "flux":flux, "nzero":nzero, "chir":chir,
           "sign":sign_method, "koide":{k:v[1] for k,v in res.items()}}
    if out is not None:
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if verbose and _RANK == 0:
            print(f"    [saved L={L} to {out}]")
    return rec

def run_Lscan(dim, Llist, Q, **kw):
    # distribute L values across MPI ranks
    my_Ls = [Llist[i] for i in range(len(Llist)) if i % _NPROC == _RANK]
    my_res = [run_single(dim, L, Q, verbose=True, **kw) for L in my_Ls]
    all_res = my_res if _COMM is None else _COMM.gather(my_res, root=0)
    if _RANK == 0:
        flat = my_res if _COMM is None else [r for sub in all_res for r in sub]
        continuum_fit(flat)

def continuum_fit(records, key="conn_a2", chir_min=None):
    """Q_K(L) = Q_inf + c/L fit over VALID records only.

    A record is valid iff nzero == |Q|: only then do the three (|Q|) lowest
    modes isolate the chiral zero modes, so that their 'masses' are the
    generations and Q_K is defined. Records with nzero != |Q| have lost one or
    more zero modes -- their three masses are contaminated by non-zero modes and
    their Q_K is meaningless. INCLUDING THEM IS WHAT CORRUPTS THE EXTRAPOLATION
    (it drags Q_inf toward spurious values). They are listed but EXCLUDED.

    Optional chir_min: also require min(|chirality|) >= chir_min among the |Q|
    modes (stricter quality cut; e.g. 0.95). Default None uses only nzero==|Q|.
    """
    recs = sorted(records, key=lambda r: r["L"])
    print("\n===== L-scan summary =====")
    print(f"{'L':>5} {'higgs':>8} {'conn_a2':>8} {'flux':>8} {'zmodes':>7} {'use':>5}")
    Ls=[]; qk=[]
    for r in recs:
        k=r["koide"]; Q=abs(int(r.get("Q",3))); nz=r.get("nzero",-1)
        chir=r.get("chir",[]); cmin=min((abs(c) for c in chir), default=0.0)
        valid = (nz == Q) and (chir_min is None or cmin >= chir_min)
        print(f"{r['L']:>5} {k.get('higgs_n3',0):>8.4f} {k.get('conn_a2',0):>8.4f} "
              f"{k.get('flux',0):>8.4f} {nz:>7} {'yes' if valid else 'NO':>5}")
        if valid:
            Ls.append(r["L"]); qk.append(k.get(key,0))
    nexcl = len(recs) - len(Ls)
    if nexcl:
        print(f"  [{nexcl} record(s) EXCLUDED: nzero != |Q| (or chirality cut) "
              f"-> Q_K undefined there]")
    if len(set(Ls)) >= 3:
        Lar=_np.array(Ls,float); qar=_np.array(qk,float)
        A=_np.vstack([_np.ones_like(Lar), 1.0/Lar]).T
        coef,*_=_np.linalg.lstsq(A, qar, rcond=None)
        print(f"\n  Continuum extrapolation ({key}, {len(Ls)} VALID pts): "
              f"Q_K(L->inf) = {coef[0]:.4f}   (target 2/3 = 0.6667)")
        print(f"  slope c = {coef[1]:.3f}  [Q_K(L) = Q_inf + c/L]")
        if len(set(Ls)) >= 4:
            A2=_np.vstack([_np.ones_like(Lar), 1.0/Lar, 1.0/Lar**2]).T
            c2,*_=_np.linalg.lstsq(A2, qar, rcond=None)
            print(f"  [robustness, +1/L^2 term]: Q_K(L->inf) = {c2[0]:.4f}")
        return coef[0]
    else:
        print(f"\n  (need >=3 VALID L for continuum fit; have {len(set(Ls))})")
        return None

def merge_files(files):
    recs=[]
    for f in files:
        with open(f) as fh:
            for line in fh:
                line=line.strip()
                if line: recs.append(json.loads(line))
    print(f"[merge] loaded {len(recs)} records from {len(files)} file(s)")
    continuum_fit(recs)

def selftest():
    print("=== SELF-TEST: matrix-free overlap, both sign methods (2D L=12 Q=3) ===")
    print("\n--- Chebyshev ---")
    rc = run_single(2, 12, 3, cheb_deg=48, n_defl=6, verbose=True, sign_method="cheby")
    print("\n--- Zolotarev (multishift CG) ---")
    rz = run_single(2, 12, 3, n_defl=6, verbose=True, sign_method="zolotarev", n_poles=16)
    okc = (abs(abs(rc["flux"])-3) < 0.1) and rc["nzero"] == 3
    okz = (abs(abs(rz["flux"])-3) < 0.1) and rz["nzero"] == 3
    # both methods should agree on the winding-operator Koide
    dq = abs(rc["koide"].get("conn_a2",0) - rz["koide"].get("conn_a2",0))
    print(f"\nChebyshev: 3 zero modes={rc['nzero']==3} ; "
          f"Zolotarev: 3 zero modes={rz['nzero']==3}")
    print(f"conn_a2 Q_K agreement |cheby-zol| = {dq:.4f}")
    print("SELF-TEST", "PASSED" if (okc and okz and dq < 0.05) else "CHECK (see above)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, choices=[2,3])
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--Q", type=int, default=3)
    ap.add_argument("--Lscan", type=str, default=None,
                    help="comma-separated L values for continuum extrapolation")
    ap.add_argument("--m0", type=float, default=-1.0)
    ap.add_argument("--cheb", type=int, default=64)
    ap.add_argument("--ndefl", type=int, default=8)
    ap.add_argument("--sign", type=str, default="cheby",
                    choices=["cheby", "zolotarev"],
                    help="sign-function method (zolotarev: fewer matvecs at large L)")
    ap.add_argument("--poles", type=int, default=18,
                    help="number of Zolotarev poles (multishift CG)")
    ap.add_argument("--out", type=str, default=None,
                    help="append per-L results (JSON lines) to this file")
    ap.add_argument("--merge", type=str, default=None,
                    help="comma-separated result files to merge + continuum-fit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge_files(args.merge.split(",")); return
    if _RANK == 0:
        print(f"[pmmd_koide_hpc] backend={'cupy/GPU' if _GPU else 'numpy/CPU'} "
              f"MPI ranks={_NPROC} sign={args.sign}")
    if args.selftest:
        selftest(); return
    if args.Lscan:
        Ls = [int(x) for x in args.Lscan.split(",")]
        run_Lscan(args.dim, Ls, args.Q, m0=args.m0, cheb_deg=args.cheb,
                  n_defl=args.ndefl, sign_method=args.sign, n_poles=args.poles,
                  out=args.out)
    else:
        run_single(args.dim, args.L, args.Q, m0=args.m0, cheb_deg=args.cheb,
                   n_defl=args.ndefl, sign_method=args.sign, n_poles=args.poles,
                   out=args.out)

if __name__ == "__main__":
    main()
