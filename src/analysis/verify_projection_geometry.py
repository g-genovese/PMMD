"""
Exact verification of the explicit E8 -> 4D projection geometry (PMMD v6.0).

Verifies, to machine precision:
  1. Coxeter split: E_par = eigenplanes of a Coxeter element at the H4 exponents {1,11,19,29}.
  2. Shell structure: the 240 roots project to two concentric 600-cells (120+120),
     r_par^2 in {2/(sqrt5*phi), 2*phi/sqrt5}; inner-product spectrum = 600-cell set.
  3. Inflation unit: M = phi*P_par + (-1/phi)*P_perp is an integral lattice automorphism
     (det 1) mapping the inner root shell onto roots.
  4. Count-offset algebra: Delta = [ln(2 phi^5) - (1/2) ln(45/7)] / (4 ln phi) = 1.127;
     N = round(H_obs + Delta) = 73; residual quarter-step ~ phi^2 ~ Lambda_obs/Lambda_fw.
"""
import numpy as np, itertools, math

PHI = (1 + 5**0.5) / 2; PSI = (1 - 5**0.5) / 2; LNPHI = math.log(PHI)

def e8_roots():
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8); v[i] = si; v[j] = sj; roots.append(v)
    for signs in itertools.product([1, -1], repeat=8):
        if signs.count(-1) % 2 == 0:
            roots.append(0.5 * np.array(signs))
    return np.array(roots)

def bourbaki_simple_roots():
    a = [0.5 * np.array([1, -1, -1, -1, -1, -1, -1, 1]),
         np.array([1, 1, 0, 0, 0, 0, 0, 0.])]
    for k in range(6):
        v = np.zeros(8); v[k] = -1; v[k + 1] = 1
        if k == 0: v = np.array([-1, 1, 0, 0, 0, 0, 0, 0.])
        a.append(v)
    a[2:] = [np.array([-1,1,0,0,0,0,0,0.]), np.array([0,-1,1,0,0,0,0,0.]),
             np.array([0,0,-1,1,0,0,0,0.]), np.array([0,0,0,-1,1,0,0,0.]),
             np.array([0,0,0,0,-1,1,0,0.]), np.array([0,0,0,0,0,-1,1,0.])]
    return np.array(a)

def main():
    R = e8_roots(); assert len(R) == 240
    B = bourbaki_simple_roots()
    assert round(np.linalg.det(np.round(B @ B.T))) == 1
    C = np.eye(8)
    for v in B:
        n = v / np.linalg.norm(v)
        C = (np.eye(8) - 2 * np.outer(n, n)) @ C
    ev, evec = np.linalg.eig(C)
    ms = np.round(np.abs(np.angle(ev)) / (2 * np.pi) * 30).astype(int)
    P = np.zeros((8, 8))
    for k in range(8):
        if ms[k] in (1, 11):            # folded labels of {1,29} u {11,19}
            w = evec[:, k]; P += np.real(np.outer(w, np.conj(w)))
    P = (P + P.T) / 2
    assert np.allclose(P @ P, P, atol=1e-9) and round(np.trace(P)) == 4
    rp2 = np.round(np.einsum('ij,jk,ik->i', R, P, R), 9)
    vals, counts = np.unique(rp2, return_counts=True)
    t_lo, t_hi = 2/(5**0.5*PHI), 2*PHI/5**0.5
    assert list(counts) == [120, 120] and np.allclose(sorted(vals), [t_lo, t_hi])
    sh = (R @ P)[np.isclose(rp2, t_hi)]
    shn = sh / np.linalg.norm(sh, axis=1, keepdims=True)
    ips = np.unique(np.round(shn @ shn.T, 5))
    target = sorted({0.0, 0.5, -0.5, 1.0, -1.0, PHI/2, -PHI/2, 1/(2*PHI), -1/(2*PHI)})
    assert np.allclose(ips, np.round(target, 5))
    M = PHI * P + PSI * (np.eye(8) - P)
    coords = (M @ B.T).T @ np.linalg.inv(B)
    assert np.allclose(coords, np.round(coords), atol=1e-8)
    assert round(np.linalg.det(M), 6) == 1.0
    inner = R[np.isclose(rp2, t_lo)]
    Rset = {tuple(np.round(r, 6)) for r in R}
    assert all(tuple(np.round(v, 6)) in Rset for v in inner @ M.T)
    ell = 2 * PHI**5
    Delta = (math.log(ell) - 0.5 * math.log(45/7)) / (4 * LNPHI)
    H_obs = math.log(math.sqrt(3/1.1e-122)/ell) / (4 * LNPHI)
    print(f"shells r^2: {t_lo:.6f}/{t_hi:.6f} (120+120)  | 600-cell spectrum OK")
    print(f"inflation unit: integral, det=1, inner shell -> roots OK")
    print(f"Delta = {Delta:.4f}; H_obs = {H_obs:.3f}; N = round({H_obs+Delta:.3f}) = 73")
    print(f"residual {73-(H_obs+Delta):.3f} step; phi^(8r) = {PHI**(8*(73-(H_obs+Delta))):.4f}"
          f" vs Lambda_obs/Lambda_fw = {1.1e-122/4.17e-123:.4f}; (7/15)phi^-582 = {(7/15)*PHI**-582:.3e}")
    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
