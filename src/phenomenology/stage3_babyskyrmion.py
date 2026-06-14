import numpy as np
from scipy.integrate import solve_bvp
# Baby-Skyrmion hedgehog profile f(r), winding B. Solve Euler-Lagrange BVP, verify Derrick E4=E0.
def solve(B=1, kappa=1.0, mu=1.0, rmin=1e-2, rmax=15.0, N=400):
    def odes(r, y):
        f, fp = y
        s, s2 = np.sin(f), np.sin(2*f)
        A  = 2*r + 2*kappa**2*(B**2/r)*s**2
        Ap = 2 + 2*kappa**2*B**2*(s2*fp/r - s**2/r**2)
        rhs = (B**2/r)*s2 + kappa**2*(B**2/r)*s2*fp**2 + 2*mu**2*r*s - Ap*fp
        return np.vstack([fp, rhs/A])
    def bc(ya, yb): return np.array([ya[0]-np.pi, yb[0]-0.0])
    r = np.linspace(rmin, rmax, N)
    f0 = np.pi*(1-r/rmax)              # smooth initial guess pi->0
    y0 = np.vstack([f0, np.gradient(f0, r)])
    sol = solve_bvp(odes, bc, r, y0, max_nodes=20000, tol=1e-6)
    rr = np.linspace(rmin, rmax, 4000); f = sol.sol(rr)[0]; fp = sol.sol(rr)[1]
    s2f = np.sin(f)**2
    E2 = np.trapezoid(rr*fp**2 + (B**2/rr)*s2f, rr)
    E4 = np.trapezoid(kappa**2*(B**2/rr)*s2f*fp**2, rr)
    E0 = np.trapezoid(2*mu**2*rr*(1-np.cos(f)), rr)
    Rsize = rr[np.argmin(np.abs(f-np.pi/2))]
    return sol.success, E2, E4, E0, Rsize, rr, f

print("Baby-Skyrmion (kappa=mu=1): Derrick 2D predicts E4 = E0 at the minimum")
for B in [1,2,3]:
    ok,E2,E4,E0,Rs,rr,f = solve(B=B)
    print(f"  B={B}: ok={ok} E2={E2:.3f} E4={E4:.3f} E0={E0:.3f} | E4/E0={E4/E0:.4f} | size(f=pi/2)={Rs:.3f}")
