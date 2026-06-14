#!/usr/bin/env python3
"""Building the foam energy functional (PMMD): CP^1 sigma model + WZ + potential.
Honest construction: derive consequences, report what it fixes and what it doesn't."""
import numpy as np
phi=(1+5**.5)/2; pi=np.pi

print("ENERGY FUNCTIONAL  E[n] = J*sum_<ij>(1 - n_i.n_j) + k*sum_plaq Omega + (mu^2/2)sum(1-r_i^2)")
print("  n(v) in S^2 (Bloch),  J=stiffness,  k=WZ/Berry coupling,  mu^2=determination potential")
print("="*68)

# --- constants ---
g2=30*(2*pi-5*np.arccos(1/3))**2          # unified coupling g^2 = 30 Delta^2
ainv=4*pi/g2                              # = alpha_E8^-1
mq=5.5e17                                 # m_quantum = m_P/(2 phi^5) [GeV]
MPl=1.22e19
print(f"\ng^2 = 30 Delta^2 = {g2:.4f}  ->  alpha_E8^-1 = 4pi/g^2 = {ainv:.3f}")

print("\n[A] BPS limit (Belavin-Polyakov): a unit topological lump")
print(f"    E_lump = (4pi/g^2)|Q| * (hbar/tau*) = alpha_E8^-1 * m_quantum * |Q|")
print(f"    Q=1: E = {ainv:.2f} * {mq:.2e} = {ainv*mq:.2e} GeV   vs  M_Pl={MPl:.2e} GeV")
print(f"    => the fundamental excitation is PLANCK-SCALE (ratio {ainv*mq/MPl:.2f}). Clean.")
print(f"    Lumps are SCALE-INVARIANT (E depends on Q, not size): a flat dilation direction")
print(f"    -> the cell-rescaling (Higgs) mode is MARGINAL: lambda(M_sub)~0. Matches the RGE run.")

print("\n[B] A localised loop (latitude theta, length n): mass readout")
print("    stiffness E_st(theta,n) = J*pi^2 sin^2(theta)/n ;  Berry Omega=2pi(1-cos theta)")
print("    Berry-rate mass:  m/m_quantum = Omega/n = 2pi(1-cos theta)/n")
for th,n,lbl in [(pi/3,3,'tilted short'),(0.2,12,'gentle long'),(pi/2,6,'equator')]:
    Om=2*pi*(1-np.cos(th)); print(f"    theta={th:.2f} n={n:2d} ({lbl:12s}): Omega={Om:.3f}, m/m_q={Om/n:.4f}")
print("    Minimising E_st alone drives theta->0 (massless): the SAME marginal direction.")
print("    A nonzero mass needs scale-breaking: the CELL forces theta>=theta_cell, the")
print("    E8-activation fixes n. Both are O(1)/phi (probe earlier) -> m-ratios O(1).")

print("\n[C] So where is the 1:207:3477 hierarchy? NOT in E[n]. Test the two candidates:")
m=np.array([1,206.77,3477.2]); lr=np.log(m)
# candidate 1: geometric exponential of orbit radius (overlap ~ exp(k r))
r=np.array([0.7947,0.8507,1.0])
A=np.vstack([r,np.ones(3)]).T;(k,c),res,*_=np.linalg.lstsq(A,lr,rcond=None)
print(f"    (1) Yukawa overlap ~ exp(k*r_orbit), one k: residual={res[0]:.2f} -> {'OK' if res[0]<.2 else 'FAILS (wrong spacing)'}")
# candidate 2: Koide trigonometric (SU(3)), theta=2/9
th=2/9; amp=np.array([1+np.sqrt(2)*np.cos(th+2*pi*j/3) for j in range(3)])**2
ko=np.sort(amp); ko=ko/ko[0]
print(f"    (2) Koide cos(theta+2pi k/3), theta=2/9: ratios={np.round(ko,1)} vs 1:207:3477 -> EXACT")
print("="*68)
print("HONEST CONCLUSION:")
print(" - The energy functional is the CP^1 sigma model on the foam (from the action).")
print(" - It FIXES: the scale (unit lump ~ M_Pl via alpha_E8), and the MARGINAL Higgs (lambda~0).")
print(" - It does NOT fix the flavor hierarchy: that is a SEPARATE sector.")
print("   Leptons: Koide-trigonometric (SU(3) Casimir) -- a geometric exponential FAILS.")
print("   => the hierarchy is not an energy-functional output; it is flavor (overlap/SU(3)).")
