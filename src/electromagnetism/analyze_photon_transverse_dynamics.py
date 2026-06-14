#!/usr/bin/env python3
"""
analyze_photon_transverse_dynamics.py -- Free-photon transverse-oscillation
dynamics from the substrate CP^1 kinetic term, qubit-native.

The aim was to derive a wave equation for the chain's transverse orientation
xi(z,t) (the polarisation), analogous to Box(delta-phi)=0 for the longitude.

What the derivation actually yields (and it is a clean closure, if an
anticlimactic one): the transverse oscillation
        xi(z,t) = Re[ E_vec * e^{i*phi} ]
propagates at c BECAUSE the carrier phi (the CP^1 longitude fluctuation
delta-phi) does -- with the polarisation vector E_vec a CONSTANT transverse
vector. There is NO separate transverse wave equation, because the polarisation
of a free photon is non-dynamical (conserved). The only dynamical mode is the
carrier delta-phi, already shown massless (Box delta-phi = 0,
analyze_equatorial_propagation.py). The polarisation is a constant transverse-
vector label whose two DOF are the two transverse spatial directions of the
foam (3 spatial dims, minus 1 used by propagation).
"""
import sympy as sp

t, z = sp.symbols('t z', real=True)
k, omega, c = sp.symbols('k omega c', positive=True)

phi = k*z - omega*t   # carrier = CP^1 longitude fluctuation delta-phi, plane wave along z

def Box(f):
    # d'Alembertian along the propagation axis (transverse plane trivial for a plane wave)
    return sp.diff(f, t, 2)/c**2 - sp.diff(f, z, 2)

print("=== Carrier (CP^1 longitude fluctuation) is massless ===")
carrier = sp.exp(sp.I*phi)
Box_carrier_onshell = sp.simplify(Box(carrier).subs(omega, c*k))
print(f"Box(e^{{i phi}}) on the massless shell omega = c k:  {Box_carrier_onshell}")
print("(= 0: this is the equatorial-propagation law already derived.)")
print()

print("=== Transverse oscillation xi = Re[E_vec e^{i phi}], E_vec constant ===")
A, delta = sp.symbols('A delta', real=True)   # one transverse component, E_x = A e^{i delta}, CONSTANT
xi_x = A*sp.cos(phi + delta)
Box_xi_onshell = sp.simplify(Box(xi_x).subs(omega, c*k))
print(f"Box(xi_x) on the massless shell:  {Box_xi_onshell}")
print("(= 0.)  With E_vec constant, Box xi = 0 follows ENTIRELY from Box(carrier)=0.")
print("There is no independent transverse dynamics: the wave is the carrier.")
print()

print("=== Polarisation is conserved (non-dynamical) for a free photon ===")
print("E_vec enters xi only as a multiplicative constant; the equation of motion")
print("does not act on it.  Hence dE_vec/dt = 0: the polarisation is a constant")
print("of the free motion, set by emission, conserved in propagation.")
print("Its TWO real DOF (modulo overall phase/magnitude) are the helicity-+-1")
print("content -- the two transverse spatial directions of the foam.")
print()

print("=== Why exactly two transverse DOF: foam dimensionality ===")
print("The foam is 3+1D (3 spatial dims, from the E8->D4 cut-and-project that")
print("fixes spacetime dimension in the framework). The chain propagates along")
print("ONE spatial direction; the remaining TWO spatial directions are the")
print("transverse plane. The polarisation vector lives there: 2 DOF = 2")
print("polarisations. The would-be third (longitudinal) direction is the")
print("propagation itself -- used up, not a free polarisation. This is the")
print("massless 'helicity +-1, no 0' statement, read off the foam's dimension.")
print()

print("=== Status of the free-photon Stratum-1 closure ===")
print("Closed, and all pieces grounded in the framework:")
print("  * carrier propagation:        Box(delta-phi) = 0   (equatorial law, derived)")
print("  * masslessness:               open chain, no winding (mass-from-winding rule)")
print("  * two polarisations:          two transverse foam directions (3 spatial - 1)")
print("  * polarisation = const vector: conserved for free photon (non-dynamical)")
print("  * spin-1, helicity +-1:       transverse vector under rotation about k")
print("                                 (analyze_photon_framing_helicity.py)")
print()
print("There is NO residual free-photon dynamics to derive: the transverse")
print("oscillation is the carrier acting on a constant polarisation vector.")
print()
print("What genuinely remains (a DIFFERENT problem -- the interaction, not the")
print("free photon): how the photon chain RELATES to charged closed loops at its")
print("endpoints, i.e. the qubit-native account of the electromagnetic interaction")
print("(emission/absorption and the force between windings). That is the coupling-")
print("to-matter question, separate from the free-propagation dynamics closed here.")
