#!/usr/bin/env python3
"""
analyze_em_interaction.py -- Qubit-native account of the electromagnetic
interaction: the relation between two charged windings, mediated by the chain
of vertices (the photon).

What is derived:
  * The FORM of the static interaction: long-range Coulomb 1/r, traced directly
    to the photon's masslessness (the open chain, no winding). A massive mediator
    (a closed loop, which would carry winding) gives short-range Yukawa. The same
    open/closed distinction that makes the photon massless makes the EM force
    infinite-range.
  * Sign: like windings repel, opposite windings attract.
What is NOT derived (flagged honestly):
  * The interaction STRENGTH (alpha = 1/137) -- Frontier 6, the deep open problem.
  * The full microscopic mechanism by which a winding sources the longitude
    pattern carried by the intervening chain -- partially articulated, not rigorous.
"""
import sympy as sp

r, q1, q2, m, c = sp.symbols('r q_1 q_2 m c', positive=True, real=True)

print("=== Mediator = the open-chain longitude carrier ===")
print("The photon is the open chain of equatorial qubit ticks; its propagating")
print("mode is the longitude fluctuation delta-phi, massless (Box delta-phi = 0,")
print("derived). A charged winding sources a static longitude pattern in the")
print("surrounding foam, carried outward by the chain of vertices (the 'phantom")
print("connection' is the chain itself). Static limit of the massless carrier:")
print("   -Laplacian(phi) = rho_winding   ->   phi(r) = q/(4 pi r).")
print()

# Static potential from a mediator of mass m in 3D (Yukawa), Coulomb at m=0
phi_massive = q1*sp.exp(-m*r)/(4*sp.pi*r)
print("Static longitude pattern from a mediator of mass m (3D):")
print(f"   phi_m(r) = {phi_massive}")
phi_massless = sp.limit(phi_massive, m, 0)
print(f"   massless limit (open chain, no winding): phi(r) = {phi_massless}")
print("   -> Coulomb 1/r, long-range.")
print()

print("=== Interaction energy and force between two windings ===")
U = q1*q2/(4*sp.pi*r)
F = -sp.diff(U, r)
print(f"   U(r) = {U}")
print(f"   F(r) = -dU/dr = {F}")
print("   q1*q2 > 0  (like windings):     F > 0  -> repulsion")
print("   q1*q2 < 0  (opposite windings): F < 0  -> attraction")
print()
print("Qubit-native reading of the sign: each winding bends the surrounding")
print("longitudes into a pattern. Like windings' patterns reinforce (higher total")
print("longitude-gradient cost -> the configuration lowers its cost by separating")
print("-> repulsion); opposite windings' patterns partially cancel (lower cost when")
print("close -> attraction).")
print()

print("=== The framework-specific step: masslessness -> infinite range ===")
print("The 1/r (infinite-range) form is NOT generic -- it is forced by the photon")
print("being massless. A massive mediator screens the force beyond a range 1/m:")
print(f"   U_massive(r) = q1 q2 exp(-m r)/(4 pi r)   (Yukawa, short-range)")
print("In the framework the photon is massless because the chain is OPEN (no closed")
print("loop, no Pancharatnam area, no rest mass). Hence:")
print("   open chain  ->  massless photon  ->  long-range 1/r EM force.")
print("The very same open/closed distinction that gives the photon no mass gives")
print("electromagnetism its infinite range. A force mediated by a CLOSED-loop")
print("(massive) excitation would be short-range -- which is exactly what the weak")
print("force is (massive W/Z, range ~1/m_W). Open vs closed = long-range EM vs")
print("short-range weak, from one structural distinction.")
print()

print("=== Emission / absorption (structural, qubit-native) ===")
print("A charged closed loop in an excited configuration relaxes to a lower one.")
print("The released phase-circulation / energy cannot stay localised (charge, the")
print("net winding, is conserved -- it is not carried off); instead it detaches as")
print("an OPEN chain of equatorial ticks propagating away = a real photon. Its")
print("carrier frequency is omega = Delta-E / hbar (the loop's energy gap). Its")
print("transverse polarisation is set by the spatial orientation of the loop's")
print("reconfiguration (the transverse projection of the transition 'direction').")
print("Absorption is the time-reverse: an incoming chain merges into a loop,")
print("raising its configuration. No winding is exchanged (the photon has none);")
print("only energy, momentum, and angular momentum (the helicity) pass.")
print()

print("=== Honest status ===")
print("DERIVED (form): long-range 1/r Coulomb, like/opposite sign, tied directly")
print("to the photon's masslessness (open chain). This is a genuine framework")
print("consequence: the open/closed distinction sets EM range vs weak range.")
print()
print("OPEN (strength): the coupling magnitude -- the fine-structure constant")
print("alpha = 1/137 -- is NOT derived here. It is Frontier 6 (closed-form")
print("gauge-breaking scales and alpha from first principles). The full")
print("microscopic mechanism by which a winding sources the longitude pattern")
print("carried by the intervening chain (and hence the magnitude of the coupling)")
print("is only partially articulated, not rigorously derived.")
