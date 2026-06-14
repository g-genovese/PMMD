#!/usr/bin/env python3
"""
analyze_loop_wave_emission.py -- Three refinements (the user's):
 (1) the charged loop is ALREADY a wave -- a circulating matter-wave with
     winding, not a static object with a wave added on top;
 (2) the emitted wavefront rotates from circulating (front-normal tangent) to
     concentric/spherical (front-normal radial) as it becomes free radiation;
 (3) what detaches is not 'an extra wave' but the UNBALANCED component that
     appears when two loop-eigenstates beat together (a transition) -- the
     component with a distinct net circulation, carrying angular momentum away
     as the photon's helicity. Photon spin-1 => the circulation difference is
     +-1 => the selection rule Delta-m = +-1.
"""
import sympy as sp

phi, t, mi, mf, Ei, Ef, hbar, a, b = sp.symbols('phi t m_i m_f E_i E_f hbar a b', real=True)

print("=== (1) The loop is already a wave (a circulating eigenstate) ===")
print("A loop-eigenstate is a circulating matter-wave of definite angular")
print("momentum m:  psi_m = exp(i m phi - i E t/hbar).  The particle IS a wave;")
print("'a wave on the loop' was redundant. An 'excited' loop is simply a higher-m")
print("(higher-energy) circulating eigenstate of the same loop.")
print()

print("=== (2) A single eigenstate is stationary -> does not radiate ===")
psi_single = sp.exp(sp.I*(mi*phi) - sp.I*Ei*t/hbar)
rho_single = sp.simplify(psi_single*sp.conjugate(psi_single))
print(f"  |psi_m|^2 = {rho_single}   (steady -- no oscillating, circulating pattern)")
print("  A steady pattern has no part to shed: a single eigenstate does not")
print("  radiate. (Framework form of 'stationary states do not radiate'.)")
print()

print("=== (3) A transition = beat of two eigenstates = the radiating part ===")
psi = a*sp.exp(sp.I*(mi*phi) - sp.I*Ei*t/hbar) + b*sp.exp(sp.I*(mf*phi) - sp.I*Ef*t/hbar)
rho = sp.expand(psi*sp.conjugate(psi))
print("  Superpose initial (m_i, E_i) and final (m_f, E_f). |psi|^2 develops a")
print("  CROSS term that oscillates and circulates:")
print("    cross ~ 2 a b cos((m_i - m_f) phi - (E_i - E_f) t / hbar)")
print("  This unbalanced, circulating beat is the part that detaches and flies")
print("  off. It oscillates at omega = (E_i - E_f)/hbar (the photon frequency)")
print("  and circulates with net angular momentum (m_i - m_f).")
print()
print("  For a single photon (spin 1, helicity +-1) the shed circulation is +-1:")
print("        m_i - m_f = +-1   <=>   the selection rule Delta-m = +-1.")
print("  The 'distinctly different winding' that detaches is exactly this")
print("  one-unit circulation difference -- the photon's helicity, carried off.")
print("  The topological winding (the charge) is untouched and stays on the loop.")
print()

print("=== (4) Wavefront rotates from circulating to concentric ===")
print("  Circulating beat on the loop: travels along the TANGENT, wavefronts")
print("  perpendicular to the tangent (front-normal = tangent).")
print("  Free radiation: travels RADIALLY outward, wavefronts perpendicular to")
print("  the radial direction -- concentric spherical shells about the emission")
print("  point (front-normal = radial).")
print("  As the shed beat goes bound -> free, the front-normal rotates from")
print("  tangent to radial (~90 deg): the front 'rotates and becomes concentric'.")
print("  (Speculative: the foam's own natural winding -- the coherently-propagated")
print("   Bloch direction -- may be what turns the circulating front outward at")
print("   the detachment point. Suggested, not derived.)")
print()

print("=== Payoff: stability AND selection rule, together, geometrically ===")
print(" * Stability: a single (lowest) eigenstate is steady -> no beat -> no shed")
print("   component -> no radiation; and there is nothing lower to beat with.")
print("   The atom is stable because one circulating wave alone has nothing to")
print("   shed.")
print(" * Selection rule: radiation is the beat of two eigenstates; the shed")
print("   circulating component carries Delta-m = +-1 = the photon's spin.")
print("Both follow from: the loop is a circulating wave; radiation is the beat")
print("between two such waves, the unbalanced circulating part flying off tangent")
print("with its front rotating to concentric. No field, no coupling -- only waves,")
print("their beat, and geometry. The RATE still needs the strength (alpha, Frontier 6).")
