#!/usr/bin/env python3
"""
analyze_loop_higgs_photon_hierarchy.py

Documents the qubit-native synthesis built up in the v6.0 working session:

  The Bloch equator -> pole axis IS the mass hierarchy.

    equator (latitude 0)        : massless; equatorial/transverse excitation
                                  -> the PHOTON  (vector, spin 1, transverse front)
    increasing latitude          : heavier matter loops (e, mu, ... toward top)
    maximum latitude (the TOP)    : the foam's natural winding scale
    the polar/latitude mode       : the HIGGS  (scalar, spin 0, concentric front),
                                    which IS the latitude = the mass.

Mass = latitude (polar tilt away from the equator), NOT a separate "polar axis".

The loop's oscillations decompose by direction relative to the ring:
  - LONGITUDINAL  (phase circulates around the ring)         -> WINDING = charge
  - POLAR/LATITUDE (qubits tilt poleward, in phase)          -> SCALAR, spin 0,
                                                                concentric front
                                                                = the mass = HIGGS
  - EQUATORIAL TRANSVERSE (a wave runs round the ring,
        displacement perpendicular to the tangent)            -> VECTOR, spin 1,
                                                                transverse front,
                                                                massless = PHOTON

Status: the geometry (scalar=concentric=Higgs vs vector=transverse=photon;
mass=latitude) is sound. The TOP as the *maximum* loop at the foam winding,
and y_t ~ 1 as "the loop fills the foam winding", is a striking but UNDERIVED
resonance (Frontier).
"""
import numpy as np

print("=" * 70)
print("THE TOP-QUARK RESONANCE (why the top is special)")
print("=" * 70)
v, m_t, m_H, m_b = 246.0, 173.0, 125.0, 4.18     # GeV
y_t = np.sqrt(2) * m_t / v
y_b = np.sqrt(2) * m_b / v
print(f"  y_t = sqrt(2) m_t / v = {y_t:.3f}   (~ 1: the top SATURATES the coupling)")
print(f"  y_b = sqrt(2) m_b / v = {y_b:.4f}  (next-heaviest, already ~{y_t/y_b:.0f}x smaller)")
print(f"  m_t={m_t}, m_H={m_H}, v={v} GeV  --  one scale (electroweak).")
print("  The top is the ONLY fermion with O(1) coupling to the mass-giving sector.")
print("  Resonance with 'top = maximum loop at the foam winding': beyond the top")
print("  there is no matter loop, only the foam winding itself; y_t ~ 1 = the")
print("  loop has grown to fill the foam winding.  (SUGGESTIVE, not derived.)")

print()
print("=" * 70)
print("BLOCH EQUATOR -> POLE  =  PHOTON -> HIGGS  =  MASSLESS -> MASSIVE")
print("=" * 70)
rows = [
    ("equator, latitude 0",      "massless",  "equatorial/transverse",  "PHOTON (vector, spin 1)"),
    ("increasing latitude",      "heavier",   "matter loops e,mu,...",   "(toward the top)"),
    ("max latitude = the top",   "heaviest",  "foam natural winding",    "TOP quark"),
    ("the polar/latitude mode",  "= mass",    "scalar, concentric",      "HIGGS (scalar, spin 0)"),
]
print(f"  {'Bloch position':<26}{'mass':<11}{'character':<24}{'object'}")
for a, b, c, d in rows:
    print(f"  {a:<26}{b:<11}{c:<24}{d}")

print()
print("=" * 70)
print("MODE / SPIN / FRONT CORRESPONDENCE ON THE LOOP")
print("=" * 70)
print("  Decompose the loop's qubit oscillations by direction vs the ring:")
print("   LONGITUDINAL (phase circulates round the ring) ............ WINDING = charge")
print("   POLAR/LATITUDE (qubits tilt poleward, in phase) .......... SCALAR, spin 0,")
print("        no preferred spatial direction -> radiates spherically = CONCENTRIC")
print("        front. IS the latitude = the mass.  ->  HIGGS")
print("   EQUATORIAL TRANSVERSE (wave runs round the ring, ......... VECTOR, spin 1,")
print("        displacement perpendicular to the tangent)            TRANSVERSE front,")
print("        has a direction -> massless; detaches tangentially -> PHOTON")
print()
print("  Exact: scalar <-> isotropic/concentric front;  vector <-> transverse front.")
print("  A light loop (near the equator) is transverse in character; climbing in")
print("  latitude toward the top it becomes polar/concentric -- the front goes")
print("  transverse -> concentric as the wave fills the loop.")

print()
print("=" * 70)
print("INSIDE THE LOOP: ONE WAVE OR A PACKET?  HELICAL?")
print("=" * 70)
print("  A definite eigenstate (definite mass/energy/ang.mom.) = ONE circulating")
print("  wave.  A localised or transitioning loop = a PACKET (superposition).")
print("  (Emission was a BEAT of two eigenstates -- a two-component packet.)")
print("  Helical?  The EQUATORIAL TRANSVERSE modes ARE helical -- the transverse")
print("  framing rotates as the wave circulates (that rotation = the twist = the")
print("  would-be photon helicity).  The POLAR/LATITUDE (Higgs) mode is NOT helical")
print("  -- it tilts in phase, no rotation.  The longitudinal phase circulation is")
print("  the winding itself.")

print()
print("=" * 70)
print("STATUS")
print("=" * 70)
print("  SOUND (geometry): scalar=concentric=Higgs vs vector=transverse=photon;")
print("    mass=latitude; equator->pole as the massless->massive hierarchy.")
print("  SUGGESTIVE (not derived): the top as the MAXIMUM loop at the foam's")
print("    natural winding, and y_t ~ 1 as 'the loop fills the foam winding'.")
print("    The numerical resonance (y_t ~ 1, unique among fermions) is real; the")
print("    framework derivation that the top sits exactly there is OPEN.")
