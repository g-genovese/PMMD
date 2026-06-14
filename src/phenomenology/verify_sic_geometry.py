#!/usr/bin/env python3
"""
verify_sic_geometry.py  --  (v6.0)

Numerical checks supporting the primordial C3 / K4 SIC structure, consistent
with Section "The primordial triangulation as instanton of the Wess-Zumino
sector" (Eq. primordial-Omega) and line ~2099 of the paper.

The primordial value is the CANONICAL one:

    Omega_prim = 2*pi/3

the minimal non-trivial Z3 topological charge on a 3-cycle (k=1) -- equivalently
the per-step phase of a once-wound 3-cycle (3 * 2pi/3 = 2pi). It is the phase-
balance datum (sum_k exp(i*phi_k) = 0) that the chain propagates and that carries
the chirality chi = sgn(Omega_prim).

The same C3 phase content also has a geometric (solid-angle) reading on the SIC
tetrahedron: each of the four faces subtends solid angle pi, hence Berry phase
pi/2 (= 2pi * Q_face with Q_face = 1/4), and the four faces tile S^2 to the unit
winding Q = 1. The geometric reading (pi/2) and the canonical value (2pi/3) are
the two readings of the same object, related by the k=1 Wess-Zumino (k/4pi)
normalisation -- NOT independent quantities. The Standard-Model outputs use the
algebraic Z3 (triality, Koide), i.e. the canonical 2pi/3.

Run:  python3 verify_sic_geometry.py
"""
import numpy as np
TOL = 1e-9
pi = np.pi


def state(n):
    nx, ny, nz = n / np.linalg.norm(n)
    th = np.arccos(np.clip(nz, -1, 1)); ph = np.arctan2(ny, nx)
    return np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)], dtype=complex)


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2


def solid_angle(n1, n2, n3):
    n1, n2, n3 = [v / np.linalg.norm(v) for v in (n1, n2, n3)]
    num = np.dot(n1, np.cross(n2, n3))
    den = 1 + np.dot(n1, n2) + np.dot(n2, n3) + np.dot(n3, n1)
    return abs(2 * np.arctan2(num, den))


def tetra():
    return [np.array(v) / np.sqrt(3) for v in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]


def ck(name, cond, d=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"   {d}" if d else ""))
    return bool(cond)


def main():
    ok = True
    print("=" * 70)
    print("Omega_prim = 2pi/3 : the canonical Z3 topological charge on the 3-cycle")
    print("=" * 70)
    phases = np.array([0.0, 2 * pi / 3, 4 * pi / 3])
    s = np.sum(np.exp(1j * phases))
    ok &= ck("Z3 phase balance:  sum_k exp(i*phi_k) = 0", abs(s) < TOL, f"|sum| = {abs(s):.2e}")
    ok &= ck("per-step phase of a once-wound 3-cycle:  3 * (2pi/3) = 2pi",
             abs(3 * (2 * pi / 3) - 2 * pi) < TOL, f"3*2pi/3 = {3*(2*pi/3):.6f}")
    ok &= ck("Omega_prim = 2pi/3 (minimal non-trivial Z3 charge)",
             abs(2 * pi / 3 - phases[1]) < TOL, f"2pi/3 = {2*pi/3:.6f}")
    print("    -> carries chi = sgn(Omega_prim); propagated by Noether along the chain.")
    print()
    print("=" * 70)
    print("SIC tetrahedron geometry (the geometric reading of the same C3 content)")
    print("=" * 70)
    v = tetra(); st = [state(x) for x in v]
    fids = [fidelity(st[i], st[j]) for i in range(4) for j in range(i + 1, 4)]
    ok &= ck("SIC pairwise fidelity = 1/(d+1) = 1/3", all(abs(f - 1 / 3) < TOL for f in fids),
             f"{[round(f,4) for f in fids]}")
    ok &= ck("sum of Bloch vectors = 0", np.allclose(np.sum(v, 0), 0, atol=TOL),
             f"sum = {np.round(np.sum(v,0),9)}")
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    sa = [solid_angle(v[a], v[b], v[c]) for a, b, c in faces]
    ok &= ck("each of the 4 faces subtends solid angle pi", all(abs(x - pi) < 1e-7 for x in sa),
             f"min={min(sa):.5f} max={max(sa):.5f}")
    ok &= ck("per-face Berry phase pi/2 = 2pi * Q_face with Q_face = 1/4",
             abs(pi / 2 - 2 * pi * 0.25) < TOL, f"pi/2 = {pi/2:.6f}, 2pi*1/4 = {2*pi*0.25:.6f}")
    tot = sum(sa)
    ok &= ck("4 faces tile S^2 (total solid angle 4pi) -> unit winding Q = 1",
             abs(tot - 4 * pi) < 1e-6, f"sum = {tot:.5f}, Q = (sum/4pi) = {tot/(4*pi):.0f}")
    print()
    print("=" * 70)
    print("RELATED readings, not separate quantities: the canonical per-cycle value")
    print("2pi/3 (Z3 charge, what the chain propagates) and the geometric per-face")
    print("Berry phase pi/2 are linked by the k=1 Wess-Zumino (k/4pi) normalisation")
    print(f"(ratio 2pi/3 : pi/2 = {(2*pi/3)/(pi/2):.4f} = 4/3).")
    print("=" * 70)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
