#!/usr/bin/env python3
"""Cross-platform per-machine driver (Linux + Windows). Runs the 2D pilot for ONE coupling g:
sweeps L, calls Stage A (soliton) and Stage B/C (overlap), extrapolates |c| to the continuum.
Writes cvals_g<g>.dat and result_g<g>.txt. Independent job -> one g per machine.

  usage:  python run_pilot.py <g> [--quick]
"""
import sys, subprocess, numpy as np, os
PY = sys.executable                      # same interpreter on Linux/Windows
HERE = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print(">", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)

def main():
    g = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    quick = "--quick" in sys.argv
    Ls   = [48, 64] if quick else [96, 128, 160, 192]
    steps = 2000 if quick else 30000
    BOX, B = 8.0, 3
    tag = f"g{g}"
    pts = []
    for L in Ls:
        sol = os.path.join(HERE, f"sol_{tag}_L{L}.npy")
        a = run([PY, "stage_A_soliton.py", "--L", str(L), "--B", str(B), "--mu", "0.3",
                 "--kappa", "1.0", "--steps", str(steps), "--box", str(BOX), "--out", sol])
        salines = a.stdout.strip().splitlines() if a.stdout else []
        diag = next((ln for ln in salines if "Berg-Luscher" in ln),
                    salines[-1] if salines else (a.stderr[-300:] if a.stderr else ""))
        print("  [A]", diag, flush=True)   # <-- shows Berg-Luscher Q and Derrick E4/E0
        bc = run([PY, "stage_BC_overlap.py", "--soliton", sol, "--g", str(g),
                  "--nmodes", "8", "--B", str(B)])
        c = None
        for line in bc.stdout.splitlines():
            if "|c| (mass matrix)" in line:           # primary value recorded for the sweep
                c = float(line.split("=")[1].split()[0])
        if c is None:
            print("  WARN: no |c| parsed (background may have unwound); skipping L=", L, flush=True)
            continue
        a_spacing = BOX * 2 / L
        pts.append((a_spacing, c))
        print(f"  L={L}: a={a_spacing:.4f}  |c|={c:.4f}", flush=True)
    with open(os.path.join(HERE, f"cvals_{tag}.dat"), "w") as f:
        for asp, c in pts: f.write(f"{asp} {c}\n")
    # continuum extrapolation |c|(a)=c0+k a^2
    out = [f"g={g}"]
    if len(pts) >= 2:
        A = np.array([[1, p[0]**2] for p in pts]); y = np.array([p[1] for p in pts])
        (c0, k), *_ = np.linalg.lstsq(A, y, rcond=None)
        verdict = "Koide DERIVED" if abs(c0 - 1/np.sqrt(2)) < 0.02 else "|c|_0 != 1/sqrt2 at this order"
        out.append(f"continuum |c|_0 = {c0:.4f}  (target {1/np.sqrt(2):.4f})  slope {k:+.3f} a^2  -> {verdict}")
    else:
        out.append("insufficient points for extrapolation (check soliton stability: Q must be -3)")
    res = "\n".join(out)
    print(res, flush=True)
    with open(os.path.join(HERE, f"result_{tag}.txt"), "w") as f: f.write(res + "\n")

if __name__ == "__main__":
    main()
