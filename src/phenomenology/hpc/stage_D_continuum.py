#!/usr/bin/env python3
"""STAGE D - continuum limit of |c| by finite-size / lattice-spacing extrapolation.
Reads (a, |c|) pairs from the per-lattice runs (a = box/L is the lattice spacing) and
fits |c|(a) = |c|_0 + k*a^2, reporting the continuum value |c|_0 and comparing to 1/sqrt(2).
This extrapolation is what makes the result trustworthy; a single lattice is NOT the answer."""
import sys, numpy as np
# input: lines "a c" on stdin or argv file
data=np.loadtxt(sys.argv[1]) if len(sys.argv)>1 else np.loadtxt(sys.stdin)
a=data[:,0]; c=data[:,1]
A=np.vstack([np.ones_like(a), a**2]).T
coef,*_=np.linalg.lstsq(A,c,rcond=None); c0,k=coef
print(f"continuum |c|_0 = {c0:.4f}  (slope {k:+.3f} a^2)   target 1/sqrt2 = {1/np.sqrt(2):.4f}")
print("=> Koide DERIVED" if abs(c0-1/np.sqrt(2))<0.02 else "=> |c|_0 != 1/sqrt2 within fit: Koide is an input at this order")
