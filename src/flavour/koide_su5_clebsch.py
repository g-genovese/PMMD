#!/usr/bin/env python3
"""SU(5) origin of the quark Koide deviation: down amplitude from leptons via
the fixed 10*5bar Georgi-Jarlskog Clebsch; up (10*10) as residue. (PMMD v6.0)"""
import numpy as np
def Q(m): m=np.array(m,float); return m.sum()/np.sqrt(m).sum()**2
def r_of(Q): return np.sqrt(2*(3*Q-1))  # Q=(1+r^2/2)/3
mlep=[0.51099895,105.6584,1776.86]; me,mm,mt=mlep
print(f"leptons  : Q={Q(mlep):.4f} r={r_of(Q(mlep)):.4f} (sqrt2={np.sqrt(2):.4f})")
print(f"down naive(m_d=m_e)         : Q={Q([me,mm,mt]):.4f}  (=lepton, WRONG)")
print(f"down Georgi-Jarlskog(45-H)  : Q={Q([3*me,mm/3,mt]):.4f} r={r_of(Q([3*me,mm/3,mt])):.4f}")
print(f"down observed (M_Z)         : Q={Q([2.9e-3,0.055,2.9]):.4f}")
print(f"up observed (10*10, residue): Q={Q([2.2e-3,1.27,172.8]):.4f} r={r_of(Q([2.2e-3,1.27,172.8])):.4f}")
print("ordering r_up>r_down>sqrt2:", r_of(Q([2.2e-3,1.27,172.8]))>r_of(Q([3*me,mm/3,mt]))>np.sqrt(2))
