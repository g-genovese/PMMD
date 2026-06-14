import numpy as np
from scipy.optimize import brentq
# Koide |c|=1/sqrt2  <=>  Q=2/3  <=>  45 deg  <=>  equipartition (uniform power = structured power)
def koide_from_c(c, delta=2/9):
    a=np.abs(np.array([1+2*c*np.cos(delta+2*np.pi*k/3) for k in range(3)]))
    u=np.ones(3)/np.sqrt(3)
    return np.sum(a**2)/np.sum(a)**2, (a@u)**2/(a@a), (a@u)**2, a@a-(a@u)**2
for c in [0.0,0.5,1/np.sqrt(2),0.9]:
    Q,cos2,U,St=koide_from_c(c); print(f'|c|={c:.4f}: Q={Q:.4f} cos2={cos2:.4f} uniform/structured={U/St if St>1e-9 else float(\"inf\"):.3f}')
# observed leptons
m=np.array([0.51099895,105.6583755,1776.86]); a=np.sqrt(m); u=np.ones(3)/np.sqrt(3)
print(f'leptons: Q={np.sum(m)/np.sum(a)**2:.5f} theta={np.degrees(np.arccos(np.sqrt((a@u)**2/(a@a)))):.2f}deg |c|={np.sqrt((((a@a)/(a@u)**2)-1)/2):.4f}')
# genericity: Gaussian loop profile, |c| = normalized Z3-adjacent overlap
th=np.linspace(0,2*np.pi,2000,endpoint=False)
wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
cg=lambda s:np.sum(np.exp(-wrap(th)**2/(2*s**2))*np.exp(-wrap(th-2*np.pi/3)**2/(2*s**2)))/np.sum(np.exp(-wrap(th)**2/(2*s**2))**2)
print(f'Gaussian |c|=1/sqrt2 at sigma={brentq(lambda s:cg(s)-1/np.sqrt(2),0.5,5):.3f} rad (spacing 2pi/3={2*np.pi/3:.3f}) -> fine-tuned, non-generic')
