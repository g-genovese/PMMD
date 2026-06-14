#!/usr/bin/env python3
"""Do the quark Koide Q values run toward 2/3 at high scale? (one-loop SM Yukawa RGE)
Q is invariant under common rescaling -> depends only on Yukawa RATIOS."""
import numpy as np
pi=np.pi; v=246.0
def Q(m):
    m=np.array(m,float); return m.sum()/np.sqrt(m).sum()**2

# M_Z-scale MSbar running masses (GeV) -- standard approximate values
mu0={'u':1.27e-3,'c':0.62,'t':171.0,'d':2.90e-3,'s':0.055,'b':2.90}
y={k:np.sqrt(2)*m/v for k,m in mu0.items()}
g3,g2,g1=1.22,0.652,np.sqrt(5/3)*0.357   # GUT-normalised g1

def rhs(y,g3,g2,g1):
    yt,yc,yu,yb,ys,yd=[y[k] for k in ['t','c','u','b','s','d']]
    Y2=3*(yt**2+yc**2+yu**2)+3*(yb**2+ys**2+yd**2)   # + tiny leptons
    k=1/(16*pi**2)
    cu=8*g3**2+(9/4)*g2**2+(17/20)*g1**2
    cd=8*g3**2+(9/4)*g2**2+(1/4)*g1**2
    dy={}
    for f,yf,yp in [('t',yt,yb),('c',yc,ys),('u',yu,yd)]:
        dy[f]=k*yf*((3/2)*(yf**2-yp**2)+Y2-cu)
    for f,yf,yp in [('b',yb,yt),('s',ys,yc),('d',yd,yu)]:
        dy[f]=k*yf*((3/2)*(yf**2-yp**2)+Y2-cd)
    dg3=k*(-7)*g3**3; dg2=k*(-19/6)*g2**3; dg1=k*(41/10)*g1**3
    return dy,dg3,dg2,dg1

MZ=91.19; scales={'M_Z':MZ,'1e10':1e10,'M_GUT~2e16':2e16,'M_sub~5.5e17':5.5e17}
targets=sorted(scales.values()); tmax=np.log(targets[-1]/MZ)
N=400000; dt=tmax/N; t=0; out={}
next=0
prog=sorted(scales.items(),key=lambda kv:kv[1])
def snap(): return ([y['u'],y['c'],y['t']],[y['d'],y['s'],y['b']])
for i in range(N+1):
    mu=MZ*np.exp(t)
    while next<len(prog) and mu>=prog[next][1]:
        u,d=snap(); out[prog[next][0]]=(Q(u),Q(d))
        next+=1
    dy,dg3,dg2,dg1=rhs(y,g3,g2,g1)
    for f in y: y[f]=y[f]+dt*dy[f]
    g3+=dt*dg3; g2+=dt*dg2; g1+=dt*dg1; t+=dt
print(f"{'scale':16}{'Q_up':>10}{'Q_down':>10}   (Koide target 2/3=0.6667)")
for name,_ in prog:
    if name in out:
        qu,qd=out[name]; print(f"{name:16}{qu:10.4f}{qd:10.4f}")
print("\n(Q from sqrt(m)=y; using y^2 as mass. Q invariant under common rescaling.)")
print("Honest read: report the TREND toward/away from 0.6667.")
