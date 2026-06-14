#!/usr/bin/env python3
"""STAGE A - bosonic B-charge baby-Skyrmion background (CP^1, full Skyrme term).
JAX autodiff projected gradient flow with DERRICK EARLY-STOPPING: the lattice baby-Skyrmion
is metastable (the vacuum E=0 is lower), so a long flow tunnels to Q=0. We monitor the
topological charge Q and the Derrick ratio E4/E0 every `check` steps, keep the best config
with Q=-B nearest the Derrick balance E4=E0, and stop as soon as Q leaves the -B sector.
Saves that best soliton for Stage B. Scales to GPU (JAX_PLATFORM_NAME=gpu).

  usage: python3 stage_A_soliton.py --L 96 --B 3 --mu 0.3 --kappa 1.0 --steps 30000 --out sol.npy
"""
import argparse, numpy as np, jax, jax.numpy as jnp
from jax import grad, jit

def make_energy(kappa, mu):
    def energy(n):
        n = n/jnp.linalg.norm(n,axis=-1,keepdims=True)
        d1 = jnp.roll(n,-1,0)-n; d2 = jnp.roll(n,-1,1)-n
        E2 = jnp.sum(d1*d1+d2*d2)
        cr = jnp.cross(d1,d2); E4 = kappa**2*jnp.sum(cr*cr)
        E0 = mu**2*jnp.sum(1.0-n[...,2])
        return E2+E4+E0,(E2,E4,E0)
    return energy

def berg_luscher_Q(n):                       # vectorised geometric topological charge
    n=np.asarray(n); n=n/np.linalg.norm(n,axis=-1,keepdims=True)
    a=n[:-1,:-1]; b=n[1:,:-1]; c=n[1:,1:]; d=n[:-1,1:]
    def tri(p,q,r):
        num=np.sum(p*np.cross(q,r),-1); den=1+np.sum(p*q,-1)+np.sum(q*r,-1)+np.sum(r*p,-1)
        return 2*np.arctan2(num,den)
    return float((tri(a,b,c).sum()+tri(a,c,d).sum())/(4*np.pi))

def relax(L,B,kappa,mu,max_steps,lr,box,check=500):
    xs=jnp.linspace(-box,box,L); X,Y=jnp.meshgrid(xs,xs,indexing='ij')
    r=jnp.hypot(X,Y); phi=jnp.arctan2(Y,X); f=jnp.pi*jnp.exp(-r/(box/2.7))
    n=jnp.stack([jnp.sin(f)*jnp.cos(B*phi),jnp.sin(f)*jnp.sin(B*phi),jnp.cos(f)],-1)
    en=make_energy(kappa,mu); eg=jit(grad(lambda m:en(m)[0]))
    @jit
    def step(n):
        g=eg(n); g=g-jnp.sum(g*n,-1,keepdims=True)*n; n=n-lr*g
        return n/jnp.linalg.norm(n,axis=-1,keepdims=True)
    best=None; best_score=1e9; best_info=None
    for it in range(max_steps):
        n=step(n)
        if (it+1)%check==0:
            Q=berg_luscher_Q(n); (_,(E2,E4,E0))=en(n)
            E2,E4,E0=float(E2),float(E4),float(E0); ratio=E4/(E0+1e-12)
            if abs(Q-(-B))<0.3:                          # in the -B sector
                score=abs(ratio-1.0)                     # distance from Derrick balance
                if score<best_score:
                    best_score=score; best=np.array(n); best_info=(it+1,E2,E4,E0,ratio,Q)
            if abs(Q-(-B))>0.5 and best is not None:      # unwinding -> stop, keep best
                print(f"  unwind at it={it+1} (Q={Q:.2f}); stopping, returning best"); break
    return best, best_info

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for k,v,t in [("L",96,int),("B",3,int),("mu",0.3,float),("kappa",1.0,float),
                  ("steps",30000,int),("lr",0.01,float),("box",8.0,float),
                  ("check",500,int),("out","soliton.npy",str)]:
        ap.add_argument("--"+k,default=v,type=t)
    a=ap.parse_args()
    best,info=relax(a.L,a.B,a.kappa,a.mu,a.steps,a.lr,a.box,a.check)
    if best is None:
        print(f"DONE L={a.L} B={a.B}: NO Q=-{a.B} soliton found (collapsed). "
              f"Try larger --box / smaller --mu / larger L."); raise SystemExit(1)
    it,E2,E4,E0,ratio,Q=info
    print(f"DONE L={a.L} B={a.B}: E2={E2:.2f} E4={E4:.2f} E0={E0:.2f} "
          f"Derrick E4/E0={ratio:.3f} Berg-Luscher Q={Q:.3f} (captured at it={it})")
    np.save(a.out, best); print("saved", a.out)
