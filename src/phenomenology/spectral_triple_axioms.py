import numpy as np
np.set_printoptions(precision=3, suppress=True)

# ============================================================
# Finite spectral triple axiom check for the framework's D_F
# Lepton sector, one generation, KO-dimension 6 (a la Chamseddine-Connes-Marcolli)
# Basis particles:    0=nuL 1=eL 2=nuR 3=eR
# Basis antiparticles:4=nuLc 5=eLc 6=nuRc 7=eRc   (J maps i<->i+4 with conjugation)
# ============================================================
N=8
# --- grading gamma: chirality (left=+1,right=-1) on particles, FLIPPED on antiparticles (KO-dim 6) ---
gamma = np.diag([+1,+1,-1,-1, -1,-1,+1,+1]).astype(complex)
# --- real structure J = S o (complex conjugation), S = particle<->antiparticle swap ---
S = np.zeros((N,N))
for i in range(4):
    S[i,i+4]=1; S[i+4,i]=1
# J psi := S @ conj(psi)   (antilinear)

# --- framework D_F: Dirac (Yukawa) masses + Majorana M_R on nuR ---
mnu, me, MR = 0.05e-9, 5.11e-4, 6e14   # representative values (GeV); axioms are value-independent
D = np.zeros((N,N), dtype=complex)
def sym(a,b,v): D[a,b]=v; D[b,a]=np.conj(v)
sym(0,2,mnu); sym(1,3,me)              # Dirac, particles
sym(4,6,np.conj(mnu)); sym(5,7,np.conj(me))  # Dirac, antiparticles (conjugate)
sym(2,6,MR)                            # Majorana: nuR <-> nuRc  (particle<->antiparticle)

def chk(name,cond): print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("KO-dimension 6 axioms (eps,eps',eps'')=(+1,+1,-1):")
chk("gamma^2 = I",                 np.allclose(gamma@gamma, np.eye(N)))
chk("gamma = gamma^dagger",        np.allclose(gamma, gamma.conj().T))
chk("D = D^dagger (self-adjoint)", np.allclose(D, D.conj().T))
chk("gamma D = - D gamma  (D odd)",np.allclose(gamma@D, -D@gamma))
chk("J^2 = +1            (eps=+1)", np.allclose(S@S, np.eye(N)))           # J^2 = S^2
chk("J gamma = -gamma J  (eps''=-1)",np.allclose(S@gamma, -gamma@S))        # S gamma = -gamma S
chk("J D = D J          (eps'=+1)", np.allclose(S@D.conj(), D@S))           # S conj(D) = D S

# isolate WHY the Majorana term needs KO-dim 6: drop the antiparticle grading flip -> KO-dim 0
gamma0 = np.diag([+1,+1,-1,-1, +1,+1,-1,-1]).astype(complex)  # same sign both sectors
print("\nCounterfactual: same-sign grading (KO-dim 0 style):")
chk("gamma0 D = -D gamma0 (Majorana would break oddness)", np.allclose(gamma0@D, -D@gamma0))

# ============================================================
# Order-one condition factorises through generation multiplicity:
#   D = C (x) X_block ,  a = I3 (x) A ,  b0 = I3 (x) B
#   [[D,a],b0] = C (x) [[X,A],B]   -> vanishes iff one-generation does
# C = circulant Koide generation matrix (framework's flavour structure)
# ============================================================
print("\nOrder-one factorisation through generation space (tensor identity):")
rng=np.random.default_rng(0)
c=0.842; C=np.array([[1+2*c,0,0],[0,1-c,0],[0,0,1-c]],dtype=complex)  # diag of circulant eigvals (Koide form)
# build a genuine 3x3 circulant (the actual flavour object) instead of its diagonal:
a0,a1,a2 = 1.0, 0.30, 0.30
C=np.array([[a0,a1,a2],[a2,a0,a1],[a1,a2,a0]],dtype=complex)          # circulant
X=rng.standard_normal((8,8))+1j*rng.standard_normal((8,8)); X=X+X.conj().T
A=rng.standard_normal((8,8))+1j*rng.standard_normal((8,8))
B=rng.standard_normal((8,8))+1j*rng.standard_normal((8,8))
I3=np.eye(3)
Dg=np.kron(C,X); ag=np.kron(I3,A); bg=np.kron(I3,B)
lhs=(Dg@ag-ag@Dg)@bg-bg@(Dg@ag-ag@Dg)
rhs=np.kron(C,(X@A-A@X)@B-B@(X@A-A@X))
chk("[[C(x)X, I3(x)A], I3(x)B] = C (x) [[X,A],B]", np.allclose(lhs,rhs))
chk("=> order-one(3 gen, circulant) holds iff order-one(1 gen) holds", np.allclose(lhs,rhs))
