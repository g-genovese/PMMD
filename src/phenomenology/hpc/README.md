## Stage B (physical): emergent-gauge overlap  [stage_B_gauge_overlap.py]
The Yukawa-to-(n.sigma) coupling gives index 0 (vector-like, NO chiral zero modes) - verified
and discarded. The correct coupling is to the CP^1 EMERGENT U(1) gauge field (skyrmion = flux):
a charge-1 overlap Dirac then has index = Q, i.e. |Q| chiral zero modes of one chirality = the
generations. Verified: Q=-3 soliton -> exactly 3 zero modes, projected-g5 = [+1,+1,+1], index 3.
Koide Q_K from the zero-mode mass matrix (naive Higgs profile) ~ 0.43 (target 2/3): the generation
COUNT/chirality are topological and robust; the Koide VALUE depends on the (open) mass operator.
Dense sign() here is small/medium-L only; L >~ 128 and 3D need a matrix-free (Chebyshev/Zolotarev)
overlap -> HPC Stage 6 on Server C.
  usage: python3 stage_B_gauge_overlap.py --soliton sol.npy --m0 1.0

## One-command distributed scan (orchestrator)
Run `orchestrate.sh` from ONE Linux machine with passwordless SSH (keys) to all nodes.

Edit the CONFIG block at the top:
  * VENV="rs"  -> uses $HOME/rs (Linux) or %USERPROFILE%
s (Windows); auto-created if missing,
    with DEPS (jax scipy numpy) installed, on the first `--setup` run.
  * NODES: one line per machine: NAME|SSH_IP|USER|OS|TASKS|INTERNAL_10G_IP
    - TASKS is comma-separated; each task is `g` or `g@numanode`. A `g@N` task runs under
      `numactl --cpunodebind=N --membind=N` and is issued as its OWN ssh (required). Example:
      Server C (2 sockets) -> `1.5@0,1.6@1` runs two couplings, one per socket.
    - Windows DOMAIN user: write USER as `DOMAIN\user` (double backslash) or UPN `user@domain.local`.
    - INTERNAL_10G_IP: the servers' 10Gbps NIC; unused by the 2D scan (independent jobs),
      recorded for the later 3D MPI hostfile.

Commands:
    bash orchestrate.sh all --setup     # first time: deploy + create venv + install deps + run + collect
    bash orchestrate.sh all             # subsequent runs
    bash orchestrate.sh deploy|launch|collect   # individual phases
Results land in ./scan_results/ ; the script prints the g -> continuum |c|_0 table.
Per-machine driver is run_pilot.py (cross-platform). Manual single run: `python run_pilot.py <g>`.


## Pipeline
- stage_A_soliton.py   : bosonic B-charge baby-Skyrmion background (full Skyrme term). VERIFIED.
- stage_BC_overlap.py  : Wilson-Dirac zero modes in that background -> generation overlaps -> |c|.
- stage_D_continuum.py : a^2 extrapolation of |c| to the continuum (the trustworthy number).
- run_pilot.sh         : runs A->B->C->D over L = 96,128,160,192.

## How to run (2D pilot, single node)
    pip install jax scipy numpy
    bash run_pilot.sh        # ~hours; prints continuum |c|_0 vs 1/sqrt2

## CRITICAL checks before trusting any run
1. Stage A must print Berg-Luscher Q = -B (e.g. -3) AND E4/E0 ~ 1 (Derrick balance).
   Small/coarse lattices (L < 96) UNWIND (Q -> 0) and give meaningless |c|. Use L >= 96.
2. Stage B should show ~B near-zero Dirac eigenvalues (the generations). If the lowest |eig|
   are O(1) (not near 0), the background unwound (Q=0) -> rerun Stage A with larger L / smaller mu.

## Two physics decisions to validate (flagged in the paper)
- The fermion-soliton coupling in stage_BC (here the standard Yukawa g*n.sigma to an internal
  doublet). This is the model-sensitive Stage-6 choice; vary g and the coupling form.
- 2D pilot vs faithful 3D Hopfion (pi_3, Hopf-Wess-Zumino term, Vakulenko-Kapitanskii bound).
  If the 2D pilot gives |c|_0 ~ 1/sqrt2, escalate to 3D (replace stage_A by a 3D Hopfion
  relaxer on a 128^3-256^3 grid; multi-GPU/small cluster, weeks).

## Decisive output
The continuum |c|_0 from Stage D. If |c|_0 = 1/sqrt(2) within errors: Koide is DERIVED
(close Frontier 5). If not: Koide is an input at this order.
