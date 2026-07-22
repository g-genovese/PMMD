# PMMD — Maximal Mutual Determination

**From a Qubit System to the Standard Model via E₈ Foam**

Author: **Gianluca Genovese**

This repository accompanies the PMMD framework paper. It contains the full LaTeX
source and compiled PDF of the manuscript together with **all the code** used to
produce the framework's numerical results — percolation criticality, the
Bombelli–Sorkin dimension `d_BS`, the 3-fold merger statistics, the V₁₁₂/Koide
phenomenology, the bond-bias RG runs, the τ-demarcation / dark-sector statistics,
and the E₈→H₄ cut-and-project geometry.

> **Status.**<br>
> This is **v7.0, a work in progress.**<br>
> v6.0 is the latest version previously archived on Zenodo; v7.0 is the current release.<br>
> **New in v7.0:** the discrete relational action made **operative** (δS = 0 as
> the working principle; D = 3/4 derived by granularity-forced self-counting
> stationarity) and the **flavour sector closed through it**: the full CKM from
> the electroweak gateway with every input derived at leading order (the
> excursion-curvature gaps Δ_p = 3·M_y(ρ_p), Δ₅ = 4/3 exact in form; a
> zero-free-parameter CKM at the 3–15% level, χ²/dof = 0.55 at the data-pinned
> refinements; CP structurally necessary, branch data-sealed); the neutrino
> oscillation sector at **zero parameters** from derived constants (χ² = 1.42,
> sin δ_CP = −1 exact); colour-flux census laws from 1.46×10¹⁰ lattice loops
> (exact quantisation Φ ∈ (2/√3)ℤ, κ = 4π/(3√3), unit-diffusivity ⟨q²⟩ ≈ A at
> 29.6σ); three exact flavour–gauge bridges and the (α_s(M_Z), √σ) c-dictionary;
> scheme covariance (the golden inflation as renormalisation group); and the
> honest ledger of **44 adjudications, 11 self-inflicted**. Also included: the
> four v6.1 working-label sharpenings (self-duality via Mordell; the intrinsic
> Stage 4–5 reformulation; the two coherence thresholds; the Koide value
> re-grounded as maximal shared indetermination). The primordial value
> Ω_prim = ±2π/3 and every derived scale are **unchanged**. See the paper's
> closing *Predictions and named deciders* summary. The framework is developed under an explicit
> *epistemic stratification* (Stratum 1 = rigorous; Stratum 2 = theorem with
> distributed/numerical proof; Stratum 3 = structural articulation; Stratum 4–5 =
> heuristic/order-of-magnitude). The code below supports specific claims at the
> stratum stated **in the paper**; please read each result's stratum there before
> citing a number as "derived."

---

## Repository structure

```
pmmd-framework/
├── paper/                     Manuscript (LaTeX source + PDF)
│   ├── PMMD_v7.0.tex
│   └── PMMD_v7.0.pdf
├── src/                       ALL code, grouped by topic
│   ├── flavour/               Koide (HPC + extrapolation), V₁₁₂, generations, PMNS
│   ├── gauge/                 SU(5)/SO(10)/SM identification (X-charge, Berry split, spinor)
│   ├── e8_structure/          E₈ roots, percolation p_c, 3-fold merger, group theory
│   ├── dimension_dBS/         Bombelli–Sorkin dimension d_BS (cut-and-project, growth, BLS)
│   ├── phenomenology/         foam probes, FSS, bond-bias RG, v6.0 geometry checks, hpc/ soliton pipeline
│   ├── dark_sector/           τ-demarcation surface / bubble stats, Ω_DM/Ω_B
│   ├── dynamics/              growth dynamics + induced-gravity (foam rigidity)
│   ├── electromagnetism/      qubit-native photon / EM dynamics
│   └── analysis/              result merging and combined analysis
├── retired/                   superseded scripts (the d_BS-at-p_c partial-order probe) — see retired/README.md
├── scripts/                   distributed-run launchers (bash / PowerShell)
├── data/                      small result/parameter files (JSON, NPZ)
├── figures/                   generated figures (PNG) — see docs/SCRIPTS.md
└── docs/                      protocols, run plans — and SCRIPTS.md (full per-script catalogue)
```

> **Full per-script documentation** — what each script computes, where it hooks into
> the paper, how to run it, and how to read its output — is in
> [`docs/SCRIPTS.md`](docs/SCRIPTS.md). The index below is a short overview.

## Code index

### `src/e8_structure/`
- `e8_group_theory.py` — E₈ / W(E₈) group-theoretic computations (GAP-style checks, V₁₁₂ decomposition support).
- `w_e8_character_analysis.gap` — GAP script (verified against the character-table library): the permutation character R_perm of W(E₈) on the 240 roots = **1 ⊕ 8 ⊕ 35 ⊕ 84 ⊕ V₁₁₂** with ⟨R_perm,R_perm⟩ = 5; χ of V₁₁₂ = [31,4,4,4] on the four order-3 classes; and the 72 roots fixed by the A₂-Coxeter element (= E₆), the remaining 168 in 56 size-3 orbits.
- `option_G_W6_decomposition.g` — GAP script (verified): W(E₆)-module decomposition of V₁₁₂ via `PossibleClassFusions` — the A₂-Coxeter order-3 class (centraliser 311040, size 2240) gives the Koide split **58 ⊕ 27 ⊕ 27̅** (the other three order-3 classes give 40 ⊕ 36 ⊕ 36), with V₁₁₂|_{W(E₆)} dimension multiset {1:4, 6:3, 20:3, 30:1} = 112. Companion to `option_B_V112_koide.py` (flavour/).
- `e8_percolation_2color_v2.py` — two-colour (matter/antimatter) percolation on the E₈ foam graph.
- `e8_percolation_implicit_v42.py` — the current percolation-criticality estimator `p_c ≈ 1/175` (implicit-adjacency Newman–Ziff; cited in the paper). The earlier partial-order versions (v43, v44) were the `d_BS`-at-`p_c` probe and are now in `retired/`.
- `e8_percolation_hpc.py` — high-performance percolation driver (numba JIT + fork-based shared memory; Linux): finite-size-scaling scan of the foam-graph p_c.
- `e8_percolation_winhpc.py` — cross-platform (Windows/Linux/macOS) HPC percolation driver (numba + shared_memory): same FSS scan.
- `e8_percolation_2color_v2.py` — two-colour (matter/antimatter) percolation; `analyze_pc_cycle_sum.py` — the 4-cycle correction to `1/p_c`.
- `e8_3fold_merger_stats_v2.py` — 3-fold cluster-merger event statistics (Σ⁺/Σ⁻ merging).

### `src/dimension_dBS/`
- `compute_dBS.py`, `compute_dBS_full.py` — Myrheim–Meyer / Bombelli–Sorkin combinatorial dimension on causal sets.
- `compute_BLS_rigorous.py` — Bombelli–Lee–Meyer–Sorkin continuum-limit dimension estimator.
- `compute_df_L_scaling.py` — fractal-dimension finite-size scaling in `L`.
- `analyze_d_BS.py` — analysis/aggregation of d_BS runs.
- `cutproject_e8_dBS.py` — ideal E₈ cut-and-project `d_BS` (box-counting `3.85`, longest-chain `3.75–3.85`, FSS `4.06`). The retired `sim_c_cutproject_dBS.py` (the `d_BS`-at-`p_c` probe, `≈3.6`) is now in `retired/`.
- `growth_e8_dBS.py`, `growth_e8_phase_dBS.py` — d_BS on the framework's own **growth-dynamics** ensemble (the Stratum-2 support, converging to ≈ 3.97).

### `src/flavour/` (Koide, V₁₁₂, generations, PMNS)
- `option_B_V112_koide.py` — V₁₁₂ Z₃ decomposition → `Q = 2/3`, angle `θ = 2/9`; `pmmd_koide_hpc.py` + `extrapolate_koide.py` — the continuum Koide value `Q_K → 0.674`; `v112_construction.py`, `pmns_chain_architecture_v1.py`, `cycle_classifier_v1.py`, … (full list in `docs/SCRIPTS.md`).

### `src/gauge/`
- SU(5)/SO(10)/SM identification: `berry_and_su5_split.py`, `su5_xcharge_and_orthberry.py`, `spin10_split_test.py`, `sm_identification.py`.

### `src/phenomenology/`
- `option_D_d4_orbit.py` — D₄ orbit structure of the chain base.
- `sim_bond_bias_RG_v42.py` — bond-bias renormalisation-group run (ε_macro).
- `option_A_FSS_extrapolation.py` — finite-size-scaling extrapolation (e.g. `f_c·p_c`).
- `verify_sic_geometry.py` — **(new in v6.0)** self-contained checks of the primordial C₃/K₄ structure: the canonical value Ω_prim = 2π/3 (the Z₃ phase balance Σ e^{iφ}=0, the per-step phase of a once-wound 3-cycle, the datum the chain propagates and that carries the chirality), and the SIC tetrahedron geometry (pairwise fidelity 1/3, Σnᵢ=0; four faces tiling S² with per-face Berry phase π/2, charge 1/4, total winding Q=1). The geometric reading (π/2) and the canonical value (2π/3) are the two readings of the same C₃ content, related by the k=1 Wess–Zumino normalisation (ratio 4/3). Run it directly.
- `verify_geometry_remarks.py` — **(new in v6.0)** self-contained checks of the v6.0 geometric remarks: E₈→H₄ two-600-cell ratio φ (Moody–Patera), the φ-power scaling of `δ_9`, the charged-lepton triple as a Berry triple, and the H₄-protected Lorentz-suppression estimate. Run it directly.
- `foam_energy_functional.py` — **(new in v6.0)** the foam energy functional (CP¹ sigma model + WZ/Berry + Bloch potential): extracts the BPS lump scale `α_E8⁻¹·m_quantum ≈ M_Pl`, the marginal Higgs (`λ(M_sub)≈0`, consistent with SM near-criticality), and the demonstration that the flavour hierarchy is a distinct sector (Koide/SU(3), not the energy functional).
- `orbit_type_geometry_probe.py` — **(new in v6.0)** honest probe of whether the foam's three icosahedral orbit-types give the generation hierarchy; finds O(1)/φ geometric ratios (not ~100–3000), localising the hierarchy in the flavour/overlap sector rather than the bare geometry.

### `src/dark_sector/`
- `tau_correlated_growth.py` — τ-correlated cluster growth (visible/dark allocation).
- `tau_demarcation_surface_stats.py` — statistics of the τ-demarcation (null) surface.
- `tau_finite_bubble_distribution.py` — finite-bubble distribution of the τ allocation.
- `omega_dm_baryon_patch_merger.py` — Ω_DM/Ω_B ≈ 5.4 as a stochastic merger outcome.

### `src/analysis/`
- `analyze_3fold_merger_combined.py`, `merge_*.py` — aggregation of distributed-run outputs.

### `scripts/`
- `launch_*.sh`, `relaunch_*.sh` — launchers for the distributed Sim B (bridge-claim Poisson-universality) and 3-fold-merger runs.

---

## Running

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Quick self-contained check (a few seconds):
python3 src/phenomenology/verify_geometry_remarks.py
```

Most physics scripts depend only on `numpy`/`scipy`; some plotting uses
`matplotlib` and some graph code uses `networkx`. The large percolation, d_BS,
and Sim B runs are computationally heavy and were executed as distributed jobs;
the `scripts/` launchers and `docs/` protocols document how. The `data/` and
`figures/` folders hold representative small outputs, not the full distributed
result sets.

## Reproducibility notes (honest)

- `verify_geometry_remarks.py` reproduces, in seconds, the φ-geometry,
  Koide, and Lorentz-suppression numbers cited in the v6.0 remarks.
- The percolation `p_c ≈ 1/175`, the growth-ensemble `d_BS ≈ 3.97`, and the
  3-fold-merger statistics require longer runs; finite-size values differ from
  the asymptotic ones by documented amounts (see the paper and `docs/`).
- The originally-envisioned percolation "Sim B / Sim C" probe of `d_BS` (the
  partial-order measurement at the bare critical density `p_c`) is **retired**:
  measured there, the dimension is the fractal *critical-cluster* value `≈3.6`,
  not the macroscopic order. The macroscopic `d_BS = 4` is carried by the
  **mature, over-percolated foam** (ideal cut-and-project `4.06`, growth ensemble
  `3.97`). The retired code is kept in [`retired/`](retired/README.md) for
  transparency; `p_c` itself (`e8_percolation_implicit_v42.py`) is **not** retired.

## Citation

If you use this work, please cite the manuscript (`paper/PMMD_v7.0.pdf`) and,
once published, the corresponding DOI. See `CITATION.cff`.

## License

Code is released under the MIT License (`LICENSE`). The manuscript text and
figures are © Gianluca Genovese.
