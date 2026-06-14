# PMMD — script catalogue

Every script, what it computes, **where it hooks into the paper**, how to run it,
and how to read its output. Strata are as stated **in the paper** — check a result's
stratum there before quoting a number as "derived".

Layout (after the v6.0 cleanup): **`paper/` holds only the manuscript**; **all code is
under `src/`**, grouped by topic; **`scripts/`** holds the bash/PowerShell launchers;
retired code is in [`retired/`](../retired/README.md).

Conventions: **Run** lines assume the repo root + `pip install -r requirements.txt`
(`numpy`/`scipy`; some use `matplotlib`/`networkx`). **[self-contained]** = seconds, prints
its result; **[heavy]** = long / distributed (see `scripts/` and `docs/` protocols).

---

## `src/flavour/` — Koide, V₁₁₂, generations, PMNS
- **`pmmd_koide_hpc.py`** `[heavy]` — matrix-free overlap-Dirac Koide `Q_K` on a charge-`Q=3` baby-skyrmion (Chebyshev/Zolotarev sign + low-mode deflation), per `L`. *Paper:* Remark `gauge-overlap-generations-v53`, flavour-soliton programme (continuum `L=64–320`). *Run:* `python3 src/flavour/pmmd_koide_hpc.py --L 96 --Q 3`. *Read:* JSONL `{L,Q,nzero,chir,koide{higgs_n3,conn_a2,flux}}`; valid only if `nzero==|Q|`; the winding operator is `conn_a2`.
- **`extrapolate_koide.py`** `[self-contained]` — `1/L` continuum extrapolation of `Q_K`. *Paper:* same remark (`Q_∞≈0.674`, `+1.1%`). *Run:* `python3 src/flavour/extrapolate_koide.py pmmd_res_L*.jsonl`. *Read:* `conn_a2→0.674`, `higgs_n3→0.356`, `flux→0.40`.
- **`koide_mass_operator.py`** `[self-contained]` — dense `L=20` operator ranking (`Q_K≈0.51` best). *Paper:* same remark.
- **`koide_equipartition.py`** `[self-contained]` — equal-`Z₃`-power (coherence) reading `Q=2/3`. *Paper:* Remark `koide-coherence-conjecture-v53`.
- **`koide_rg_running_probe.py`**, **`koide_su5_clebsch.py`** `[self-contained]` — quark-`Q` one-loop RGE toward `2/3`; the `SU(5)` Clebsch origin of the down-sector deviation. *Paper:* §quark sector.
- **`option_B_V112_koide.py`** `[self-contained]` — `V₁₁₂` `Z₃` decomposition → `Q=2/3`, `θ=2/9`. *Paper:* Theorem `koide-identity`.
- **`v112_construction.py`** `[self-contained]` — explicit degree-3 harmonic basis on the 240 roots; `A₂`-Coxeter → `58+27+27` (immature sample → `40+36+36`). *Paper:* Prop `Z3-A2-Coxeter-identification`, §cartan-A₂-unified.
- **`v112_bundle_holonomy_v1.py`**, **`pentagonal_unit_v1.py`**, **`icosian_shadows_v1.py`**, **`foam_path_holonomy_v1.py`** `[self-contained]` — pentagonal transport unit and foam-path holonomies building the **PMNS architecture v3** (no continuous phases). *Paper:* §PMNS (v3).
- **`pmns_chain_architecture_v1.py`**, **`caseG_condensate_phases_v1.py`** `[self-contained]` — the closed PMNS sector from the chain / case (G).
- **`cell_koide_probe.py`** `[self-contained]` — whether the bare cell/orbit geometry gives the generation hierarchy (it gives `O(1)/φ`, not `~100–3000` → hierarchy in the overlap sector).
- **`analyze_121_pure58.py`**, **`cycle_classifier_v1.py`** `[self-contained]` — the `121=120+1` `A₂`-orbit count and the cycle-level per-generation SM content. *Paper:* Remark `cycle-generation-identification-v53`.

## `src/gauge/` — SU(5)/SO(10)/SM identification
- **`berry_and_su5_split.py`**, **`su5_xcharge_and_orthberry.py`** `[self-contained]` — `SU(5)` X-charge and the orthogonal-`A₂`-plane / Berry split of the quark sector. *Paper:* §quark sector.
- **`spin10_split_test.py`** `[self-contained]` — the `Spin(10)` spinor (`16`) split. *Paper:* §"16 fermions from foam".
- **`sm_identification.py`** `[self-contained]` — the `E₈⊃…⊃SM` breaking-chain identification. *Paper:* §SM gauge group.

## `src/e8_structure/` — E₈ roots, percolation `p_c`, 3-fold merger
- **`e8_group_theory.py`** `[self-contained]` — `W(E₈)` permutation-rep decomposition `1⊕8⊕35⊕112⊕84`, marks. *Paper:* Theorem `Rperm-decomposition`.
- **`e8_percolation_implicit_v42.py`** `[heavy]` — **current** `p_c≈1/175` (implicit-adjacency Newman–Ziff; cited). *Paper:* §percolation `λ=p_c`, Prop `fc-FSS-discrimination`. *Read:* per-trial `1/p_c` (`172.55` at `L=12`) → feed `option_A_FSS_extrapolation.py`.
- **`e8_percolation_2color_v2.py`** `[heavy]` — two-colour (matter/antimatter) percolation. *Paper:* §`Z/2` / 3-fold merger.
- **`e8_3fold_merger_stats_v2.py`** `[heavy]` — 3-fold merger statistics (`1/4` unanimous, `3/4` split). *Paper:* §"fractal 3-fold numerical".
- **`analyze_pc_cycle_sum.py`** `[self-contained]` — the 4-cycle correction to `1/p_c` beyond the triangle term. *Paper:* §`v6-beta`.
- *(retired: the partial-order percolation versions v43/v44 → `retired/`.)*

## `src/dimension_dBS/` — Bombelli–Sorkin dimension `d_BS`
- **`cutproject_e8_dBS.py`** `[heavy]` — ideal `E₈` icosian cut-and-project (box `3.85`, longest-chain `3.75–3.85`, FSS `4.06`). *Paper:* Remark `dBS-cutproject-v53`.
- **`growth_e8_dBS.py`** `[heavy]` — `d_BS` on the framework's growth (mature-foam) ensemble (`≈3.97`). *Paper:* Remark `dBS-growth-v53`. *Read:* aggregate with `src/analysis/merge_dBS_results.py`.
- **`growth_e8_phase_dBS.py`** `[heavy]` — growth + CP¹ phase (coherent `3.6–3.9`→4; incoherent `1.9`). *Paper:* Remark `coherence-lorentzian-v53`.
- **`compute_dBS.py`**, **`compute_dBS_full.py`**, **`compute_BLS_rigorous.py`**, **`compute_df_L_scaling.py`** `[self-contained/heavy]` — Myrheim–Meyer / BLMS `d_BS` and the critical-cluster mass-dimension `d_f` (note: `d_f` = critical-cluster mass dim; `d_BS` = mature-foam causal dim; they concur at 4). *Paper:* Remark `foam-continuum-limit`, Corollary `4D-emergence`.
- **`analyze_d_BS.py`** `[self-contained]` — Myrheim–Meyer estimator + Poisson interval-cardinality checks (the remaining mature-foam target). *Paper:* §bridge-claims status.
- *(retired: `sim_c_cutproject_dBS.py` → `retired/`.)*

## `src/phenomenology/` — foam probes, FSS, geometry checks
- **`option_A_FSS_extrapolation.py`** `[self-contained]` — FSS `1/p_c→≈175`; the `f_c·p_c≈1` discrimination (rejects `183` at `>8σ`, IR-effective `176.25` at `≤2.2σ`). *Paper:* Prop `fc-FSS-discrimination`.
- **`option_D_d4_orbit.py`** `[self-contained]` — `D₄` orbit structure. *Paper:* §`D₄⊂E₈` triality.
- **`sim_bond_bias_RG_v42.py`** `[heavy]` — `ε_macro` bond-bias (`0.1622`; RG-running rejected `5.3σ`). *Paper:* Remark `two-thirds-mf-derivation`.
- **`verify_sic_geometry.py`** `[self-contained]` — **(v6.0)** `C₃/K₄` SIC + `Ω_prim=2π/3` (the SIC = a single qubit's `d²=4` state frame, four Bloch vectors on one sphere — **not four qubits**; see `figures/fig_sic_tetrahedron.png`). *Paper:* §"SIC structure of `K₄`", Prop `v6-C3-phase`.
- **`verify_geometry_remarks.py`** `[self-contained]` — **(v6.0)** φ-geometry, Koide, Lorentz-suppression numbers. *Paper:* the v6.0 geometric remarks.
- **`foam_energy_functional.py`** `[self-contained]` — foam energy functional (CP¹ σ + WZ + potential): BPS lump scale ≈ `M_Pl`, marginal Higgs. *Paper:* §Higgs-foam-mode, Remark `energy-functional-v53`.
- **`foam_overlap_probe.py`**, **`foam_fermionic_probe.py`**, **`foam_quantisation_probe.py`**, **`foam_v112_z3_probe.py`** `[self-contained]` — Yukawa overlap `⟨H|ψₖ⟩`, 3-generation loop states, harmonic quantisation, whether the 112-mode eigenspace carries `Z₃`.
- **`orbit_type_geometry_probe.py`** `[self-contained]` — whether icosahedral orbit-types give the hierarchy (honest negative).
- **`stage3_babyskyrmion.py`** `[self-contained]` / **`stage3_grid_scaffold.py`** `[heavy]` — baby-skyrmion profile (Derrick `E4=E0`) and the 2D-lattice kernel (the soliton background for the Koide overlap).
- **`spectral_triple_axioms.py`** `[self-contained]` — non-commutative-geometry spectral-triple axioms. *Paper:* §connections.
- **`src/phenomenology/hpc/`** `[heavy]` — the soliton→Koide pipeline: `run_pilot.py` (per-machine driver), `stage_A_soliton.py` (CP¹+Skyrme background), `stage_B_gauge_overlap.py` (zero modes via the emergent CP¹ gauge field; cited), `stage_BC_overlap.py` (`|c|`), `stage_D_continuum.py` (continuum `|c|`), `cont_koide.py` (continuum `Q_K`).

## `src/electromagnetism/` — qubit-native photon / EM
- **`analyze_em_interaction.py`**, **`analyze_loop_wave_emission.py`**, **`analyze_photon_framing_helicity.py`**, **`analyze_photon_transverse_dynamics.py`**, **`analyze_tangential_emission.py`**, **`analyze_z3_triangle_photon.py`**, **`analyze_equatorial_propagation.py`**, **`analyze_foam_collective_em.py`**, **`analyze_foam_collective_em_tetrahedron.py`**, **`analyze_loop_higgs_photon_hierarchy.py`**, **`verify_photon_polarisation_closure.py`**, **`cubic_rate_probe.py`** `[self-contained]` — the EM interaction, loop-wave emission, the photon's two transverse modes / framing-helicity, transverse oscillation, tangential-emission geometry, and the `Z₃`-triangle photon hypothesis. *Paper:* Part VI (photons, wave–particle duality, mode conversion).

## `src/dark_sector/` — τ-demarcation and Ω_DM/Ω_B
- **`tau_correlated_growth.py`** `[heavy]` — τ-correlated growth (visible/dark). *Paper:* Theorem `dm-time-direction`, Remark `tau-correlation-three-regimes`.
- **`tau_demarcation_surface_stats.py`** `[heavy]` — τ-demarcation (null) surface = primordial BHs. *Paper:* Theorem `tau-demarcation-PBH`.
- **`tau_finite_bubble_distribution.py`** `[heavy]` — finite-bubble τ distribution. *Paper:* Corollary `PBH-mass-spectrum`.
- **`omega_dm_baryon_patch_merger.py`** `[heavy]` — `Ω_DM/Ω_B≈5.4` stochastic outcome. *Paper:* Theorem `omega-DM-heavy-tailed`.

## `src/dynamics/` — growth dynamics and induced gravity
- **`e8_foam_growth.py`** `[heavy]` — the vertex-by-vertex tetrahedral-balance growth. *Paper:* §`vertex-addition-dynamics`.
- **`foam_rigidity.py`** `[heavy]` — induced-gravity worker (foam curvature-rigidity). *Paper:* Theorem `einstein-hilbert-effective`.
- **`bd_action.py`** `[heavy]` — Benincasa–Dowker causal-set route to `S_EH`. *Paper:* §causal-set action convergence.
- **`farm.py`** `[heavy]` — orchestrator for the foam-rigidity farm.

## `src/analysis/` — aggregation
- **`analyze_3fold_merger_combined.py`** — combines distributed 3-fold-merger runs.
- **`merge_dBS_results.py`** — aggregates `growth_e8_dBS.py` per-trial `d_BS` (FSS over trials).
- **`merge_BLS_results.py`** — aggregates `compute_BLS_rigorous.py` outputs.
- **`verify_projection_geometry.py`** `[self-contained]` — exact `E₈→4D` projection geometry. *Paper:* Remark `foam-continuum-limit`.
- *(retired: `merge_sim_b_results.py`, `merge_sim_c.py` → `retired/`.)*

## `scripts/` and `docs/`
- **`scripts/launch_*.sh`, `relaunch_*.sh`, `run_pmmd.sh`, `run_pmmd_win.ps1`** — launchers for the distributed heavy runs and the per-machine pilots.
- **`docs/`** — simulation protocols, run plans, and **this catalogue**.

## `figures/` — generated figures
- **`fig_chain.png`** — the 9-stage chain `φ₀→…→foam` (single dynamics, growth crosses `p_c` → mature foam → 4D).
- **`fig_sic_tetrahedron.png`** — the qubit SIC: four Bloch-vector **states** of **one** qubit (`C₃` = 3 states, `K₄` adds the 4th, `Σn=0`).
- **`fig_dBS_methods.png`** — `d_BS` by method: mature foam → 4 (ideal `4.06`, growth `3.97`); bare critical cluster at `p_c` → fractal `3.6` (retired).
- **`fig_koide_continuum.png`** — continuum `Q_K`: only the winding-sensitive connection operator → `2/3` (`0.674`).
- **`fig_pc_fss.png`** — `1/p_c` FSS → `≈175`; the triangle-corrected branching value `183` vs the Bethe `239`.
- *(plus the original run diagnostics: `option_A_FSS_extrapolation.png`, `e8_3fold_L6_synthesis.png`, `2color_L10_mode1_synthesis.png`, `sim_b_L12_diagnostic.png`.)*

---

## Quick start
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 src/phenomenology/verify_sic_geometry.py        # C₃/K₄ SIC + Ω_prim
python3 src/phenomenology/verify_geometry_remarks.py    # φ-geometry, Koide, Lorentz suppression
python3 src/flavour/extrapolate_koide.py pmmd_res_L*.jsonl   # Koide Q_∞ ≈ 0.674
```
The percolation `p_c`, the `d_BS` ensembles, and the dark-sector / 3-fold-merger statistics are heavy distributed jobs; finite-size values differ from the asymptotic ones by amounts documented in the paper and `docs/`.
