# Retired scripts

These scripts implement the **percolation partial-order probe of `d_BS` at the
bare critical density `λ = p_c`** — the originally-planned "Sim B / Sim C" route
to the Bombelli–Sorkin dimension of the foam's causal set.

**Why retired.** The framework's reading of the foam matured (see the paper,
Remark *λ = p_c* and §"The foam at percolation criticality"): the macroscopic
4D Lorentzian causal order is carried by the **mature, over-percolated foam**
(the growth *crosses* `p_c` and densifies beyond it), **not** by the sparse
critical cluster sitting exactly at the bare threshold `p_c`. Measured on that
bare critical cluster, the partial-order dimension comes out at
`d_BS ≈ 3.6` — the fractal *critical-cluster* dimension, the wrong object for the
smooth Lorentzian order.

The `d_BS = 4` result is therefore established by the **mature-foam** measurements,
which live in `src/dimension_dBS/` and are **current**:

| Current probe (kept)            | What it measures                          | Result        | Paper |
|---------------------------------|-------------------------------------------|---------------|-------|
| `cutproject_e8_dBS.py`          | ideal E₈ icosian cut-and-project to 4D    | `d_BS ≈ 4.06` | Remark `dBS-cutproject-v53` |
| `growth_e8_dBS.py`              | framework growth-dynamics ensemble        | `d_BS ≈ 3.97` | Remark `dBS-growth-v53` |
| `growth_e8_phase_dBS.py`        | growth + coherent CP¹ phase tilt          | `d_BS ≈ 3.6–3.9` (coherent), `→4` as α→0 | Remark `coherence-lorentzian-v53` |

The only piece of the old route that **remains a forward target** (not retired)
is the verification of the *full Poisson universality* (interval-cardinality
ratios `N₂/N₃`, `N₃/N₄`) **of the mature foam** — but that is run with the
current `growth_*` machinery, not with the scripts here.

> **Note.** `p_c` itself (the coupling `λ = p_c ≈ 1/175`) is **not** retired: it
> is measured by `src/e8_structure/e8_percolation_implicit_v42.py` (Newman–Ziff,
> cited in the paper) and extrapolated by
> `src/phenomenology/option_A_FSS_extrapolation.py`. Those stay. The earlier
> partial-order percolation versions below were the d_BS-probe machinery and are
> retained here only for transparency / reproducibility of the retired result.

## Files

| File | Was | Superseded by |
|------|-----|---------------|
| `sim_c_cutproject_dBS.py`          | "Sim C" cut-and-project `d_BS` estimator on the bare critical ensemble (`d_BS ≈ 3.6`) | `cutproject_e8_dBS.py` (ideal) + `growth_e8_dBS.py` (mature foam) |
| `e8_percolation_v44_multisnapshot.py` | multi-snapshot partial-order percolation run (generated the Sim C data) | growth-dynamics ensemble |
| `e8_percolation_v43_partial_order.py` | earlier partial-order percolation `d_BS` run | growth-dynamics ensemble |
| `merge_sim_c.py`                   | aggregation + FSS of the Sim C `d_BS` output | `merge_dBS_results.py` (current `growth_*` aggregation) |
| `merge_sim_b_results.py`           | aggregation of the Sim B partial-order output | `merge_dBS_results.py` |

These run, but their output (`d_BS ≈ 3.6` at `p_c`) should **not** be read as the
macroscopic spacetime dimension. Use the `src/dimension_dBS/` scripts for that.
