# PMMD v5.1 — local-phase patch
**Date**: 2026-05-18
**Trigger**: Conversation audit identified internal inconsistency between `sec:stage-1-chirality` (which explicitly states τ(v) is per-vertex independent) and Proposition `prop:cosmological-phase-selection` in `sec:foam-causal-phase` (which claimed "one phase dominates" globally).

## Logical inconsistency exposed

The framework's matter--dark-matter structural account requires:
- Visible matter (τ = +1) and dark matter (τ = -1) **coexist** in the **same** percolating foam cluster
- Both contribute additively to gravitational mass (Theorem `thm:gravitational-cluster-integrity`)

This is **incompatible** with the original Proposition `prop:cosmological-phase-selection` claim that "one phase dominates by structural majority" within a causally connected region: if dark matter exists in our cluster, then phases τ = ±1 cannot be uniformly aligned.

The i.i.d. random-walk argument in the original proof sketch was also internally inconsistent: i.i.d. variables don't "align by majority" — they fluctuate around their mean, and the mean is not a per-vertex label but a statistical average.

## Changes applied

### Revision 1: Proposition `prop:cosmological-phase-selection` (line 4509-4511)
- **OLD**: "Within a single causally connected region, one phase dominates by structural majority (the macroscopic Lorentzian signature is uniform)."
- **NEW**: Replaced by per-vertex Φ(v) heterogeneous distribution language, with macroscopic Lorentzian signature emerging as a **statistical regularity of the partial order**, recovered from Bombelli–Sorkin combinatorial dimensionality.

### Revision 2: Proof sketch (line 4513-4515)
- **OLD**: "the connectivity choices propagate by majority dynamics (the same i.i.d. random-walk argument as Theorem `thm:spontaneous-Z2-breaking`): the cumulative dominant phase fixes the macroscopic signature"
- **NEW**: Replaced with: per-vertex τ(v) independent classical bit, both signs present; macroscopic signature emerges from partial-order's statistical regularity (central tendency + heterogeneity profile); sub-leading fluctuations preserved as 1/√N_cl scaling from independent cluster-merger contributions.

## Downstream impact

Downstream references to `prop:cosmological-phase-selection`:
- Line 124 (Intro overview): UNCHANGED, abstract reference still valid
- Line 4796 (dark energy decomposition): UNCHANGED, regional-variance prediction preserved
- Line 4823 (Stratum status summary): UNCHANGED, still at Stratum 3
- Line 4730 (`rem:foam-continuum-limit`): UNCHANGED, mathematical setup compatible

The patch is **purely clarifying**: no formal results change, no Stratum classifications shift, no quantitative predictions affected. Downstream references work without modification.

## Recommended next step

- **For v5.1.1 patch release on Zenodo**: a minor revision, no new version number needed if uploaded as "minor correction" to the v5.1 record
- **For v5.2 release** (recommended): bundle this fix with d_BS verification results from Sim B run currently in progress

## Numerical implications

The local-phase view changes nothing about the d_BS = 4 prediction (Bridge Claim 6). The Bombelli–Sorkin combinatorial dimensionality depends on the partial order ⪯_Φ, not on whether per-vertex Φ values are aligned. The Sim B verification path is unchanged.
