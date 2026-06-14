# Esplorazione (A) — Risultati per v6.0

## Conclusione strutturale chiave

La predizione di PBH IMBH macroscopici (10²-10⁵ M_⊙) del framework v5.2 richiede **specifica condizione strutturale** non chiaramente articolata in v5.2:

**Il foam causal phase Φ deve generare τ-correlations near the Ising critical point** (β ≈ β_c), non strettamente i.i.d. Bernoulli per-vertex.

## Evidenza numerica (3D cubic, validation framework-indipendente)

### Statistical bubble distribution
| Regime | Largest finite bubble | Frazione di N | PBH IMBH? |
|---|---|---|---|
| Bernoulli i.i.d. (β=0) | ~30 vertici | 0.005% | NO |
| Sub-critical (β=0.1-0.18) | ~50 vertici | 0.02% | NO |
| **Ising-critical (β_c)** | **180 vertici** | **0.08%** | **YES** |
| Ferromagnetic (β>β_c) | ~15000 vertici | 7% | YES (ma asimmetrico) |

### Costanti universali (per qualsiasi β)
- 2 sub-regioni τ-dominated frammiste, ciascuno ~50% del volume del cluster percolante
- Interfaccia globale ~z/2 · (1-1/2) edges/vertex (60 in 8D foam con z=240)
- Frazione singletons τ-opposti: ~1.7%

## Conseguenze per il theorem v5.2

Il `thm:tau-demarcation-PBH` v5.2 va raffinato in v6.0:

### Parte (a) — Strutturalmente forzata (qualsiasi β)
Co-presenza di sub-regioni dominate da τ=+1 e da τ=-1 nei cluster percolanti, con interfaccia distribuita. **Questo è il meccanismo dark matter / visible matter di `thm:dm-time-direction`.** NON è event horizon.

### Parte (b) — Strutturalmente forzata (qualsiasi β)
Esistenza di τ-bubbles finite con compact closed boundaries. **Quelle finite SONO event horizons (PBH).**

### Parte (c) — Condizionata su Φ-dynamics → Ising-critical
Macroscopic IMBH spectrum 10²-10⁵ M_⊙. Richiede τ-correlation near β_c. **Stratum 2-3** invece di puro 2.

## Plausibilità strutturale del Ising-critical tuning

1. **Φ(v) è "stochastic outcome of local connectivity choices"** → correlazione corta range naturale
2. **Percolation criticality at foam edge** → coupling con Φ dynamics
3. **Marginal coupling** strutturale (né indipendenza pura né forte coupling)
4. Storicamente analogo: percolation + Ising entrambe critiche → criticality coupling in mean-field domain (d > d_c)

## Impatto su exploration (B) — Ω_DM/Ω_B = 5.4

Nuovo quadro:
- I "decisive cluster mergers" del random walk Ω_DM/Ω_B sono **fusioni di sub-cluster τ-omogenei** durante la formazione del foam
- Il numero N_eff = O(10-100) (Equation eq:N-prediction) corrisponde al numero di sub-cluster macroscopici τ-omogenei nel cluster cosmologico
- A β=β_c questo è derivabile dal Ising-critical scaling: N_eff ~ ξ_Ising^d / V_cosmic

## Impatto su exploration (C) — evaporazione PBH ↔ CMB

I foam-scale PBH (Bernoulli regime) evaporano in 10^(-37) s. Energy released:
- m_* ~ 10² M_P per PBH foam-scale
- ~0.02 N PBH per vertex inizialmente
- N_cosmo ~ 10^180 vertici
- Total energy release: 2 × 10^178 M_P ~ 10^178 g · c²

Questa energy distribuzione presumibilmente contribuisce al substrate radiation background (Section CMB-density) **but at the foam scale, redshifted by ~ 10^60 expansion factor**, riducendosi a energy density << ρ_observed. Solo i Sopravvissuti IMBH contribuiscono Hawking radiation detectable.

## Patch da fare in v6.0 (NON ORA — solo nota)

```latex
\begin{theorem}[τ-demarcation surfaces: refined formulation (v6.0)]
[...]
Provided that the foam causal phase Φ generates per-vertex τ assignments
with local correlations characterised by a coupling near the Ising critical
point β ≈ β_c^{Ising}, the distribution of finite τ-bubble sizes follows a
power-law scaling sufficient to produce macroscopic intermediate-mass black
holes in the range 10²-10⁵ M_⊙ as survivors of Hawking evaporation. Under
the alternative Bernoulli-i.i.d. assignment (β=0), only foam-scale finite
bubbles exist, all evaporating within ~10^(-37) s.
\end{theorem}
```

Con corollary aggiornato:
- Distinzione tra "infinite-extent interface between sub-regioni dominantly opposite-τ" (non event horizon, distribuzione decoherence) e "compact closed boundary of finite τ-bubble" (genuine event horizon)
- Mass spectrum derivation dalla power-law scaling: log-log slope ≈ -τ_Ising^{3D} ≈ -2.18 (giusto sopra d_c)
- Probabilistic argument per Φ-dynamics → β ≈ β_c (Stratum 2-3)

## Action items

1. **Tenere documento questo** per v6.0 release content
2. **Procedere a (B)** con il quadro corretto (random walk = sub-cluster fusion events)
3. **Procedere a (C)** con la classificazione delle 2 popolazioni PBH (foam-scale evanescenti + cosmological survivors)
4. **Eventualmente**: simulazione 8D del foam con Ising-critical τ allocation (computational target avanzato)


---

## CORREZIONE TERMINOLOGICA (importante per v6.0)

Si è chiarito a livello concettuale: **non esistono cluster τ-omogenei a nessuna scala
del foam**, né microscopica né macroscopica. La struttura corretta:

- **Esiste un UNICO cluster percolante** (a Stage 9 criticality)
- Questo unico cluster contiene **entrambe le direzioni τ frammiste** a tutte le scale
- A scale intermedie/cosmologiche emergono **sub-regioni statisticamente dominate**
  da una direzione (mai omogenee, sempre con minoritari dell'altra)
- Le τ-demarcation surfaces (PBH) sono boundary tra sub-regioni di dominanza opposta,
  NON tra "cluster omogenei separati"

## Frame symmetry observation (NEW per v6.0)

Sotto Ω_DM/Ω_B ≈ 5.4, nel nostro patch:
- **Materia visibile (noi)**: 15.6% della massa totale (minoranza)
- **Dark matter (τ opposto)**: 84.4% della massa totale (maggioranza)

Per la simmetria 𝒞-substrato, esiste un **osservatore τ-coniugato** che:
- Si identifica come τ=-1 (la nostra "dark matter")
- Vede 84.4% di "sua materia visibile"
- Ci percepisce come "dark matter di lui" (15.6%)

Entrambi i frame sono **strutturalmente equivalenti** sotto la simmetria 𝒞. La nostra
identificazione di "noi visibili / loro scuri" è frame-dependent.

**Predizione candidate per v6.0 (Remark)**:

```latex
\begin{remark}[Frame symmetry under 𝒞: minority observers]
Under the substrate 𝒞-symmetry (k → -k, φ → \bar{φ}), the framework predicts 
the existence of a τ-conjugate observer for whom our "dark matter" appears as
visible matter and vice versa. The observed ratio Ω_DM/Ω_B = 5.4 in our frame
corresponds to Ω_B/Ω_DM = 5.4 in the conjugate frame. Both readings are 
structurally indistinguishable; the choice of "visible" vs "dark" is 
frame-dependent. The framework therefore predicts that we are minority 
observers (~15.6% by mass) within a unique percolating cluster dominated 
by the opposite τ allocation.
\end{remark}
```
