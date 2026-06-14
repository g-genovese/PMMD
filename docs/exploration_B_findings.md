# Esplorazione (B) — Ω_DM/Ω_B = 5.4 derivazione strutturale

## Sintesi del risultato

Il valore Ω_DM/Ω_B ≈ 5.4 è strutturalmente **derivabile come outcome tipico** (P ~ 1-5%) sotto il quadro corretto di causal-consistency + percolation-criticality, NON ~10σ improbabile come nel quadro v5.2 attuale (i.i.d. random walk implicito).

## Confronto modelli

| Modello | Origine | P(≥5.4) a N=30 | Interpretation |
|---------|---------|----------------|-----------------|
| A (i.i.d. random walk) | v5.2 implicito | 1.5×10⁻⁵ | "Anthropic" / accidental |
| B (Pareto α=1.5) | Heavy-tail generic | 1.3% | Typical |
| C (percolation τ_F=2.31) | Foam-critical | 1.9% | **Strutturalmente derived** |

Il Model C è strutturalmente forzato dal quadro PMMD:
- Foam alla percolazione critica (Stage 9): cluster size distribution n_s ~ s^(-τ_F) automatic
- τ_F ≈ 2.31 per 4D-effective (after cut-and-project from 8D foam)
- Heavy-tailed merger events → 5.4 è outcome plausibile

## Predizioni strutturalmente nuove sotto causal-consistency

### 1. Asimmetria sistematica median(Ω_DM/Ω_B) > 1
Sotto heavy-tail, anche con simmetria 𝒞 esatta, median(max(r, 1/r)) ~ 1.5-2.5
typically. Non è solo "anywhere", è sistematicamente shifted.

### 2. Distribuzione regionale skewed/kurtotic
σ_reg ~ 10-20% (v5.2 attuale) sotto-stima la varianza. Sotto heavy-tail
la varianza è dominata dalle rare deviazioni grandi.
Predizione raffinata: 
  - σ_reg^(0) (Gaussian-component) ~ 5-10%
  - σ_reg^(tail) ~ 20-40% (heavy-tail events)
  - Distribution skewed, kurtosis > 3 (non-Gaussian)

### 3. Universal scaling with cluster size
La distribuzione di Ω_DM/Ω_B nei sotto-cluster cosmologici scala con
una power-law, NON con un Gaussian normalizzato. Predizione testabile:
maggiori SuperGalactic patches hanno tail più pesante.

## Patch v6.0 per §sec:omega-dm-baryon-derivation

### Cosa cambia nel quadro strutturale

**Vecchio (v5.2, implicito Model A)**:
```
N ~ 30 i.i.d. cluster merger events, σ ~ 1/√N → fluttuazione
Gaussian ~ 18%, 5.4 è outcome accidentale (specific draw).
```

**Nuovo (v6.0 candidate, causal-consistency)**:
```
N "decisive cluster merger events" sono FUSIONI DI PATCH CAUSALMENTE
COERENTI, con taglie distribuite secondo percolation-cluster distribution
n_s ~ s^(-τ_F), τ_F ≈ 2.31. Heavy-tailed dynamics. 5.4 è outcome
typical (P ~ 2-5%), non accidentale.
```

### Nuovo theorem candidate per v6.0

```latex
\begin{theorem}[Heavy-tailed Ω_DM/Ω_B distribution from causal-consistency]
Under the causal-consistency dynamics of foam crystallisation (Definition
def:foam-causal-phase), the cluster-merger events that determine Ω_DM/Ω_B
are not i.i.d. but follow the percolation-critical patch size distribution
n_s ∼ s^{−τ_F}, with τ_F = 1 + d_eff/d_f where d_eff is the projected 
spatial dimension and d_f the foam fractal dimension. The resulting 
distribution of Ω_DM/Ω_B in the ensemble of causally disconnected patches
is heavy-tailed with:
  - Mean ⟨Ω_DM/Ω_B⟩ = 1 (substrate 𝒞-symmetry)
  - Median(max(r, 1/r)) > 1 (heavy-tail bias)
  - Skewness > 0 (positive bias toward extreme deviations)
  - Tail probability P(r ≥ 5.4) ≈ 2-5% (typical, not anthropic)
\end{theorem}
```

### Implicazioni per le predizioni testabili

1. **Survey LSS** (Euclid, Roman, LSST): cercare correlazioni dell'Ω_DM/Ω_B
   regionale con la distribuzione di SMBH e PBH. Heavy-tail predicts:
   - Pochi outlier patches con Ω_DM/Ω_B molto alto
   - Correlazione con densità di sub-cluster mergers attivi
   - Asimmetria positiva sistematica

2. **CMB lensing**: power spectrum della densità DM dovrebbe mostrare 
   non-Gaussianities sub-leading rispetto al baseline ΛCDM.

3. **PBH IMBH density**: sotto heavy-tail, la densità di PBH IMBH
   correla con il valore locale di Ω_DM/Ω_B (entrambi sono outcomes
   degli stessi cluster-merger events).

## Connessione con (A) e con interface-Λ

L'esplorazione (A) ha mostrato che τ-correlation deve essere causal-mediated.
L'esplorazione (B) mostra che le fusioni heavy-tail dominate spiegano 5.4.

Entrambi convergono in un unico quadro: **il foam cristallizza tramite 
fusioni di patch causalmente coerenti, con taglie distribuite secondo
percolation criticality. Questi cluster mergers determinano:**

1. **τ-demarcation surfaces** alla scala del patch (PBH IMBH spectrum, A)
2. **Ω_DM/Ω_B** come heavy-tail outcome (B)
3. **Possibly Λ_observed** se interface contribuisce additivamente al vacuum
   energy (Λ = 3/2 · Λ_V112+58, 3.4% match — TBD)

Tutti e tre i risultati sono strutturalmente conseguenze dello stesso 
meccanismo: foam-formation tramite cluster mergers in regime critical-percolation.

## Action items per v6.0 (NOT NOW)

1. Patch al §sec:omega-dm-baryon-derivation con il theorem causal-consistency
2. Articolazione esplicita della patch-size distribution
3. Nuova prediction σ_reg sub-leading + tail-leading
4. Cross-correlation con PBH IMBH density observable
5. Integrazione con (A) e possibilmente con interface-Λ hypothesis


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
