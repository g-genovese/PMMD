# PMMD v4.2 Simulation Protocol — Server Execution Guide

**Versione**: 1.0 (16 maggio 2026)
**Riferimento**: Remark `rem:epsilon-bps-rigorous` e Remark `rem:foam-continuum-limit` del PMMD v4.1 (Sections foam-causal-phase).

Questo documento specifica le due simulazioni numeriche aperte per il programma v4.2. Sono progettate per essere eseguibili su un server con ~64 GB RAM e 8-16 core.

---

## Sommario delle simulazioni

| Sim | Scopo | Codice | Runtime stimato | Priorità |
|-----|-------|--------|------------------|----------|
| **A** | Verificare empiricamente l'esponente RG di bond-bias `|λ_bb| ≈ 0.045` | `sim_bond_bias_RG_v42.py` (nuovo) | ~6h single-core; ~1h con 8 core | Alta |
| **B** | Risolvere la discrepanza estimator a L=12 (1/p_c χ=172.4 vs dS=169.5) | Codice esistente 8D E_8 dell'utente | ~40h con 128 trials | Media |

Entrambe sono indipendenti; possono girare in parallelo se vuoi.

---

## Simulazione A — Bond-bias RG measurement (Task 2 verifica)

### Obiettivo

L'analisi del paper (Remark `rem:epsilon-bps-rigorous`) deriva:
- ε_struct (substrate scale, BPS-rigoroso) = 0.25
- ε_macro (foam scale, empirico da L=30) ≈ 0.20
- Renormalisation R = 0.80 → |λ_bb|/ν_4d ≈ 0.066 → |λ_bb| ≈ 0.045

Questa simulazione misura **indipendentemente** lo scaling di Δ(ε, L) = 1/3 − ⟨π_v⟩ per estrarre |λ_bb| direttamente. Se il fit dà |λ_bb| ≈ 0.045, l'analisi RG del paper è confermata. Se diverge significativamente, l'argomento RG va rivisto.

### Setup

- Reticolo: 4D simple cubic, periodic BCs
- p = p_c^(4d) = 0.16013 (Lorenz-Ziff 1998)
- Etichette phase {v, s, c} assegnate uniformemente per sito
- Bond bias: bond_prob(label_u, label_v) = p_c · (1 + δ(label_u, label_v)/2)
  - Realisation A: δ_v=−ε, δ_s=+ε, δ_c=−ε
  - Realisation B (C_CP-image): δ_v=−ε, δ_s=−ε, δ_c=+ε
  - Si fanno entrambe e si media (forza C_CP-symmetric)

### Comando di base (production)

```bash
# Scaricare il file da Anthropic
# Posizione: /mnt/user-data/outputs/sim_bond_bias_RG_v42.py

# Test rapido per validare l'environment (~30s)
python3 sim_bond_bias_RG_v42.py --L_values 8 --epsilon_values 0.20 --n_trials 2

# Production run (ε scan a 3 L values, 32 trials → ~6h single core)
python3 sim_bond_bias_RG_v42.py \
    --L_values 16 24 32 \
    --epsilon_values 0.10 0.15 0.20 0.25 0.30 \
    --n_trials 32 \
    --output_dir ./results_v42_simA \
    --run_id simA_$(date +%Y%m%d)

# Production extended (4 L values, 64 trials → ~30h single core, ~4h con 8 core)
python3 sim_bond_bias_RG_v42.py \
    --L_values 16 24 32 48 \
    --epsilon_values 0.10 0.15 0.20 0.25 0.30 \
    --n_trials 64 \
    --output_dir ./results_v42_simA_extended \
    --run_id simA_ext_$(date +%Y%m%d)
```

### Parallelizzazione

Lo script è single-thread. Per parallelizzare, suddividi i valori di (L, ε) su istanze separate:

```bash
# Esempio: 4 worker, ciascuno con un sottoinsieme
for L in 16 24 32 48; do
    python3 sim_bond_bias_RG_v42.py \
        --L_values $L \
        --epsilon_values 0.10 0.15 0.20 0.25 0.30 \
        --n_trials 64 \
        --output_dir ./results_simA_L${L} \
        --run_id simA_L${L} &
done
wait
```

I file JSON output possono essere uniti facilmente in post-processing.

### Output e interpretazione

Lo script stampa direttamente:

```
=== RG Analysis ===
Delta(epsilon, L) = 1/3 - <pi_v>:
 epsilon |  L= 16 |  L= 24 |  L= 32
  0.100  | +0.0XXX | +0.0XXX | +0.0XXX
  0.150  | +0.0XXX | +0.0XXX | +0.0XXX
  ...

Log-log fits: Delta(L) ~ L^(-|lambda_bb|/nu_4d)
  epsilon=0.100: slope=-0.0XX, |lambda_bb|/nu_4d=0.0XX, |lambda_bb|=0.0XX
  ...

Average |lambda_bb| across epsilon values: 0.0XX +- 0.0XX
Analytical estimate from paper: 0.045
Verification: CONSISTENT / DISCREPANT
```

### Risultati attesi

- **Δ(ε, L) deve essere positivo e crescente in ε**: il bond bias riduce sistematicamente π_v dal valore di equipartizione 1/3 (questo controlla la correttezza del setup)
- **Δ(ε, L) decresce lentamente con L**: lo scaling `Δ ~ L^(-|λ_bb|/ν_4d)` con |λ_bb|/ν_4d ≈ 0.066 implica che passando da L=16 a L=32 la riduzione è di un fattore `(32/16)^(-0.066) ≈ 0.955`, cioè ~5% di decrescita per raddoppio di L
- **|λ_bb| inferito**: dovrebbe essere ≈ 0.045 ± 0.020 se l'argomento RG del paper è corretto

### Possibili esiti

1. `|λ_bb|_misurato ≈ 0.045 ± 0.020` → **Conferma l'analisi RG**. Il gap ε_struct/ε_macro è autoconsistentemente spiegato.
2. `|λ_bb|_misurato ≈ 0.10–0.20` → **Bond-bias è più rilevante del previsto**. Il framing RG va sostituito con un meccanismo di tipo "bulk operator marginally relevant" — riapertura della questione.
3. `|λ_bb|_misurato ≤ 0.01 (zero entro errori)` → **Il gap non viene da RG**, ma da un meccanismo statico (e.g. lattice artifact, non-universal correction). L'argomento del paper va rivisto.
4. `Δ(ε, L) non scala lineare in ε` → **Necessita un fit non-lineare**; il regime small-ε potrebbe non applicarsi.

### Cosa rispedire

Il file `bondbias_rg_run<run_id>.json` e l'output del log. Posso riusare quei dati per:
- Aggiornare la stima di |λ_bb| nel paper con error bar
- Spostare lo statement RG da Stratum 2-3 a Stratum 2 nel paper
- Iterare se l'esito è (2) o (3) sopra

---

## Simulazione B — 8D E_8 L=12 estimator refinement (Task 4)

### Obiettivo

La discrepanza tra estimatori a L=12 (32 trials):
- 1/p_c (chi peak) = 172.4
- 1/p_c (dS/dp peak) = 169.5
- Discrepanza: 1.7%

è inconsistente con sub-percent FSS expectations a L=12. Le tre possibilità sono:
- (i) Statistica insufficiente (32 trials)
- (ii) Effetti finite-size sistematici diversi per i due estimatori
- (iii) Bug in una implementazione

Questa simulazione USA IL TUO CODICE 8D E_8 ESISTENTE (`e8_percolation_L8.json`, `e8_percolation_results_L10.json` etc) e aggiunge:
- Più trials a L=12 (target 128 o 256)
- Cross-validation a L=10 (dove i dati attuali sono solidi)
- Computazione di entropy varianti multiple (S1, S2, S3) per identificare se la discrepanza è specifica di una definizione

### Specifiche

Sul tuo codice 8D E_8 esistente, modifica/aggiungi:

1. **L=12 esteso**: 128 trials (4× il dato attuale), stesso range di p
2. **L=10 verifica**: 128 trials (più del dato originale) per cross-check
3. **Entropie multiple per ogni p, per ogni trial**:
   - S1 = −Σ_clusters (s/N) log(s/N) [Shannon by site]
   - S2 = −Σ_clusters (1/n_clusters) log(1/n_clusters) [Shannon by cluster count]
   - S3 = −log(Σ_clusters (s/N)²) [Rényi-2 by site]
4. **Estimator computation per trial e per curva media**: `1/p_c` da peak di χ(p), da peak di |dS_k/dp| per k=1,2,3

### Runtime stimato

- L=12, 128 trials, 21 p values: ~40h single core (4× il run originale a 32 trials)
- L=10, 128 trials: ~10h
- Parallelizzabile linearmente

### Output atteso

Per ogni L (10, 12):
```
Estimator results:
  1/p_c (chi peak)              = AAA.AAA ± a.aaa
  1/p_c (dS1/dp, Shannon-site)  = BBB.BBB ± b.bbb
  1/p_c (dS2/dp, Shannon-count) = CCC.CCC ± c.ccc
  1/p_c (dS3/dp, Renyi-2)       = DDD.DDD ± d.ddd
```

### Interpretazione

- Se `1/p_c (dS1)` ≈ 169.5 ma `1/p_c (dS2 o dS3)` ≈ 172.4: la discrepanza è specifica della scelta di entropia. **Identifichi quale Shannon variant l'analisi originale usava.** Le 3 varianti sono ben definite e ciascuna dà un estimator universalmente valido per p_c — convergenti nel limite L → ∞ ma differenti a L finito.
- Se TUTTE le entropie convergono a ~169.5 mentre χ-peak è ~172.4: la discrepanza è genuinely tra observable diversi (χ vs S). Possibile correzione finite-size diversa per due osservabili scalanti diversamente.
- Se 1/p_c (chi peak) a 128 trials è significativamente diverso dal valore a 32 trials: era statistica.

### Output per il paper

L'estremale finite-size scaling per 1/p_c(∞):
```
1/p_c(∞) = 1/p_c(L) + a·L^(-1/ν)
```
con ν ≈ 0.689 per 4D percolation (analogo per 8D E_8). Con L=8, 10, 12 (e 14 se disponibile), il fit dà 1/p_c(∞) con error bar che entrerà nella Table di p_c estimates del paper.

---

## Cosa metto nel paper dopo i risultati

A seconda dei numeri:

### Se Sim A conferma `|λ_bb| ≈ 0.045 ± 0.020`

Aggiorno Remark `rem:epsilon-bps-rigorous` con:
- "Empirically verified at $|\lambda_{\mathrm{bb}}| = 0.0X \pm 0.0X$ from 4D bond-bias simulation (n_trials × n_L = ...)"
- Upgrade dello statement RG da "Stratum 2-3" a "Stratum 2" pieno
- Eventualmente lo elevo a Theorem se la verifica è abbastanza pulita

### Se Sim B chiarisce la discrepanza estimator

Aggiorno la Section foam-causal-phase numerical evidence con:
- Table dei 1/p_c estimators per L=8, 10, 12 con i 4 valori (χ, S1, S2, S3) ciascuno
- Discussione del FSS extrapolation
- Quale estimator è preferibile e perché (di solito χ-peak è meno rumoroso ma più dipendente da L)
- 1/p_c(∞) finale con error bar

### Se uno dei due dà esito inatteso

Aggiungo una Remark dedicata che discute il risultato negativo e ripropone il meccanismo. Risultati negativi sono parte legittima del programma v4.2.

---

## Note operative

- Tutti gli script salvano JSON + NPZ + log
- Il file JSON ha tutta l'informazione necessaria; l'NPZ è per FSS analysis su dati raw
- I file di log includono trial-by-trial timing per stimare il tempo rimanente
- Posso aggregare dati di run multipli in post-processing (non serve un unico run massivo)

## Cosa aspettarmi dai risultati

Stima dei tempi totali per il programma v4.2 completo:
- Sim A (production): ~30 ore su single core con 64 trials, o ~5h con 8 core paralleli
- Sim B (production): ~40 ore con 128 trials a L=12, parallelizzabile
- Totale: 2-3 settimane di calcolo elapsed con setup parallelo, oppure ~10 giorni con 16 core dedicati

Suggerisco di lanciare Sim A per prima (più piccolo, e Task 2 è il fronte più importante per la coerenza interna del paper).
