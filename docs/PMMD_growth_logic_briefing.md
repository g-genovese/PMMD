# PMMD — Logica fondante della dinamica di crescita del foam

> Documento di briefing da caricare all'inizio di ogni nuova sessione, per non
> perdere la logica corretta tra una chat e l'altra. Versione bozza — da
> correggere dove non corrisponde all'intenzione dell'autore.

## Principio guida (il cuore di tutto)

**La dinamica del sistema tende alla MASSIMA MUTUA DETERMINAZIONE.**

Ogni vertice è determinato dai vicini (le sue coordinate/fasi/informazioni sono
il risultato delle relazioni con i vicini). Il sistema globale evolve verso lo
stato in cui questa mutua determinazione è massima — cioè massima auto-consistenza,
minima frustrazione. È un principio variazionale bootstrap: la struttura si
determina da sé, senza input esterno.

## Unità fondamentale: il triangolo (3 vertici)

- **3 è il minimo per mutua determinazione robusta**: con 2 elementi c'è solo
  accordo/disaccordo; con 3 c'è sempre una maggioranza (2 vs 1), quindi sempre
  una "decisione" stabile. (Analogia: triade di Simmel in sociologia.)
- Ogni vertice di un triangolo è bilanciato dagli altri due. Il criterio NON è
  media aritmetica (darebbe fasi tutte uguali) ma **bilanciamento a somma nulla**:
  Σ e^{iφ} = 0 → le tre fasi a 120° (struttura trifase).
- Le tre fasi a 120° = sistema di radici **A₂** = simmetria **Z₃**. Questo è già
  identificato nel framework (v5.2, prop:Z3-A2-Coxeter-identification) e collega
  alle tre generazioni di fermioni e a SU(3).
- Analogia fisica: corrente alternata trifase. Tre fasi a 120°, configurazione a
  triangolo (delta) senza neutro → sistema auto-contenuto.

## Tempo e chiralità

- **τ (direzione del tempo) NON è un bit fondamentale.** Emerge come **chiralità**
  del triangolo = verso di rotazione delle tre fasi (orario/antiorario), come il
  campo magnetico rotante del motore trifase.
- τ è quindi una proprietà DERIVATA dalla configurazione delle fasi, non scritta
  a priori da nessuna parte.

## Crescita: random + rilassamento (NON deterministica)

1. **I vertici nascono RANDOM**, non in ordine deterministico di "minor costo".
2. Conseguenza: la struttura di partenza è **disordinata** — NON c'è una striscia
   △▽△▽ perfetta subito, NON c'è E8 subito.
3. Dopo ogni nascita, il sistema **si riadatta** (le fasi/posizioni dei vicini si
   propagano e riaggiustano per aumentare la mutua determinazione).
4. **E8 è l'ATTRATTORE** di questo rilassamento — la configurazione 8D di massima
   auto-consistenza — non un input né uno stato raggiunto immediatamente.

## Fasi omogenee, chiralità alternata (risultato verificato)

- Le **fasi** crescono omogenee: restano sui valori Z₃ (0°, 120°, 240°).
- La **chiralità** cresce **alternata**: due triangoli che condividono un bordo
  hanno verso di rotazione OPPOSTO (pattern △▽△▽).
- Meccanismo: un nuovo vertice su un edge forma un triangolo dall'altro lato
  dell'edge rispetto al vicino → geometricamente ribaltato → chiralità opposta.
- Quindi: un cluster nasce GIÀ MISTO in chiralità (non omogeneo), per dinamica
  intrinseca — non solo unendosi ad altri cluster.

## Materia visibile e dark matter

- Materia visibile = sotto-reticolo di una chiralità (es. △).
- Dark matter = sotto-reticolo di chiralità opposta (▽), **intrecciato** col primo
  fin dalla nascita del cluster (non in cluster separati, non mescolato a caso:
  alternato sistematicamente).
- Sotto cut-and-project 8D→4D, i due sotto-reticoli △/▽ danno le due popolazioni
  τ=±1 della Theorem dm-time-direction.
- **Noi siamo la MINORANZA** (~15.6% per massa; Ω_DM/Ω_B ≈ 5.4). La chiralità di
  cui siamo fatti è quella minoritaria nel nostro patch. Per simmetria 𝒞 esiste
  un osservatore coniugato che vede l'opposto (frame-dependent).

## Cluster, interfacce, struttura cosmologica

- I cluster crescono, si uniscono. Possono esistere cluster non connessi (oltre
  il nostro orizzonte).
- Quando cluster a dominanza di chiralità opposta si uniscono → **vertici di
  interfaccia** = τ-demarcation surfaces = **primordial black holes** (esplorazione A).
- Nessuna regione è mai τ-omogenea a nessuna scala: solo **sub-regioni dominate**
  (mai pure) da una chiralità, con minoritari sempre presenti.

## Cosa il sistema NON è (errori da evitare)

- ❌ NON ci sono "bit di configurazione" τ fondamentali (τ = chiralità emergente).
- ❌ NON ci sono cluster τ-omogenei a nessuna scala (solo dominati).
- ❌ NON nasce ordinato/E8 subito (nascita random, E8 è attrattore).
- ❌ NON è media aritmetica delle fasi (è bilanciamento a somma nulla → 120°).
- ❌ La dimensione d_BS NON viene dall'embedding geometrico (è combinatoria,
   emerge dalla connettività della struttura rilassata + cut-and-project).

## Stato della verifica numerica (prototipi)

- ✓ Ordine endogeno/causale-consistente → partial order denso (r ~12× il random).
- ✓ Chiralità → minoranza preservata (pattern △▽ alternato verificato, 50% flip
   tra triangoli adiacenti).
- ✓ d_BS dalla crescita grezza ~7-8 (≈ dimensione substrato 8D); il cut-and-project
   8D→4D è il passo che dovrebbe portarlo a 4 (DA TESTARE).
- ⧖ Prototipo corretto da implementare: nascita random + rilassamento verso massima
   mutua determinazione + cut-and-project, poi misurare d_BS.

## Collegamento al framework pubblicato (v5.2/v6.0)

- Z₃/A₂/Coxeter: prop:Z3-A2-Coxeter-identification (v5.2)
- dark matter come direzione temporale: thm:dm-time-direction (v4.0)
- PBH da τ-demarcation: thm:tau-demarcation-PBH (v5.2)
- causal-consistency, heavy-tail Ω_DM/Ω_B, three-tier emergence: v6.0 (esplorazioni A/B/C)
