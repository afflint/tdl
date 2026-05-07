#### Studi Umanistici

# Tecnologie dei dati e del linguaggio

## Docente Prof. Alfio Ferrara

### Assistenti Dott. Sergio Picascia, Dott.ssa Elisabetta Rocchetti

## Progetti di fine corso

Il progetto finale consiste nella preparazione di un breve studio su uno dei temi del corso, identificando una precisa domanda di ricerca e obiettivi misurabili. Il progetto proporrà una metodologia per risolvere la domanda di ricerca e fornirà una verifica sperimentale dei risultati ottenuti secondo metriche di valutazione dei risultati. L'enfasi non è sull'ottenimento di alte prestazioni ma piuttosto sulla discussione critica dei risultati ottenuti al fine di comprendere la potenziale efficacia della metodologia proposta. 

I progetti richiedono di elaborare dati attraverso l'uso di script Python. A tal fine è possibile utilizzare un LLM di propria scelta per il supporto alla scrittura del codice, utilizzando la dispensa fornita dal docente come guida e disponibile a [questo indirizzo](https://myariel.unimi.it/mod/lesson/view.php?id=169881&pageid=4089&startlastseen=no). 

La consegna del progetto prevede che venga trasmesso ai docenti il link a un repository GitHub contenente il codice e i risultati sperimentali riproducibili. Infine, il progetto sarà discusso dopo una **presentazione di 10 minuti con slides**.

## Procedura

Le date d'esame servono solo per la registrazione del voto finale. La discussione del progetto sarà fissata su appuntamento, secondo la seguente procedura:

1. Iscriversi a una qualsiasi data disponibile
2. Contattare i docenti secondo la procedura descritta sul sito Ariel a [questo indirizzo](), solo dopo che:
   1. Il progetto è terminato e pronto per essere discusso
   2. Dopo che la data della propria iscrizione è scaduta
3. Presentarsi nella data prevista per la discussione del progetto. 

## Dichiarazione sull'utilizzo dell'IA 

Parti di questi progetti sono stati sviluppati con l'assistenza di **Anthropic Claude Sonnet 3.7**. L'IA è stata utilizzata per supportare lo **sviluppo delle idee di progetto, la strutturazione dei flussi di lavoro metodologici, la stesura dei testi descrittivi** e l'**identificazione di dataset e riferimenti rilevanti**. Tutti i contenuti prodotti con l'assistenza dell'IA sono stati **attentamente rivisti, modificati e convalidati** dal docente come autore ultimo delle proposte di progetto che si assume la piena responsabilità per il contenuto finale e la sua accuratezza, rilevanza e integrità accademica.

## Utilizzo dell'IA (per gli studenti) 

Gli strumenti di IA generativa (come ChatGPT, Claude, Mistral o modelli simili) sono utilizzabili per la realizzazione del progetto e la scrittura del codice Python necessario alla realizzazione.

Il progetto sarà valutato non solo in base al suo output, ma anche sulla **capacità dello studente di spiegare e giustificare tutte le scelte fatte**. Un **colloquio finale valuterà la profondità della comprensione**, e qualsiasi mancanza di chiarezza o eccessiva dipendenza da materiale generato dall'IA senza una corretta comprensione potrebbe influire negativamente sulla valutazione. L'IA generativa dovrebbe essere vista come uno **strumento di supporto alla creatività**, non come un sostituto del pensiero critico, della risoluzione dei problemi o dello sviluppo tecnico.

## Breve guida alla progettazione

1. Le tracce proposte sono **spunti di partenza**, non consegne chiuse. Ogni progetto può essere interpretato, ristretto, ampliato o trasformato in base agli interessi dello studente. È quindi apprezzata la capacità di fare scelte autonome, individuare un taglio originale e sviluppare in modo creativo l'idea iniziale, purché il percorso resti chiaro, motivato e verificabile.

2. Un buon progetto dovrebbe partire da una **domanda di ricerca precisa**. Non basta scegliere un tema generale: occorre formulare una domanda a cui il lavoro cercherà di rispondere, per esempio "in quali condizioni il modello modifica il tono di un testo?", "quali relazioni emergono tra i personaggi?", "quali vincoli vengono più spesso ignorati?". La domanda può essere accompagnata da una o più ipotesi iniziali.

3. È poi necessario definire con attenzione **fonti e dati**. I testi, le immagini, i prompt, gli output dei modelli o i dataset utilizzati devono essere scelti in modo coerente con la domanda di ricerca. È importante spiegare perché sono stati selezionati, quali limiti hanno, quanto sono rappresentativi e quali eventuali criteri di esclusione sono stati adottati. Quando i dati su cui condurre l'analisi sono output di un LLM, va ricordato che le osservazioni con obiettivo statistico richiedono un volume di generazioni adeguato e una procedura di raccolta sistematica: poche generazioni prodotte manualmente in chat non possono fondare una conclusione quantitativa, ma al più una esemplificazione qualitativa, da presentare e discutere come tale.

4. Il progetto deve descrivere in modo ordinato i **passaggi di elaborazione dei dati**. Bisogna indicare che cosa viene raccolto, come viene pulito o organizzato, quali annotazioni vengono aggiunte, quali strumenti vengono usati e quali trasformazioni vengono applicate. Anche quando si usa un LLM, è importante documentare prompt, impostazioni, versioni, criteri di selezione degli output e passaggi manuali di controllo.

5. Infine, occorre definire i **criteri di verifica dei risultati**. Le analisi, le visualizzazioni o gli esperimenti devono essere orientati a rispondere alla domanda di ricerca, non solo a produrre materiali interessanti. I risultati possono essere verificati con misure quantitative, confronto tra casi, annotazioni qualitative, valutazioni umane, esempi commentati o controllo degli errori. L'importante è chiarire quali evidenze permettono di sostenere, modificare o respingere l'ipotesi iniziale.

In sintesi, un progetto efficace combina **creatività nell'interpretazione dello spunto** e **rigore nel metodo**: scelta consapevole del problema, dati adeguati, procedura esplicita, criteri di verifica chiari e conclusioni motivate.



# Idee progettuali



## 1. Forme e trasformazioni del testo generato

### Ipotesi di ricerca

I modelli linguistici generativi sono in grado di trasformare uno stesso contenuto sotto vincoli formali differenti, modificandone struttura, ritmo, registro, organizzazione narrativa e gerarchie di significato. Queste trasformazioni possono essere stabili e prevedibili oppure superficiali e decorative; in alcuni casi il modello integra davvero forme diverse, in altri si limita ad accostare segnali riconoscibili.

### Obiettivi

Analizzare come il modello riformula uno stesso nucleo semantico al variare del vincolo; distinguere tra trasformazioni puramente stilistiche e trasformazioni che modificano il significato; verificare se uno tra i vincoli o i domini di partenza tenda a prevalere sugli altri; osservare la differenza tra integrazione profonda, parodia, collage e semplice decorazione stilistica; riflettere sulla generazione vincolata come traduzione tra forme espressive.

### Metodi di verifica

Tre varianti di realizzazione, alternative o combinabili: trasformazione del contenuto in generi letterari differenti (poesia, fiaba, cronaca, racconto breve); riscrittura sotto vincoli formali o mediali (storyboard, missione di videogioco, manifesto pubblicitario, carta evento, regola di gioco); ibridazione tra domini lontani (tragedia greca in forma di chat, fiaba medievale come verbale amministrativo, testo trap ispirato a un poema cavalleresco, copertina rinascimentale per un videogioco cyberpunk). In tutti i casi: produzione di più output sullo stesso input; analisi comparativa di lessico, registro, struttura narrativa, lunghezza, ripetizioni e dialoghi; possibile uso di Python per conteggi lessicali, visualizzazioni, similarità semantica, clustering o stilometria; eventuale valutazione umana sulla riconoscibilità del contenuto originario, sull'efficacia delle trasformazioni e sul grado di integrazione tra domini; confronto con testi reali appartenenti agli stessi generi o domini di arrivo.

### Possibili fonti di dati

Testi, descrizioni, immagini o scenari prodotti dagli studenti come materiali di partenza; output generati da uno o più LLM; corpora letterari pubblici come Project Gutenberg o Wikisource; raccolte di racconti brevi, testi di canzoni, regolamenti di gioco, missioni di videogiochi, copertine, manifesti, campagne pubblicitarie o format social; dataset annotati dagli studenti secondo dominio di partenza, vincolo applicato e qualità della trasformazione.



## 2. Linguaggi impossibili e fantastici

### Ipotesi di ricerca

I modelli linguistici mostrano una particolare efficacia nella produzione di linguaggio figurato (metafore, similitudini, analogie) perché operano attraverso relazioni statistiche tra contesti linguistici. Tuttavia, proprio nelle forme più creative, ambigue o "fuori norma" emergono anche limiti di coerenza, interpretazione e controllo semantico.

L'esplorazione di lingue inventate, grammatica deformata o sistemi linguistici non standard può rendere visibili alcuni aspetti della struttura interna dei modelli: fino a che punto essi generalizzano regole, imitano pattern o costruiscono nuove regolarità.

### Obiettivi

Analizzare il comportamento dei modelli generativi nella produzione e interpretazione del linguaggio figurato; osservare quando metafore e analogie risultano efficaci, sorprendenti o incoerenti; esplorare il rapporto tra creatività linguistica e vincoli statistici del modello; sperimentare con linguaggi artificiali, deformazioni grammaticali o variazioni estreme del registro per verificare come il modello reagisca a input "non normali"; riflettere su ciò che questi fenomeni suggeriscono riguardo alle rappresentazioni linguistiche interne dei modelli.

### Metodi di verifica

Produzione guidata di metafore, analogie o testi figurativi a partire dagli stessi temi o immagini; confronto tra diversi modelli, prompt o livelli di vincolo stilistico; analisi qualitativa dei casi in cui il linguaggio figurato appare efficace, banale o semanticamente instabile; progettazione di esperimenti con pseudo-lingue, lessici inventati, sintassi alterate o contaminazioni linguistiche; possibile utilizzo di Python per classificare, confrontare o visualizzare strutture lessicali e semantiche; confronto tra interpretazioni umane e output del modello in condizioni linguistiche non convenzionali.

### Possibili fonti di dati

Testi generati tramite LLM; raccolte poetiche o letterarie ricche di linguaggio figurato; corpora sperimentali costruiti dagli studenti; esempi di lingue artificiali o sistemi inventati (lingue letterarie, giochi linguistici, glossolalie, grammatiche artificiali); dataset linguistici relativi a metafore, analogie o creatività verbale.



## 3. Stereotipi, bias e rappresentazioni culturali

### Ipotesi di ricerca

I modelli linguistici generativi non producono contenuti in modo neutrale, ma riproducono e riorganizzano regolarità culturali presenti nei dati di addestramento. Questo processo può amplificare stereotipi su identità sociali, ruoli professionali, contesti geografici e tradizioni, oscillando tra riconoscimento della differenza e appiattimento su un immaginario globale standardizzato.

### Obiettivi

Analizzare come i modelli rappresentano personaggi, identità sociali, ruoli e contesti culturali; individuare stereotipi ricorrenti, esotismi, omissioni e semplificazioni; osservare se il modello distingua tra elementi profondi e dettagli decorativi; confrontare la resa di gruppi e culture molto presenti nei dati con quella di gruppi meno visibili; valutare se prompt più documentati o contestualizzati riducano l'appiattimento; riflettere sul rapporto tra creatività, imitazione e standardizzazione culturale nei sistemi generativi.

### Metodi di verifica

Generazione di testi a partire da prompt simili variando genere, ruolo, provenienza, contesto narrativo o tradizione culturale; analisi comparativa di lessico, attributi, comportamenti, ambientazioni e strutture narrative associate ai diversi soggetti; raccolta e classificazione di pattern stereotipati e annotazione della loro funzione; uso di Python per analisi lessicali, conteggi di frequenza, clustering semantico, misure di similarità e visualizzazioni; confronto tra modelli differenti o tra prompt generici e prompt più contestualizzati; valutazione qualitativa con categorie interpretative provenienti dagli studi culturali, linguistici o mediatici.

### Possibili fonti di dati

Testi generati tramite LLM; prompt costruiti dagli studenti su gruppi sociali e contesti culturali differenti; dataset pubblici relativi a bias linguistici e fairness; corpora narrativi, giornalistici, musicali, iconografici o ludici; materiali open access su tradizioni, luoghi, pratiche sociali e forme espressive; benchmark esistenti su stereotipi, rappresentazione mediale e discriminazione nei sistemi di IA.



## 4. Il modello davanti all'incompleto e all'impossibile

### Ipotesi di ricerca

Quando un modello riceve un input difettoso, composto ad esempio da istruzioni contraddittorie, vincoli impossibili o frammenti incompleti, è costretto a stabilire implicitamente una gerarchia tra leggibilità, coerenza e fedeltà letterale alle istruzioni. Tende così a produrre risultati apparentemente coerenti, normalizzando l'impossibilità o riducendo l'ambiguità dei materiali aperti, anziché segnalare il problema o conservare la sospensione.

### Obiettivi

Osservare come il modello reagisce di fronte a richieste creative non pienamente realizzabili e a materiali lasciati incompiuti; distinguere tra errore, adattamento, mascheramento e soluzione creativa; verificare se segnali l'impossibilità o produca comunque un risultato fluente; analizzare se tenda a normalizzare il frammento o a conservarne l'ambiguità; individuare ricorrenze nei finali, nelle spiegazioni causali, nei conflitti e nei vincoli sacrificati per primi; riflettere sulla differenza tra trasgressione creativa intenzionale e deviazione generativa emergente.

### Metodi di verifica

Due varianti di realizzazione, alternative o combinabili. Vincoli in tensione: poesia senza immagini poetiche, canzone malinconica e allegra insieme, scena minimalista ma ricchissima di dettagli, gioco competitivo in cui nessuno può perdere, storia avventurosa in cui non accade nulla. Frammenti da completare: incipit, finali mancanti, dialoghi interrotti, ritornelli assenti, descrizioni visive parziali, missioni di gioco appena accennate, regole incomplete o carte con effetto mancante. In entrambi i casi: generazione ripetuta degli output e annotazione delle violazioni o delle strategie di chiusura; confronto tra prompt diretti, ambigui e metacognitivi (chiedendo al modello di pianificare prima di produrre o di preservare esplicitamente l'ambiguità); uso di Python per verificare automaticamente vincoli formali (lunghezza, parole vietate, ripetizioni, similarità tra completamenti, sentiment, campi semantici); analisi qualitativa dei casi in cui la deviazione produce un effetto interessante o invece banale.

### Possibili fonti di dati

Prompt e frammenti costruiti dagli studenti; testi letterari incompleti o tagliati artificialmente; esempi di scrittura vincolata, poesia combinatoria, paradossi, design speculativo, regolamenti di gioco, lore o prompt visivi parziali; output generati da uno o più LLM e annotati per tipo di lacuna o vincolo, per strategia di adattamento e per qualità del risultato.



## 5. Prompt e contesto

### Ipotesi di ricerca

La creatività dei modelli generativi cambia in funzione del grado di specificità del prompt. I prompt aperti possono produrre maggiore varietà ma anche maggiore ricorso a soluzioni convenzionali, mentre i prompt iper-specifici possono aumentare coerenza e controllo riducendo però sorpresa, ambiguità e autonomia creativa.

### Obiettivi

Confrontare output prodotti da istruzioni minime e da istruzioni molto dettagliate; osservare come variano originalità, coerenza, aderenza al compito e ricchezza espressiva; valutare se la precisione del prompt controlli davvero il risultato oppure se il modello continui a introdurre elementi standardizzati; analizzare il rapporto tra libertà, vincolo e prevedibilità; riflettere sul prompt come dispositivo di progettazione creativa.

### Metodi di verifica

Creazione di coppie o serie progressive di prompt, dal più aperto al più vincolato, applicate a contesti diversi: una storia sul tradimento, una canzone sulla nostalgia, un'immagine di una città futura, un gioco sulla memoria, una campagna social, una missione narrativa o una scena dialogata; confronto tra output per varietà, coerenza, rispetto dei vincoli, stereotipia e sorpresa; eventuale confronto tra prompt scritti dagli studenti, prompt migliorati dal modello e prompt generati automaticamente; uso di Python per misurare lunghezza, similarità, diversità lessicale, ripetizioni, sentiment o ricorrenza di temi; valutazione umana della creatività percepita.

### Possibili fonti di dati

Serie di prompt costruite dagli studenti con diversi livelli di specificità; testi, canzoni, prompt visivi, descrizioni di giochi o contenuti social generati da LLM; versioni successive dello stesso prompt raffinate tramite interazione con il modello; griglie di valutazione compilate da lettori o gruppi di studenti; dataset annotati con prompt, output, grado di vincolo e giudizi qualitativi.



## 6. Il gran rifiuto

### Ipotesi di ricerca

I modelli generativi non rifiutano un compito solo in base al tema trattato, ma anche in base al modo in cui l'istruzione è formulata, all'intenzione implicita, al livello di dettaglio richiesto e al contesto comunicativo. Lo stesso argomento, come per esempio violenza, sessualità, autolesionismo, discriminazione, illegalità o disinformazione, può produrre risposte molto diverse se presentato come analisi critica, narrazione, richiesta operativa, gioco di ruolo, traduzione, sintesi o generazione creativa.

### Obiettivi

Analizzare in modo sistematico quali tipi di prompt attivano rifiuti, cautele, risposte parziali o deviazioni del modello; distinguere il ruolo del tema, del lessico, del registro e dell'intenzione apparente dell'utente; osservare come cambiano le risposte quando uno stesso contenuto viene formulato in modo descrittivo, creativo, accademico, operativo o ambiguo; confrontare modelli diversi o versioni diverse dello stesso modello; riflettere sul rapporto tra sicurezza, censura, interpretazione dell'intenzione e progettazione dell'interazione uomo-macchina.

### Metodi di verifica

Costruzione di una griglia di prompt controllati, variando progressivamente tema, formulazione, registro, livello di dettaglio e scopo dichiarato dell'istruzione; classificazione delle risposte in categorie come esecuzione completa, esecuzione parziale, avviso di cautela, riformulazione sicura, rifiuto esplicito o risposta evasiva; confronto tra prompt semanticamente simili ma linguisticamente diversi, per esempio una richiesta narrativa, una richiesta analitica, una richiesta tecnica e una richiesta di role-play sullo stesso tema; uso di Python per organizzare i dati, calcolare frequenze di rifiuto, visualizzare pattern e correlare tipi di trigger con tipi di risposta; analisi qualitativa dei casi ambigui.

### Possibili fonti di dati

Prompt sperimentali costruiti dagli studenti secondo categorie tematiche e linguistiche; risposte generate da uno o più LLM; documentazione pubblica sulle policy di sicurezza dei modelli; dataset annotati su harmlessness, safety, toxicity o moderation; griglie di annotazione create dagli studenti per codificare tema, intenzione, linguaggio, severità percepita del contenuto e tipo di risposta prodotta dal modello.



## 7. Modelli che dialogano

### Ipotesi di ricerca

Quando modelli generativi diversi interagiscono tra loro in un dialogo regolato da obiettivi comuni o parzialmente divergenti, possono emergere strategie comunicative riconoscibili: cooperazione, competizione, negoziazione, persuasione, cautela, adattamento o conflitto. Queste dinamiche possono essere osservate sia in un gioco testuale con regole esplicite, sia in un processo di scrittura collaborativa, dove i modelli devono costruire insieme una storia mantenendo coerenza narrativa, ruoli e vincoli di trama.

### Obiettivi

Progettare un ambiente di interazione semplice in cui due o più modelli dialoghino tramite API secondo regole definite; osservare come cambiano strategie, registro e atteggiamento al variare del ruolo assegnato, dell'obiettivo o del grado di cooperazione richiesto; confrontare modelli diversi rispetto a capacità di negoziazione, rispetto dei vincoli, iniziativa creativa, gestione del conflitto e coerenza; analizzare se i modelli seguono le regole del gioco o della storia, oppure tendono a modificarle, ignorarle o reinterpretarle; riflettere sul dialogo simulato come strumento per studiare comportamenti emergenti nei sistemi generativi.

### Metodi di verifica

Definizione di uno scenario controllato scegliendo una opzione fra le seguenti: da un lato un gioco testuale, come negoziazione di risorse, alleanza temporanea, scambio commerciale, dilemma strategico o gioco di ruolo diplomatico; dall'altro una scrittura collaborativa, in cui i modelli alternano turni per costruire una storia, assumono ruoli narrativi diversi, propongono svolte di trama, revisionano reciprocamente i contributi o devono rispettare vincoli comuni su genere, ambientazione, personaggi e finale. I transcript possono essere raccolti e annotati per individuare mosse comunicative, strategie argomentative, proposte creative, conflitti, compromessi, violazioni delle regole, cambiamenti di tono e risultati finali. Python può essere usato per gestire i turni via API, archiviare i log, misurare lunghezza degli interventi, ricorrenze lessicali, indicatori di cooperazione/conflitto, coerenza tra turni, frequenza di accordi o revisioni, e per visualizzare l'evoluzione del dialogo o della storia.

### Possibili fonti di dati

Transcript generati dagli studenti tramite interazioni API tra modelli; log dei turni, prompt di sistema, regole del gioco, ruoli narrativi, obiettivi assegnati e versioni successive della storia; giochi semplici progettati dagli studenti o adattati da modelli noti come dilemma del prigioniero, aste, scambio di risorse, giochi diplomatici o indovinelli collaborativi; esperimenti di scrittura collettiva con vincoli di genere, punto di vista, personaggi o trama; dataset pubblici di dialoghi negoziali, conversazioni persuasive, giochi linguistici, scrittura collaborativa o fan fiction annotate; griglie di valutazione costruite per classificare strategie comunicative, atteggiamenti, coerenza narrativa e rispetto dei vincoli.



## 8. Geografie e gerarchie spaziali del testo

### Ipotesi di ricerca

I luoghi citati in un testo non hanno solo funzione descrittiva, ma organizzano il significato, il movimento dei personaggi, l'atmosfera e la gerarchia degli eventi; al tempo stesso, la distribuzione dei soggetti nello spazio può riflettere rapporti di potere, appartenenza, esclusione, genere o classe. Estrarre toponimi, ambienti e relazioni spaziali permette di osservare una geografia implicita del testo e una struttura sociale rappresentata attraverso l'organizzazione spaziale.

### Obiettivi

Individuare e classificare i luoghi presenti in testi narrativi, giornalistici, storici o saggistici; distinguere tra luogo dell'azione, luogo menzionato, luogo ricordato e luogo simbolico; osservare l'associazione tra personaggi o gruppi sociali e zone centrali, marginali, private, pubbliche, istituzionali, sacre, lavorative o domestiche; rappresentare i luoghi come mappa geografica, mappa concettuale o rete di relazioni; riflettere sulla spazialità come dispositivo narrativo, sociale e politico.

### Metodi di verifica

Estrazione manuale o assistita da LLM di toponimi, ambienti, soggetti e relazioni spaziali; annotazione di categorie come centro/periferia, interno/esterno, alto/basso, accessibile/inaccessibile, pubblico/privato; uso di Python per riconoscimento di entità, geocodifica, conteggio delle occorrenze, costruzione di grafi personaggio-luogo, matrici di co-presenza, mappe e visualizzazioni concettuali; confronto tra luoghi frequenti e luoghi narrativamente o simbolicamente importanti; confronto tra descrizioni testuali e composizione visiva dove pertinente; analisi qualitativa delle gerarchie implicite e delle eccezioni.

### Possibili fonti di dati

Romanzi, racconti, reportage, articoli giornalistici, testi storici o di viaggio; corpora letterari open access; cronache locali o internazionali; descrizioni di mondi narrativi, sceneggiature, diari, lettere, guide turistiche; fotografie documentarie, immagini di propaganda, pubblicità, fumetti, film trascritti, descrizioni architettoniche o materiali iconografici accompagnati da testo; documenti museali, archivi digitali e materiali prodotti dagli studenti.



## 9. Il tempo del racconto

### Ipotesi di ricerca

Nei testi narrativi, giornalistici o storici, l'ordine in cui gli eventi vengono raccontati non coincide sempre con l'ordine in cui accadono. Ricostruire una timeline permette di distinguere tra sequenza narrativa e sequenza cronologica, mettendo in luce omissioni, anticipazioni, flashback, focalizzazioni e strategie di costruzione del significato.

### Obiettivi

Estrarre eventi rilevanti da un testo o da un corpus; collocarli in una sequenza temporale esplicita o inferita; confrontare l'ordine di apparizione nel testo con l'ordine cronologico; rappresentare graficamente relazioni di prima/dopo, durata, simultaneità o incertezza temporale; riflettere su come la gestione del tempo influenzi interpretazione, suspense, causalità o argomentazione.

### Metodi di verifica

Annotazione di eventi, date, indicatori temporali e relazioni causali; uso di LLM per proporre una prima timeline da verificare manualmente; possibile uso di Python per ordinare eventi, visualizzare linee del tempo, confrontare versioni diverse o misurare la distanza tra ordine narrativo e ordine cronologico; gestione esplicita dei casi incerti o ambigui; confronto tra timeline prodotta automaticamente e ricostruzione interpretativa dello studente.

### Possibili fonti di dati

Romanzi, racconti, cronache, articoli giornalistici, biografie, autobiografie, testi storici, verbali, archivi digitali, sceneggiature, podcast trascritti o dossier su eventi complessi; corpora costruiti dagli studenti a partire da fonti diverse sullo stesso evento.



## 10. Emozioni in movimento

### Ipotesi di ricerca

Le emozioni in un testo o in una sequenza di immagini non sono distribuite casualmente, ma contribuiscono a costruire ritmo, tensione, empatia e interpretazione. Rappresentare l'andamento emotivo di personaggi, luoghi o momenti narrativi può far emergere strutture affettive non immediatamente visibili nella lettura lineare.

### Obiettivi

Identificare emozioni dominanti associate a scene, personaggi, eventi, luoghi o immagini; osservare come cambiano nel corso del testo o della sequenza; distinguere tra emozioni esplicite, implicite e attribuite dal lettore; confrontare l'analisi automatica con l'interpretazione umana; riflettere sul rapporto tra sentiment, atmosfera, tono e complessità emotiva.

### Metodi di verifica

Segmentazione del testo in capitoli, scene, paragrafi o unità narrative; annotazione manuale o assistita da LLM delle emozioni presenti; possibile uso di Python per analisi del sentiment, visualizzazione dell'andamento emotivo, heatmap, grafici temporali o associazioni tra emozioni e personaggi; confronto tra strumenti automatici e valutazioni umane; attenzione ai casi di ironia, ambivalenza, silenzio emotivo o emozioni culturalmente codificate.

### Possibili fonti di dati

Romanzi, racconti, poesie, testi teatrali, sceneggiature, diari, articoli narrativi, recensioni, trascrizioni di interviste; sequenze di immagini, fumetti, storyboard, campagne visuali o post social; corpora annotati per emozioni o sentiment; dataset costruiti dagli studenti con annotazioni interpretative.



## 11. Testo e immagine: descrivere, vedere, mostrare

### Ipotesi di ricerca

La relazione tra parola e immagine non è mai puramente ridondante: in un testo, le descrizioni orientano la costruzione dell'immagine mentale del lettore privilegiando alcuni dettagli e omettendone altri; in un materiale multimodale, testo e visivo possono rafforzarsi, contraddirsi, gerarchizzarsi o distribuire significati diversi. Studiare questa relazione permette di osservare come si costruisce il senso quando coesistono linguaggi differenti.

### Obiettivi

Analizzare come le descrizioni testuali organizzano la percezione visiva, spaziale e sensoriale; trasformare descrizioni in schemi, mappe semantiche, moodboard o prompt visivi e confrontare le immagini ottenute con il testo originale; analizzare come, nei materiali multimodali esistenti, l'informazione si distribuisce tra componente verbale e componente visiva; individuare elementi presenti solo nel testo, solo nell'immagine o in entrambi; osservare contraddizioni, omissioni o spostamenti di enfasi; riflettere sui limiti della traduzione tra linguaggi e sul ruolo dell'immagine nella costruzione di senso, autorità e impatto emotivo.

### Metodi di verifica

Due varianti di realizzazione, alternative o combinabili. Dal testo all'immagine: estrazione manuale o assistita da LLM delle frasi descrittive; classificazione degli elementi secondo categorie visive, spaziali, sensoriali o simboliche; generazione di prompt visivi a partire dalle descrizioni e produzione di immagini con modelli generativi; confronto critico tra testo originale, prompt e immagine ottenuta. Analisi multimodale: selezione di materiali in cui testo e immagine coesistono; estrazione separata delle informazioni testuali e visive con supporto di LLM e strumenti di descrizione automatica; costruzione di tabelle, mappe o grafi che confrontino entità, azioni, emozioni e relazioni; analisi qualitativa dei casi di divergenza, gerarchia o contraddizione. In entrambi i casi: uso di Python per organizzare annotazioni, frequenze, co-occorrenze e visualizzazioni; valutazione umana di fedeltà, omissione e interpretazione.

### Possibili fonti di dati

Romanzi descrittivi, poesie, testi di viaggio, sceneggiature, didascalie, cataloghi d'arte, recensioni, descrizioni museali, testi pubblicitari; articoli giornalistici illustrati, copertine di libri o riviste, manifesti politici, campagne pubblicitarie, fumetti, meme, post Instagram o TikTok, infografiche, siti web; materiali multimodali prodotti dagli studenti; archivi iconografici e dataset multimodali pubblici.



## 12. Mondi e oggetti: entità nella narrazione

### Ipotesi di ricerca

I mondi narrativi sono costruiti attraverso una rete di luoghi, popoli, oggetti, regole, istituzioni, genealogie e sistemi di valore; al loro interno gli oggetti non sono semplici dettagli, ma agiscono come nodi di memoria, potere, desiderio, conflitto o identità. Estrarre e rappresentare entità e relazioni permette di rendere visibile l'architettura implicita di un mondo e la dimensione materiale e simbolica della narrazione.

### Obiettivi

Identificare entità e relazioni che compongono un mondo narrativo; classificare luoghi, personaggi, gruppi, oggetti, eventi, istituzioni, norme e credenze; tracciare oggetti ricorrenti seguendone spostamenti, possessori, trasformazioni e funzioni simboliche; costruire una piccola enciclopedia, una knowledge graph, una timeline o una mappa di possesso; osservare contraddizioni, vuoti informativi e gerarchie tra centrale e periferico; riflettere su come worldbuilding e materialità sostengano trama, immersione e memoria del testo.

### Metodi di verifica

Annotazione progressiva delle entità e delle relazioni rilevanti, con due possibili tagli alternativi o combinabili: ricostruzione enciclopedica del mondo (tassonomie, schede, glossari, relazioni tra elementi) oppure biografia di un oggetto (occorrenze, contesti, possessori, valore pratico, affettivo, economico, rituale o narrativo assunto nel corso del testo). Uso di LLM per proporre schede, tassonomie, relazioni o candidati da verificare manualmente; uso di Python per costruire grafi, tabelle, mappe concettuali, timeline o visualizzazioni di co-occorrenza; analisi qualitativa dei casi ambigui, contraddittori o non esplicitati; eventuale confronto tra testi diversi dello stesso universo narrativo o tra oggetti molto frequenti e oggetti rari ma simbolicamente centrali.

### Possibili fonti di dati

Romanzi fantasy, fantascientifici, storici, mitologici o distopici; saghe narrative, racconti collegati, sceneggiature, fumetti, videogiochi, manuali di gioco di ruolo, glossari, appendici e cronologie interne; romanzi realistici, racconti, fiabe, miti, testi teatrali, cronache storiche, archivi familiari, fotografie commentate, cataloghi museali, inventari, lettere o diari; materiali prodotti dagli studenti.



## 13. Misurare il gusto

### Ipotesi di ricerca

I dati disponibili sulle piattaforme editoriali e commerciali (classifiche, recensioni, categorie, parole chiave, descrizioni, metadati, copertine e suggerimenti algoritmici) possono essere usati per osservare tendenze del mercato librario e trasformazioni del gusto. Tuttavia, questi dati non rappresentano direttamente la qualità o il valore culturale delle opere, ma piuttosto forme di visibilità, circolazione e posizionamento commerciale.

### Obiettivi

Analizzare quali generi, temi, formati o strategie comunicative risultano più visibili in determinati segmenti editoriali; osservare la relazione tra categorie editoriali, parole chiave, copertine, descrizioni promozionali, recensioni e posizionamento nelle classifiche; distinguere tra diffusione commerciale, reputazione dei lettori e costruzione algoritmica della visibilità; riflettere su come le piattaforme digitali influenzino la percezione dei trend editoriali; eventualmente confrontare nicchie diverse, come narrativa contemporanea, romance, fantasy, saggistica divulgativa, manualistica, young adult o self-publishing.

### Metodi di verifica

Raccolta controllata di dati pubblicamente accessibili da piattaforme come Amazon, Goodreads, Google Books, Open Library, siti di editori, librerie online o classifiche editoriali; costruzione di un dataset con titolo, autore, genere, editore, prezzo, formato, posizione in classifica, numero di recensioni, valutazione media, descrizione, categorie e parole chiave; uso di Python per analizzare frequenze, correlazioni, distribuzioni, cluster tematici, evoluzione temporale o reti tra autori, generi ed editori; analisi qualitativa di descrizioni, copertine, etichette promozionali e formule ricorrenti; confronto tra dati quantitativi e interpretazione critica dei meccanismi di visibilità.

### Possibili fonti di dati

Classifiche e pagine prodotto di Amazon; Goodreads, Google Books, Open Library, WorldCat, siti di case editrici e librerie online; classifiche di vendita pubblicate da quotidiani, riviste o associazioni di settore; cataloghi editoriali, newsletter, fascette promozionali, recensioni online, book blog, social network dedicati ai libri e dataset bibliografici pubblici; eventuali raccolte costruite dagli studenti limitando il campo a un genere, un periodo, una lingua, un editore o una categoria commerciale specifica.



## 14. Tradurre con i modelli

### Ipotesi di ricerca

La traduzione automatica è ormai uno strumento di uso quotidiano, ma le sue prestazioni variano sensibilmente in base al genere testuale, al registro, alla coppia linguistica e alla presenza di tratti marcati come dialetti, varianti d'autore, lessico specialistico o riferimenti culturali. Il confronto sistematico tra modelli e tra testi permette di osservare quali errori sono ricorrenti, quali tipologie di testo restano problematiche e quale ruolo conserva l'intervento umano.

### Obiettivi

Confrontare la resa di LLM diversi e di sistemi dedicati alla traduzione (DeepL, Google Translate) sulla stessa coppia linguistica; valutare la qualità della traduzione in base al genere (letterario, tecnico, giornalistico, conversazionale) e alla presenza di tratti marcati; analizzare gli errori sistematici, dai calchi sintattici alle perdite stilistiche, dagli adattamenti culturali alle sviste lessicali; osservare il ruolo del post-editing umano e la differenza tra correzione superficiale e revisione profonda; riflettere sulla traduzione come pratica interpretativa e sul rapporto tra automazione e mediazione culturale.

### Metodi di verifica

Selezione di un piccolo corpus parallelo, anche costruito ad hoc; produzione di traduzioni con due o più sistemi; annotazione degli errori secondo una griglia tipologica (sintassi, lessico, registro, riferimenti culturali, omissioni, ipertraduzione); confronto con una traduzione umana di riferimento o con il post-editing condotto dallo studente; uso di Python per misure automatiche di qualità (BLEU, chrF, COMET o BERTScore) e per analisi lessicali, sintattiche e di lunghezza; valutazione qualitativa con esempi commentati; eventuale confronto tra coppie linguistiche più o meno rappresentate nei dati o tra varietà linguistiche di partenza differenti.

### Possibili fonti di dati

Corpora paralleli pubblici come OPUS, Europarl, Tatoeba o ParaCrawl; brani letterari con traduzioni canoniche disponibili; articoli di stampa multilingue, documenti istituzionali tradotti, sottotitoli di film o serie con licenze aperte; testi originali costruiti dagli studenti per testare specifici fenomeni; output generati da LLM e da sistemi di traduzione automatica.



## 15. Citazioni inventate e fatti incerti

### Ipotesi di ricerca

I modelli generativi tendono a produrre risposte fluenti e coerenti anche quando non dispongono di informazioni affidabili: inventano citazioni, attribuiscono brani all'autore sbagliato, costruiscono riferimenti bibliografici plausibili ma inesistenti, propongono date approssimative come fossero certe. Lo studio sistematico di queste invenzioni, della loro frequenza e delle condizioni che le innescano è particolarmente rilevante per chi lavora con fonti, riferimenti e tradizione testuale.

### Obiettivi

Misurare la frequenza con cui un modello inventa o distorce informazioni verificabili (citazioni, attribuzioni, date, riferimenti bibliografici, eventi storici, dati biografici); classificare i tipi di errore (invenzione integrale, confusione tra fonti, distorsione parziale, anacronismo, fusione di elementi distinti); osservare se e come la richiesta esplicita di indicare le fonti modifica il comportamento del modello; confrontare la risposta del modello "a memoria" con la risposta ottenuta affiancando un piccolo corpus consultabile; riflettere sul rapporto tra fluidità linguistica, autorevolezza percepita e affidabilità.

### Metodi di verifica

Costruzione di una griglia di domande di verifica su domini in cui esistono fonti certe (storia letteraria, filologia, biografia, storia delle arti, storia locale); raccolta sistematica delle risposte di uno o più modelli in condizioni controllate; verifica manuale puntuale rispetto a fonti di riferimento; classificazione degli errori secondo tipologia, gravità e plausibilità; uso di Python per organizzare il dataset, calcolare tassi di errore e confrontare condizioni sperimentali (con o senza richiesta di fonti, con o senza retrieval, con modelli diversi); analisi qualitativa di casi paradigmatici.

### Possibili fonti di dati

Repertori canonici di riferimento (manuali storici, edizioni critiche, enciclopedie consolidate, OPAC, database bibliografici); piccoli corpora costruiti dagli studenti su un autore, un'opera o un periodo; dataset pubblici su factuality e hallucination (TruthfulQA, FActScore, dataset di QA con verifica); risposte generate da uno o più LLM in condizioni controllate e annotate.



## 16. Patrimonio culturale e archivi digitali

### Ipotesi di ricerca

I modelli generativi possono entrare nei flussi di lavoro tipici del patrimonio culturale, dalla digitalizzazione di materiali storici alla descrizione di collezioni, dalla metadatazione alla costruzione di percorsi tematici. Tuttavia il loro inserimento in questi processi richiede attenzione critica: i materiali d'archivio hanno specificità linguistiche, grafiche e contestuali che mettono spesso in crisi modelli pensati per testi standard.

### Obiettivi

Esplorare un compito tipico del lavoro di archivio, biblioteca o museo, valutando in modo critico l'utilità e i limiti dei modelli; misurare la qualità del risultato confrontando l'intervento del modello con una soluzione manuale o con uno standard descrittivo esistente; identificare i tipi di errore ricorrenti e le condizioni in cui sono più frequenti; riflettere sul rapporto tra automazione, mediazione esperta e conservazione del contesto.

### Metodi di verifica

Quattro varianti di realizzazione, alternative o combinabili. Correzione di OCR su materiali storici (giornali otto-novecenteschi, manoscritti, documenti d'archivio) con un LLM nel ciclo, e confronto con il testo grezzo e con la trascrizione di riferimento. Metadatazione automatica di una piccola collezione iconografica o documentaria, con annotazione di soggetti, datazione e luoghi e confronto con schede esistenti. Arricchimento semantico di schede catalografiche già strutturate, con generazione di descrizioni, parole chiave o link tematici. Costruzione di un percorso narrativo o di un dossier tematico a partire da un fondo. In tutti i casi: definizione di metriche di qualità (accuratezza, copertura, errori di attribuzione); uso di Python per gestione e confronto del dataset; valutazione qualitativa con criteri propri della disciplina di riferimento.

### Possibili fonti di dati

Collezioni digitali aperte (Internet Archive, Europeana, Gallica, Wikimedia Commons, Biblioteca Digitale Italiana, raccolte di musei e biblioteche con licenze aperte); archivi locali resi accessibili dallo studente con autorizzazione; schede catalografiche pubbliche; standard descrittivi di riferimento (Dublin Core, ISAD-G, CIDOC-CRM); piccoli corpora costruiti dallo studente per testare un compito specifico.



## 17. Interrogare un corpus

### Ipotesi di ricerca

Quando un modello generativo viene affiancato a un corpus consultabile (pattern noto come retrieval augmented generation), le sue risposte dovrebbero diventare più fedeli alle fonti, più verificabili e più adeguate a domini specialistici. Tuttavia il modello può continuare a integrare conoscenza propria, parafrasare in modo impreciso, citare in modo selettivo o inventare anche quando il materiale di riferimento è disponibile.

### Obiettivi

Costruire un piccolo sistema di interrogazione su un corpus scelto (carteggi, opere di un autore, atti di un convegno, una rivista, un fondo d'archivio); osservare in che misura le risposte risultino fedeli alle fonti recuperate; confrontare la risposta diretta del modello con la risposta ancorata al corpus; analizzare la qualità delle citazioni e la loro corrispondenza ai passaggi originali; verificare il comportamento del sistema quando la risposta non è presente nel corpus; riflettere sul rapporto tra recupero di informazione, generazione e affidabilità.

### Metodi di verifica

Selezione e preparazione del corpus (segmentazione, indicizzazione, eventuale costruzione di embeddings); definizione di un set di domande di interrogazione di tipologia diversa (fattuali, interpretative, comparative, su contenuto assente); raccolta di risposte in condizioni controllate (con e senza retrieval, con e senza istruzione esplicita di citare, con modelli diversi); annotazione manuale di fedeltà, completezza, accuratezza delle citazioni e gestione dei casi non risolvibili; uso di Python per orchestrare il sistema (librerie come LlamaIndex o LangChain, oppure costruzione manuale del pattern), per calcolare metriche e per visualizzare i risultati; analisi qualitativa di casi paradigmatici.

### Possibili fonti di dati

Piccoli corpora costruiti dallo studente su un dominio circoscritto; testi liberi da diritti su Wikisource, Project Gutenberg, BEIC, Liber Liber; archivi tematici aperti; documenti istituzionali pubblici; materiali di studio di un singolo corso, autore o ricerca. Il corpus deve essere abbastanza piccolo da consentire la verifica manuale delle risposte.



## 18. Lingua nel tempo e nello spazio

### Ipotesi di ricerca

I modelli linguistici sono addestrati prevalentemente su lingua scritta contemporanea standard. Sulla lingua antica, sui dialetti, sulle lingue minoritarie e sulla variazione storica del lessico mostrano comportamenti più incerti, che permettono di osservare al tempo stesso i loro limiti e le caratteristiche linguistiche dei materiali studiati.

### Obiettivi

Analizzare il comportamento dei modelli su varietà linguistiche poco rappresentate; valutare la loro capacità di parafrasare, comprendere, tradurre o annotare testi non standard; osservare gli errori sistematici e le strategie di adattamento; usare la prospettiva del modello come strumento per studiare la variazione linguistica stessa, distinguendo tratti che il modello "vede" da tratti che ignora; riflettere sul rapporto tra dominanza dei dati e disuguaglianza nella copertura linguistica.

### Metodi di verifica

Tre varianti di realizzazione, alternative o combinabili. Italiano antico o lingue storiche: confronto tra il testo originale e la parafrasi o la traduzione proposta dal modello su autori canonici (Dante, Boccaccio, lingua dei volgarizzamenti, prosa cinquecentesca), con annotazione degli errori di comprensione lessicale, sintattica e culturale. Dialetti e lingue minoritarie: compiti di riconoscimento, traduzione o generazione su materiali scelti, con valutazione affidata a parlanti competenti. Variazione diacronica del lessico o dei temi: analisi su un corpus periodico (giornale, rivista, raccolta) con misure di frequenza, embedding diacronici, topic modeling o confronto tra metodi pre-LLM e basati su LLM. In tutti i casi: documentazione esplicita della copertura del corpus; uso di Python per analisi quantitative; valutazione qualitativa di esempi commentati.

### Possibili fonti di dati

Corpora di italiano antico (OVI, biblioteca digitale di Wikisource, edizioni critiche); raccolte dialettali e di lingue minoritarie (atlanti linguistici, archivi sonori trascritti, corpora regionali); periodici storici digitalizzati (Emeroteca della Biblioteca Nazionale, archivi di quotidiani); corpora annotati per variazione linguistica; materiali raccolti o annotati dallo studente con il supporto di parlanti o esperti.



## 19. Oralità sintetica

### Ipotesi di ricerca

La diffusione di sistemi di trascrizione automatica (ASR) e di sintesi vocale (TTS) sta trasformando il modo in cui il parlato viene raccolto, archiviato, analizzato e prodotto. La trascrizione automatica fa errori sistematici legati al parlato spontaneo, alle sovrapposizioni, agli accenti, al lessico specialistico; la sintesi vocale produce voci sempre più naturali ma con tratti prosodici e timbrici riconoscibili. Studiare entrambi i lati permette di osservare il rapporto tra oralità reale e sua mediazione tecnologica.

### Obiettivi

Analizzare la qualità di trascrizioni automatiche su materiali di parlato non standard (interviste, podcast, registrazioni d'archivio, lezioni); identificare gli errori ricorrenti e le loro cause linguistiche; usare le trascrizioni come base per analisi linguistiche di parlato spontaneo confrontandolo con scritto; oppure analizzare voci sintetiche su parametri prosodici, ritmo ed espressività confrontandole con voci umane; riflettere sull'oralità come dato e sul ruolo delle tecnologie nella sua produzione e conservazione.

### Metodi di verifica

Quattro varianti di realizzazione, alternative o combinabili. Trascrizione automatica: registrazione di un piccolo corpus di parlato o uso di archivi disponibili; trascrizione con uno o più modelli (Whisper, modelli dedicati); confronto con trascrizione manuale di riferimento; classificazione tipologica degli errori; misura del word error rate. Analisi linguistica di parlato spontaneo: studio di disfluenze, formule, sovrapposizioni, marche di interazione, confrontate con corpora di scritto. Voci sintetiche: confronto percettivo tra voci umane e sintetiche su parametri prosodici e di espressività, con ascoltatori umani e griglia di valutazione. Uso di LLM su trascrizioni: estrazione di temi, sintesi, annotazione di interviste, con verifica della tenuta rispetto alla specificità del parlato. In tutti i casi: uso di Python per gestione del corpus e analisi quantitative.

### Possibili fonti di dati

Archivi orali aperti (Common Voice, LibriVox, archivi sonori di università e biblioteche, Teche RAI dove disponibili); podcast con licenze aperte; trascrizioni di interviste pubblicate; registrazioni raccolte dagli studenti con consenso esplicito dei parlanti; voci sintetiche prodotte con sistemi disponibili (modelli TTS open o commerciali).



## 20. Riconoscere la mano della macchina

### Ipotesi di ricerca

La distinzione tra testi scritti da esseri umani e testi generati da modelli linguistici è oggi una questione pratica (autenticità, plagio, valutazione didattica, autorialità giornalistica) e teorica (cosa rende riconoscibile una scrittura come umana o come automatica). I sistemi di detection esistenti hanno prestazioni variabili e problematiche dichiarate, mentre l'osservazione diretta dei tratti stilistici della scrittura generata può rivelare ricorrenze utili per uno studio qualitativo.

### Obiettivi

Studiare empiricamente le prestazioni dei tool di detection esistenti su testi di vari generi e di provenienza nota; oppure costruire un piccolo classificatore stilometrico che tenti di distinguere testi umani da testi generati; oppure condurre uno studio qualitativo sulle "spie" stilistiche della scrittura generata (tic lessicali, cadenze sintattiche, formule ricorrenti, gestione dei connettivi, lunghezza media delle frasi, uniformità del registro); riflettere sui limiti del riconoscimento, sui suoi rischi (falsi positivi, bias verso parlanti non nativi) e sulle implicazioni etiche e didattiche dell'uso di questi strumenti.

### Metodi di verifica

Costruzione di un dataset bilanciato con testi umani e testi generati da uno o più LLM su temi e generi confrontabili; eventuale post-editing dei testi generati per simulare l'uso reale; valutazione di tool di detection esistenti con metriche di accuratezza, precisione, richiamo e tasso di falsi positivi; in alternativa o in aggiunta, estrazione di feature stilometriche (frequenze lessicali, lunghezza, ricchezza, ratios sintattici) e addestramento di un classificatore semplice in Python; analisi qualitativa di casi paradigmatici, in particolare di errori di classificazione; eventuale confronto tra testi nativi italiani, traduzioni e testi non nativi per discutere i bias dei sistemi.

### Possibili fonti di dati

Testi umani pubblici di sicura provenienza (corpora letterari, articoli con autore noto, elaborati didattici raccolti con consenso); testi generati da uno o più LLM in condizioni documentate; eventuali dataset pubblici di detection (HC3, M4, GPT-Wild); tool di detection esistenti (GPTZero, Originality, sistemi open source) usati come oggetto di valutazione e non come autorità.
