#### Studi Umanistici

# Tecnologie dei dati e del linguaggio

## Docente Prof. Alfio Ferrara

### Assistente Dott. Sergio Picascia

## Progetti di fine corso

Il progetto finale consiste nella preparazione di un breve studio su uno dei temi del corso, identificando una precisa domanda di ricerca e obiettivi misurabili. Il progetto proporrà una metodologia per risolvere la domanda di ricerca e fornirà una verifica sperimentale dei risultati ottenuti secondo metriche di valutazione dei risultati. L'enfasi non è sull'ottenimento di alte prestazioni ma piuttosto sulla discussione critica dei risultati ottenuti al fine di comprendere la potenziale efficacia della metodologia proposta. I risultati devono essere documentati in un breve articolo di non meno di 4 pagine e non più di 8. Gli studenti che decidano di scrivere del codice Python per realizzare parte del progetto devono anche fornire l'accesso a un repository GitHub contenente il codice e i risultati sperimentali riproducibili. Infine, il progetto sarà discusso dopo una **presentazione di 10 minuti con slides**.

## Procedura

Le date d'esame servono solo per la registrazione del voto finale. La discussione del progetto sarà fissata su appuntamento, secondo la seguente procedura:

1. Iscriversi a una qualsiasi data disponibile
2. Contattare il Prof. Ferrara non appena:
   1. Il progetto è terminato e pronto per essere discusso
   2. Dopo che la data della propria iscrizione è scaduta
3. Fissare un appuntamento e discutere il proprio lavoro 

**Esempio**: ti iscrivi alla data d'esame del [Mese] [Giorno]. **In qualsiasi momento dopo [Mese] [Giorno]**, quando il **progetto è pronto**, contatterai il Prof. Ferrara e fisserai un appuntamento. Discuterai il progetto durante l'appuntamento. 

Se si è **interessati a svolgere la tesi magistrale finale su questi argomenti**, il progetto finale può essere un lavoro preliminare in vista della tesi. In questo caso, i contenuti del progetto vanno preliminarmente discussi con il Prof. Ferrara.

## Dichiarazione sull'utilizzo dell'IA 

Parti di questi progetti sono stati sviluppati con l'assistenza di **Anthropic Claude Sonnet 3.7**. L'IA è stata utilizzata per supportare lo **sviluppo delle idee di progetto, la strutturazione dei flussi di lavoro metodologici, la stesura dei testi descrittivi** e l'**identificazione di dataset e riferimenti rilevanti**. Tutti i contenuti prodotti con l'assistenza dell'IA sono stati **attentamente rivisti, modificati e convalidati** da me come autore ultimo delle proposte di progetto. Mi assumo la piena responsabilità per il contenuto finale e la sua accuratezza, rilevanza e integrità accademica.

## Utilizzo dell'IA (per gli studenti) 

Gli strumenti di IA generativa (come ChatGPT, Claude, Mistral o modelli simili) **possono essere utilizzati in questo progetto**, sia come oggetto di indagine sia come strumento per supportare il processo di sviluppo. Gli studenti sono incoraggiati a esplorare come funzionano questi modelli, a interagire con essi in modo creativo e a sfruttarli come **ispirazione o assistenza nell'ideazione, nella stesura o nella sperimentazione**. Tuttavia, **l'IA non dovrebbe essere utilizzata come sostituto del lavoro originale**. La responsabilità per la struttura, il ragionamento e la comprensione del progetto rimane interamente dello studente. Se l'IA generativa è stata utilizzata in qualsiasi fase del progetto, è **obbligatorio includere una dichiarazione** che specifichi chiaramente:

- **Quali modelli** sono stati utilizzati (ad es., GPT-4, Claude 3, ecc.)
- **Per quali scopi** (ad es., stesura di testi, sintesi di idee, generazione di codice o esempi)
- **In che misura** gli output sono stati modificati, verificati o integrati nella presentazione finale 

Il progetto sarà valutato non solo in base al suo output, ma anche sulla **capacità dello studente di spiegare e giustificare tutte le scelte fatte**. Un **colloquio finale valuterà la profondità della comprensione**, e qualsiasi mancanza di chiarezza o eccessiva dipendenza da materiale generato dall'IA senza una corretta comprensione potrebbe influire negativamente sulla valutazione. L'IA generativa dovrebbe essere vista come uno **strumento di supporto alla creatività**, non come un sostituto del pensiero critico, della risoluzione dei problemi o dello sviluppo tecnico.

## Idee progettuali

### 1. L'eco dei poeti dimenticati

**Descrizione**: Il progetto si propone di ricreare lo stile di poeti poco conosciuti analizzando le loro opere e generando nuove composizioni che ne catturino le peculiarità stilistiche, accompagnate da un'analisi critica sul rapporto tra creatività umana e artificiale.

**Per studenti sviluppatori**: Implementare un modello di fine-tuning basato su un corpus di poesie di autori selezionati. Utilizzare tecniche di estrazione delle caratteristiche stilistiche come la distribuzione delle parti del discorso, le strutture metriche e le scelte lessicali. Confrontare i risultati ottenuti con diversi approcci (modelli di diversa dimensione, tecniche di sampling diverse) per valutare quale meglio cattura lo stile del poeta scelto.

**Per studenti non sviluppatori**: Identificare un poeta poco noto e analizzarne le caratteristiche stilistiche distintive. Utilizzare prompt engineering avanzato con modelli generativi conversazionali per ricreare lo stile, documentando i tentativi, le strategie di prompt e le iterazioni necessarie. Creare una rubrica di valutazione qualitativa per misurare la fedeltà stilistica delle poesie generate.

**Dataset possibili**:
- Corpus poetici digitalizzati come quelli disponibili su Biblioteca Italiana (bibliotecaitaliana.it)
- Raccolte di poesie di autori minori del '900 disponibili su Wikisource
- Archivi digitali di riviste letterarie storiche italiane

**References**:
1. Chakrabarty, T., Laban, P., Agarwal, D., Muresan, S., & Wu, C. S. (2024, May). Art or artifice? large language models and the false promise of creativity. In *Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems* (pp. 1-34).
2. Elgammal, A., & Saleh, B. (2015). "Quantifying creativity in art networks." *Proceedings of the International Conference on Computational Creativity*, 39-46.
3. Oliveira, H. G. (2017). "A survey on automatic poetry generation: Languages, features, techniques, reutilisation and evaluation." *Proceedings of the 10th International Conference on Natural Language Generation*, 11-20.

---

### 2. Traduttore traditore?

**Descrizione**: Il progetto si propone di esaminare come diversi sistemi di AI traducono testi letterari complessi, analizzando le sfumature culturali e linguistiche che vengono preservate o perse, con particolare attenzione ai riferimenti culturali e al linguaggio figurato.

**Per studenti sviluppatori**: Creare un sistema di valutazione automatica che confronti le traduzioni prodotte da diversi modelli di AI con traduzioni umane di riferimento. Implementare metriche personalizzate che considerino non solo la fedeltà semantica, ma anche elementi stilistici, riferimenti culturali e ambiguità intenzionali. Visualizzare e analizzare pattern di perdita/preservazione di elementi culturali specifici.

**Per studenti non sviluppatori**: Selezionare brani letterari ricchi di riferimenti culturali, metafore o giochi di parole. Sottoporre gli stessi brani a diversi sistemi di traduzione automatica e sviluppare una griglia di analisi qualitativa per confrontare i risultati. Documentare casi specifici in cui le traduzioni automatiche falliscono o riescono sorprendentemente bene, categorizzando le tipologie di errori o successi.

**Dataset possibili**:
- Testi paralleli di opere letterarie tradotte disponibili su Project Gutenberg
- Corpus di traduzioni letterarie come OPUS ([opus.nlpl.eu](https://opus.nlpl.eu/))

**References**:

1. Besacier, L., & Schwartz, L. (2015). "Automated translation of a literary work: A pilot study." *Proceedings of the Fourth Workshop on Computational Linguistics for Literature*, 114-122.
2. Matusov, E. (2019). "The challenges of using neural machine translation for literature." *Proceedings of the Qualities of Literary Machine Translation*, 10-19.
3. Toral, A., & Way, A. (2018). "What level of quality can neural machine translation attain on literary text?" *Translation Quality Assessment*, 263-287.

---

### 3. Dialoghi impossibili

**Descrizione**: Il progetto si propone di creare conversazioni fittizie tra personaggi storici di diverse epoche, esplorando come l'AI interpreta e riproduce le loro prospettive filosofiche e il loro linguaggio.

**Per studenti sviluppatori**: Implementare un sistema di generazione di dialoghi basato su un approccio a agenti multipli, dove ogni agente è fine-tuned per rappresentare un personaggio storico specifico. Creare embeddings delle opere e dei discorsi di ciascun personaggio per informare il comportamento dell'agente. Sviluppare metriche per valutare la coerenza filosofica e stilistica delle risposte generate.

**Per studenti non sviluppatori**: Selezionare personaggi storici con visioni filosofiche ben documentate ma contrastanti. Costruire prompt dettagliati che catturino l'essenza del loro pensiero e stile comunicativo. Organizzare un "dialogo" attraverso una serie di interazioni con l'AI, documentando le strategie di prompt utilizzate per mantenere la coerenza caratteriale e la plausibilità storica.

**Dataset possibili**:
- Opere complete digitalizzate di filosofi o altri personaggi storici
- Archivi di discorsi politici e interventi pubblici di figure storiche
- Corpus di corrispondenze e epistolari

**References**:

1. Chaturvedi, I., Cambria, E., Welsch, R. E., & Herrera, F. (2018). Distinguishing between facts and opinions for sentiment analysis: Survey and challenges. *Information Fusion*, *44*, 65-77.
2. Evans, R., Hernández-Orallo, J., Welbl, J., Kohli, P., & Sergot, M. (2021). "Making sense of sensibility: Using natural language processing to investigate literary criticism." *Digital Scholarship in the Humanities*, 36(1), 95-112.
3. Hermann, I. (2023). Artificial intelligence in fiction: between narratives and metaphors. *AI & society*, *38*(1), 319-329.

---

### 4. Metamorfosi testuale

**Descrizione**: Il progetto si propone di trasformare progressivamente un testo da un genere all'altro (es. da saggio accademico a racconto gotico), analizzando i cambiamenti stilistici e le caratteristiche che definiscono ciascun genere.

**Per studenti sviluppatori**: Creare un sistema di trasformazione incrementale che modifichi gradualmente un testo, controllando parametri stilistici specifici (formalità, narratività, registro emotivo, ecc.). Implementare metriche quantitative per monitorare il cambiamento dello stile attraverso le iterazioni, visualizzando la traiettoria della trasformazione in uno spazio multidimensionale delle caratteristiche stilistiche.

**Per studenti non sviluppatori**: Selezionare un testo di partenza e definire una serie di "stazioni intermedie" verso il genere di destinazione. Per ogni tappa, utilizzare prompt mirati per guidare la trasformazione di specifici elementi stilistici, documentando sia i prompt utilizzati che un'analisi delle modifiche avvenute. Creare una rubrica per valutare l'efficacia delle trasformazioni in termini di fedeltà al genere target.

**Dataset possibili**:
- Corpora di testi letterari classificati per genere
- Raccolte di saggi accademici e testi letterari di vari generi
- Dataset di classificazione di genere letterario come quelli disponibili su [Kaggle](https://www.kaggle.com/)

**References**:

1. Reiter, E., & Sripada, S. (2002). "Human variation and lexical choice." *Computational Linguistics*, 28(4), 545-553.
2. Kao, J., & Jurafsky, D. (2012). "A computational analysis of style, affect, and imagery in contemporary poetry." *Proceedings of the NAACL-HLT Workshop on Computational Linguistics for Literature*, 8-17.
3. Manjavacas, E., Karsdorp, F., Burtenshaw, B., & Kestemont, M. (2017). "Synthetic literature: Writing science fiction in a co-creative process." *Proceedings of the Workshop on Computational Creativity in Natural Language Generation*, 29-37.

---

### 5. Il Critico artificiale

**Descrizione**: Il progetto si propone di utilizzare l'AI per generare recensioni di opere letterarie o artistiche, confrontandole con quelle scritte da critici umani e analizzando pregiudizi, schemi ricorrenti e differenze qualitative.

**Per studenti sviluppatori**: Implementare un sistema che analizza un corpus di recensioni critiche professionali per estrarne pattern ricorrenti, criteri di valutazione e bias stilistici. Addestrare un modello per generare recensioni che simulino diversi approcci critici. Sviluppare metriche per valutare l'originalità e la profondità delle osservazioni generate rispetto alle recensioni umane.

**Per studenti non sviluppatori**: Selezionare un'opera artistica o letteraria con un corpus significativo di recensioni. Progettare prompt che guidino l'AI a produrre recensioni da diverse prospettive critiche. Creare una metodologia di analisi comparativa per identificare similitudini e differenze tra le recensioni generate e quelle umane, con particolare attenzione agli aspetti qualitativi come intuizioni originali, profondità interpretativa e sensibilità contestuale.

**Dataset possibili**:
- Collezioni di recensioni letterarie da riviste accademiche digitalizzate
- Database di recensioni cinematografiche come IMDb o Rotten Tomatoes
- Archivi di riviste di critica d'arte e letteraria

**References**:
1. Hemmatian, F., & Sohrabi, M. K. (2019). A survey on classification techniques for opinion mining and sentiment analysis. *Artificial intelligence review*, *52*(3), 1495-1545.
2. Pang, B., & Lee, L. (2008). "Opinion mining and sentiment analysis." *Foundations and Trends in Information Retrieval*, 2(1–2), 1-135.
3. Tsur, O., & Rappoport, A. (2009). "RevRank: A fully unsupervised algorithm for selecting the most helpful book reviews." *Proceedings of the International AAAI Conference on Web and Social Media*, 154-161.

---

### 6. Voci dal margine

**Descrizione**: Il progetto si propone di generare testi che diano voce a prospettive storicamente marginalizzate, valutando criticamente la capacità dell'AI di rappresentare esperienze diverse da quelle dominanti nel suo dataset di addestramento.

**Per studenti sviluppatori**: Implementare un sistema che analizza corpus testuali provenienti da comunità sottorappresentate per identificare caratteristiche distintive di stile, prospettiva e tematiche. Creare un meccanismo di valutazione che misuri quantitativamente i bias nella rappresentazione di diverse identità culturali e sociali. Sperimentare con tecniche di debiasing e confrontare i risultati.

**Per studenti non sviluppatori**: Identificare una voce o prospettiva storicamente emarginata. Documentare il processo di costruzione di prompt per generare testi che rappresentino autenticamente questa prospettiva, includendo feedback iterativo da fonti primarie. Sviluppare una metodologia critica per valutare l'autenticità culturale dei testi generati, confrontandoli con opere di autori appartenenti alla comunità rappresentata.

**Dataset possibili**:
- Corpus letterari di scrittori appartenenti a minoranze etniche o culturali
- Raccolte di testimonianze orali digitalizzate di gruppi storicamente marginalizzati
- Archivi digitali di pubblicazioni di movimenti sociali e culturali minoritari

**References**:
1. Naous, T., Ryan, M. J., Ritter, A., & Xu, W. (2023). Having beer after prayer? measuring cultural bias in large language models. *arXiv preprint arXiv:2305.14456*.
2. Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). "Man is to computer programmer as woman is to homemaker? Debiasing word embeddings." *Advances in Neural Information Processing Systems*, 4349-4357.
3. Durrheim, K., Schuld, M., Mafunda, M., & Mazibuko, S. (2023). Using word embeddings to investigate cultural biases. *British Journal of Social Psychology*, *62*(1), 617-629.

---

### 7. La Fabbrica delle storie

**Descrizione**: Creare un sistema che generi variazioni di una stessa trama in base a diversi parametri culturali, temporali o di genere, analizzando le diverse manifestazioni narrative e valutando la coerenza narrativa delle storie generate.

**Per studenti sviluppatori**: Implementare un generatore di storie parametrizzato che permetta di controllare elementi come ambientazione culturale, periodo storico, genere letterario e arco narrativo. Sviluppare metriche computazionali per valutare la coerenza narrativa, la plausibilità causale degli eventi e la consistenza dei personaggi attraverso le diverse variazioni. Creare una visualizzazione interattiva che mostri come cambiano gli elementi narrativi al variare dei parametri.

**Per studenti non sviluppatori**: Selezionare una trama base e definire una serie di variazioni parametriche (es. stessa storia ambientata in culture diverse o in epoche diverse). Utilizzare prompt strutturati per generare le varianti, documentando come ogni parametro influenza elementi specifici della narrazione. Sviluppare un framework analitico per valutare la coerenza narrativa interna di ciascuna versione e come gli elementi culturali o temporali modificano la causalità, la caratterizzazione e la risoluzione dei conflitti.

**Dataset possibili**:
- Database di trame narrative
- Raccolte di fiabe e racconti popolari da diverse culture
- Corpus di sceneggiature cinematografiche categorizzate per genere

**References**:

1. Riedl, M. O., & Young, R. M. (2010). "Narrative planning: Balancing plot and character." *Journal of Artificial Intelligence Research*, 39, 217-268.
2. Bamman, D., O'Connor, B., & Smith, N. A. (2013). "Learning latent personas of film characters." *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics*, 352-361.
3. León, C., & Gervás, P. (2014). "Creativity in story generation from the ground up: Non-deterministic simulation driven by narrative." *5th International Conference on Computational Creativity*, 201-208.

---

### 8. Semantica del nonsense

**Descrizione**: Esplorare i limiti della comprensione semantica dell'AI generando e analizzando testi apparentemente privi di senso ma con strutture sintattiche corrette.

**Per studenti sviluppatori**: Implementare un sistema che generi testi sintatticamente corretti ma semanticamente anomali attraverso diverse tecniche (sostituzione selettiva di parole, generazione con vincoli grammaticali ma non semantici, ecc.). Sviluppare metriche per quantificare il grado di "nonsense semantico" preservando la correttezza sintattica. Analizzare come diversi modelli di AI interpretano e tentano di dare senso a questi testi.

**Per studenti non sviluppatori**: Creare una tassonomia di diversi tipi di nonsense linguistico (es. paradossi, contraddizioni, assurdità logiche, violazioni di presupposizioni). Utilizzare prompt mirati per generare esempi di ciascuna categoria. Documentare come i modelli di AI tentano di interpretare o "riparare" il nonsense, analizzando i pattern di risposta quando vengono interrogati sul significato di testi semanticamente anomali.

**Dataset possibili**:
- Corpus di letteratura *nonsense*
- Collezioni di paradossi e giochi linguistici
- Dataset di frasi sintatticamente corrette ma semanticamente anomale

**References**:
1. Nilsen, A. P., & Nilsen, D. L. (2018). *The language of humor: An introduction*. Cambridge University Press.
2. Danescu-Niculescu-Mizil, C., & Lee, L. (2011). Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs. *arXiv preprint arXiv:1106.3077*.
3. Cherepanova, V., & Zou, J. (2024). Talking Nonsense: Probing Large Language Models' Understanding of Adversarial Gibberish Inputs. *arXiv preprint arXiv:2404.17120*.

---

### 9. Archeologia digitale

**Descrizione**: Ricostruire stili di scrittura di epoche passate e generare "reperti testuali" fittizi, discutendo le implicazioni per la ricerca storica e filologica.

**Per studenti sviluppatori**: Implementare un sistema che analizza corpus testuali di diverse epoche storiche per estrarne caratteristiche linguistiche distintive (lessico, sintassi, riferimenti culturali). Creare un generatore di "reperti" che simuli documenti di epoche specifiche con diversi gradi di deterioramento o frammentazione. Sviluppare metriche per valutare l'autenticità storica e filologica dei testi generati.

**Per studenti non sviluppatori**: Selezionare un periodo storico e un tipo di documento (es. lettere private del Rinascimento, documenti burocratici medievali). Studiare le caratteristiche linguistiche e formali del periodo e costruire prompt dettagliati per generare documenti che ne simulino lo stile. Creare un metodo di valutazione critica che consideri sia l'accuratezza storica che le potenziali implicazioni etiche della simulazione di documenti storici.

**Dataset possibili**:
- Corpora diacronici
- Archivi digitalizzati di manoscritti e documenti storici
- Collezioni di lettere e documenti d'epoca trascritti digitalmente

**References**:
1. Tahmasebi, N., Borin, L., & Jatowt, A. (2021). "Survey of computational approaches to lexical semantic change detection." *Computational Linguistics*, 47(2), 289-367.
2. Periti, F., & Montanelli, S. (2024). Lexical semantic change through large language models: a survey. *ACM Computing Surveys*, *56*(11), 1-38.
3. Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). "A large annotated corpus for learning natural language inference." *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, 632-642.

---

### 10. I giochi da tavolo sono fatti di regole

**Descrizione**: L'obiettivo del progetto è sfidare i LLM a comprendere la logica dei giochi da tavolo reali, generare regolamenti giocabili e persino inventare nuovi giochi, utilizzando l'ingegneria dei prompt, l'estrazione di regole e il ragionamento.

**Per studenti sviluppatori**: Implementare un sistema che analizzi regolamenti di giochi da tavolo esistenti per identificarne la struttura logica, le meccaniche di gioco e i pattern ricorrenti. Sviluppare un framework che permetta di rappresentare formalmente le regole di un gioco e le relazioni tra di esse. Creare un'interfaccia che consenta di testare la comprensione del modello attraverso diverse sfide: identificazione di regole mancanti o errate, generazione di nuovi giochi basati su vincoli specifici, e simulazione di stati di gioco con suggerimenti sulla mossa successiva. Implementare metriche di valutazione per misurare la coerenza, la giocabilità e l'originalità dei regolamenti generati.

**Per studenti non sviluppatori**: Selezionare una collezione di giochi da tavolo con diversi livelli di complessità. Progettare una serie di esperimenti che testino le capacità del modello in diverse attività: spiegare regole complesse in modo semplificato per diverse fasce d'età, identificare e correggere regole problematiche in versioni modificate dei regolamenti, inventare nuovi giochi basati su temi o concetti specifici, e analizzare stati di gioco per suggerire strategie. Sviluppare una rubrica di valutazione per giudicare la qualità delle risposte del modello in base a criteri come chiarezza esplicativa, correttezza logica, creatività e giocabilità pratica.

**Dataset possibili**:

- Database di regolamenti di giochi da tavolo da BoardGameGeek
- Corpus di recensioni e spiegazioni di giochi da fonti specializzate
- Raccolte di trascrizioni di spiegazioni di regole da video tutorial di giochi
- Database di meccaniche di gioco classificate e categorizzate

**References**:
1. Hu, C., Zhao, Y., & Liu, J. (2024, August). Game generation via large language models. In *2024 IEEE Conference on Games (CoG)* (pp. 1-4). IEEE.
2. Todd, G., Padula, A. G., Stephenson, M., Piette, É., Soemers, D., & Togelius, J. (2024). GAVEL: Generating games via evolution and language models. *Advances in Neural Information Processing Systems*, *37*, 110723-110745.
3. Li, D., Zhang, S., Sohn, S. S., Hu, K., Usman, M., & Kapadia, M. (2025). Cardiverse: Harnessing LLMs for Novel Card Game Prototyping. *arXiv preprint arXiv:2502.07128*.
