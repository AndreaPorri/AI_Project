# Project Artificial Intelligence
Obbiettivo:
Generare, a partire da una mistura di PDF, nuovi dati artificiali casuali i quali forniscono
output coerenti ai risultati di l’MLP addestrata su un set di dati supervisionati.
Passaggi chiave:
- La prima cosa da fare è prendere un dataset di regressione da UCI o
altro repository, monodimensionale o riducibile a 1-dim tramite
dimensionality reduction.
- Addestrare l’ MLP su un problema supervisionato da R1 -> R1 mediante
backpropagation classica (oppure dimensionality reduction ad 1D), testando
(empiricamente) vari modelli in modo da selezionare quello più preciso.
- Per l’analisi non è richiesto l’utilizzo di una k-cross validation e grid search, ma
semplicemente verrà adottato un metodo trial & error a cui seguirà la verifica degli
errori (anche a livello grafico) per osservare cosa si ottiene cambiando:
● architettura
● learning rate
● numero di epoche
● activation function
● ecc.
- Dopo aver trovato l’MLP che fornisce risultati soddisfacenti (consigliato su dataset
1D), viene congelata e si passa alla stima della PDF.
● Per realizzarla verrà utilizzata una mixture-density di gaussiane per stimare la
distribuzione di probabilità degli input.
● Verrà poi realizzato un grafico in R1 rispetto ai dati, per capire la loro
attendibilità.
● Verrà fatta una prova con 8, 16, 32 gaussiane dipendentemente dalla
complessità del problema, o anche 4 se non è distribuita in modo complicato.

- Una volta in possesso della mistura, essa verrà usata in modo generativo.
● Si creeranno dunque dei nuovi dati di input artificiali per la rete MLP
addestrata precedentemente, generati a caso da una componente casuale
della mistura delle gaussiane.
● Questi dati saranno sì casuali, ma casuali secondo la PDF scelta.
- Per generare “n nuovi pattern artificiali” dovremo seguire il seguente algoritmo:
1. Inizializzo contatore: i = 1
2. Generare un numero casuale uniforme tra 0 e 1.
3. Per selezionare la componente della mistura di gaussiane, si dovrà:
a. In base ai mixing parameter(che definisce il range della componente)
e al valore generato al passo precedente, dovremo estrarre la
componente della mistura nel cui intervallo ricade il numero casuale
generato nel punto 2.
b. Esempio: si hanno tre componenti con parametri:
1. 0.2 → Gaussiana 1
2. 0.1 → Gaussiana 2
3. 0.7 → Gaussiana 3
4. Il numero random è 0.27
Si seleziona la Gaussiana 2 da cui viene estratto il pattern.

4. Dopo aver selezionato la gaussiana andremo a generare una variabile
aleatoria, che corrisponderà al nostro nuovo pattern artificiale.
5. Incremento il contatore: i +=1.
6. If i < n:

Torna al punto 2.
else:
termina.

- A questo punto vengono forniti questi dati artificiali all'MLP e si verifica che gli output
ottenuti siano coerenti rispetto alla PDF degli output dei dati originali.
- Dunque, questo processo generativo viene reiterato (come nell’algoritmo descritto
sopra) e mediante vari grafici con errori, confronti, ecc. si può valutare la consistenza
dei dati artificiali, ovvero che quest’ultimi si comportino come i dati forniti. Infine
verranno fatte conclusioni e paper.
