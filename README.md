# Reti_Neurali

La rete neurale convoluzionale (CNN) in questione è progettata per eseguire un'operazione di regressione su un set di dati.
## Struttura

Il modello è composto da diversi strati:

1. **Conv1D**: Questo è uno strato convoluzionale 1D. Questo strato ha 128 filtri e una dimensione del kernel di 3. La funzione di attivazione utilizzata è 'relu'. Questo strato è il primo strato del modello e quindi richiede l'input_shape specificato.

2. **MaxPooling1D**: Questo è uno strato di pooling che riduce la dimensione spaziale dell'input. Il fattore di pooling è 2.

3. **Conv1D**: Questo è un altro strato convoluzionale 1D con 256 filtri e una dimensione del kernel di 3. La funzione di attivazione utilizzata è 'relu'.

4. **Conv1D**: Questo è un altro strato convoluzionale 1D con 256 filtri e una dimensione del kernel di 3. La funzione di attivazione utilizzata è 'relu'.

5. **MaxPooling1D**: Questo è un altro strato di pooling con un fattore di pooling di 2.

6. **GlobalMaxPooling1D**: Questo strato applica un'operazione di pooling massimo su tutto l'input.

7. **Dense**: Questo è uno strato completamente connesso con 256 nodi. La funzione di attivazione utilizzata è 'relu'.

8. **Dropout**: Questo strato applica una regolarizzazione con Dropout, spegnendo casualmente un certo numero di nodi durante l'addestramento per prevenire l'overfitting. Il tasso di dropout è 0.1.

9. **Dense**: Questo è lo strato di output del modello. Ha un solo nodo, poiché il modello esegue un'operazione di regressione. Non c'è funzione di attivazione specificata per questo strato.

## Utilizzo

Dopo la compilazione, il modello è pronto per essere addestrato sui dati; Dopo l'addestramento, può essere utilizzato per fare previsioni su nuovi dati.
