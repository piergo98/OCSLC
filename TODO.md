# 🌽
- Vedere come cambia il vettore di ottimizzazione ottimo durante le varie iterazioni di IPOPT.
- Vedere anche come cambiano le metriche di ottimalità.
- Giocare sulla tolleranza di feasibility per capire cosa cambia. A volte ha un andamento a gradini.

- Dividere il tempo di creazione delle matrici e inizializzazione, con il tempo di soluzione del problema di ottimo.

- Check: bloccare il vettore degli ingressi (input e fasi). Verificare quindi se a parità di soluzione, single-shooting e multiple-shooting performano in maniera uguale. Per verificare se SS e MS sono corretti.

- Verificare integrazione nel MS.

- Evidenziare (in ottica paper) che tutte le matrici ecc del problema di ottimizzazione non sono ottenute in maniera naive con l'integrazione di IPOPT o chi per lui. Bensi, sono calcolate da noi nel metodo.

- Calcolare l'integrale in maniera esatta. Poichè per sistemi lineari si può fare, e anche per sistemi non lineari noi comunque linearizziamo.

- Altro esempio diverso.