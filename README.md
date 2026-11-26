# Projecte 5: How much is this flat worth? (Inside Airbnb Barcelona)

Aprenentatge Computacional (UAB) · Curs 2026-27

**Equip:** Joel Leon (1750060) · David Gomez (1749837) · David Cabello (1745464)
**Grup / subgrup:** ...

## Objectiu
Predir el preu per nit dels allotjaments d'Airbnb a Barcelona i identificar quines característiques n'expliquen el valor.

## Dades
* **Font:** [Inside Airbnb – Barcelona](https://insideairbnb.com/get-the-data/) (`listings.csv.gz`).
* **Snapshot principal:** setembre de 2026 (15.236 registres, 90 columnes). 
* **Snapshot de novembre de 2025:** no hem pogut obtenir-lo (vegeu la Decisió 0.1).


## Com reproduir-ho
```bash
git clone <url-del-repo>
cd <nom-del-repo>
conda env create -f environment.yml
conda activate ml-airbnb
# descarregar les dades segons data/README.md
jupyter lab   # executar notebooks/ en ordre
```

## Estructura del repositori
```
data/        dades
src/         codi reutilitzable
notebooks/   un per cada checkpoint setmanal
figures/     gràfics usats al decision log
```

# Decision log
Format de cada entrada: **Problema → Proposta → Evidència → Decisió**.

## Setmana 1: exploració, split i baseline

### Decisió 0.1: ús exclusiu del snapshot de setembre de 2026
* **Problema:** no hem pogut obtenir el snapshot de novembre de 2025 i, segons el material del curs, mesura el preu llistat, mentre que el de 2026 mesura el preu d'una estada de mostra.
* **Proposta:** treballar només amb setembre de 2026.
* **Evidència:** [pendent: comparar la definició del preu als dos snapshots, si arribem a tenir-ne l'accés].
* **Decisió:** [pendent]

## Preguntes per a la setmana 2
1. Quin impacte té tractar els outliers extrems (p. ex. percentil 99) sobre el rendiment?
2. Millora la predicció si substituïm el barri categòric per la ubicació exacta (KNN geogràfic)?
3. Quin model (Random Forest / boosting vs Ridge) captura millor les interaccions entre capacitat, tipus d'allotjament i ubicació?

## Setmana 2 · Setmana 3 · Setmana 4
(pendents)

## Resultats i valoració crítica
(pendent)
