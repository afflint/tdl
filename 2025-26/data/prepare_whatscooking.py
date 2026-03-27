"""
Prepara un sottoinsieme del dataset What's Cooking per uso didattico.

Input (atteso in /Users/Flint/Data/recipes/whatscooking/):
    train.csv        — matrice one-hot (campioni × ingredienti)
    train_labels.csv — etichette (index, cuisine)

Output (nella stessa cartella dello script):
    whatscooking_small.csv — features + colonna 'cuisine'

Parametri configurabili:
    CUISINES        — cucine da mantenere
    N_PER_CLASS     — campioni per classe (campionamento stratificato)
    TOP_INGREDIENTS — ingredienti più frequenti da mantenere
    MIN_FREQ        — ingrediente deve comparire in almeno questa % di ricette
    RANDOM_STATE    — seed per riproducibilità
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Parametri ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("/Users/Flint/Data/recipes/whatscooking")
OUT_PATH = Path("/Users/Flint/Data/recipes/whatscookingwhatscooking_small.csv")

CUISINES = ["italian", "mexican", "southern_us", 
            "japanese", "indian", "french", "chinese"]

N_PER_CLASS = 250          # campioni per cucina nel sottoinsieme
TOP_INGREDIENTS = 150      # tieni solo i K ingredienti più frequenti
MIN_FREQ = 0.01            # ingrediente presente in almeno 1% delle ricette selezionate
RANDOM_STATE = 42

# ── Caricamento ────────────────────────────────────────────────────────────────

print("Carico i dati...")
X = pd.read_csv(DATA_DIR / "train.csv", index_col=0)
y = pd.read_csv(DATA_DIR / "train_labels.csv", index_col=0)

df = X.join(y)
print(f"  Dataset originale: {df.shape[0]} ricette, {X.shape[1]} ingredienti, {y['cuisine'].nunique()} cucine")

# ── Filtro cucine ──────────────────────────────────────────────────────────────

df = df[df["cuisine"].isin(CUISINES)].copy()
print(f"\nCucine selezionate: {CUISINES}")
print(df["cuisine"].value_counts().to_string())

# ── Campionamento stratificato ─────────────────────────────────────────────────

sampled = (
    df.groupby("cuisine", group_keys=False)
    .apply(lambda g: g.sample(min(N_PER_CLASS, len(g)), random_state=RANDOM_STATE))
)
print(f"\nDopo campionamento: {len(sampled)} ricette")

# ── Riduzione vocabolario ──────────────────────────────────────────────────────

feature_cols = [c for c in sampled.columns if c != "cuisine"]
X_sub = sampled[feature_cols]

# tieni solo ingredienti che compaiono almeno MIN_FREQ delle ricette selezionate
freq = X_sub.mean()
frequent = freq[freq >= MIN_FREQ].index

# tra questi, prendi i TOP_INGREDIENTS più frequenti
top_ingredients = freq[frequent].nlargest(TOP_INGREDIENTS).index.tolist()

print(f"\nIngredienti originali nel sottoinsieme: {len(feature_cols)}")
print(f"Dopo filtro frequenza (>= {MIN_FREQ:.0%}): {len(frequent)}")
print(f"Mantenuti (top {TOP_INGREDIENTS}): {len(top_ingredients)}")

# ── Pulizia finale ─────────────────────────────────────────────────────────────

result = sampled[top_ingredients + ["cuisine"]].copy()

# rimuovi eventuali righe che dopo il filtro ingredienti non hanno alcun ingrediente
rows_before = len(result)
result = result[result[top_ingredients].sum(axis=1) > 0]
if rows_before != len(result):
    print(f"  Rimosse {rows_before - len(result)} righe senza ingredienti dopo il filtro")

result = result.reset_index(drop=True)

# ── Salvataggio ────────────────────────────────────────────────────────────────

result.to_csv(OUT_PATH, index=False)

print(f"\nDataset salvato in: {OUT_PATH}")
print(f"Shape finale: {result.shape}  ({result.shape[1]-1} ingredienti + colonna 'cuisine')")
print("\nDistribuzione finale:")
print(result["cuisine"].value_counts().to_string())

# ── Statistiche utili per il notebook ─────────────────────────────────────────

print("\n── Statistiche ──")
print(f"  Ingredienti per ricetta (media): {result[top_ingredients].sum(axis=1).mean():.1f}")
print(f"  Ingredienti per ricetta (mediana): {result[top_ingredients].sum(axis=1).median():.0f}")

top10 = result[top_ingredients].sum().nlargest(10)
print(f"\n  Top 10 ingredienti più frequenti:")
for ing, cnt in top10.items():
    print(f"    {ing}: {cnt} ({cnt/len(result):.0%})")
