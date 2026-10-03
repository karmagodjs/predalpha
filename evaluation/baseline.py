from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

BASE = Path("data/processed/splits")

train = pd.read_parquet(BASE / "train.parquet")
test = pd.read_parquet(BASE / "test.parquet")

majority_class = train["label"].value_counts().idxmax()
predictions = [majority_class] * len(test)

print("Baseline prediction:", majority_class)
print("Test accuracy:", accuracy_score(test["label"], predictions))

print("\nClassification report:")
print(classification_report(
    test["label"],
    predictions,
    labels=["DOWN", "FLAT", "UP"],
    zero_division=0,
))