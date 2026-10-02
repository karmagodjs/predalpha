from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PREDICTION_DIR = Path("data/processed/walk_forward")

def score_frame(df):
    return {
        "rows": len(df),
        "accuracy": accuracy_score(df["label"], df["prediction"]),
        "macro_f1": f1_score(
            df["label"], df["prediction"],
            average="macro", zero_division=0
        ),
    }

def main():
    files = sorted(PREDICTION_DIR.glob("fold_*_predictions.parquet"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {PREDICTION_DIR}")

    results = []

    for path in files:
        df = pd.read_parquet(path)

        if not {"label", "prediction", "split"}.issubset(df.columns):
            print(f"Skipping {path.name}: requires label, prediction, split")
            continue

        for split_name in ["train", "validation", "test"]:
            part = df[df["split"] == split_name]
            if part.empty:
                continue

            results.append({
                "fold": path.stem,
                "split": split_name,
                **score_frame(part),
            })

    if not results:
        raise ValueError("No usable split-wise prediction records found.")

    report = pd.DataFrame(results)
    print(report.to_string(index=False))

    output = PREDICTION_DIR / "overfit_report.csv"
    report.to_csv(output, index=False)
    print(f"\nSaved: {output}")

if __name__ == "__main__":
    main()