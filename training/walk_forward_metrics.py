from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def evaluate_prediction_file(path):
    df = pd.read_parquet(path)

    required = {"label", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    return {
        "rows": len(df),
        "accuracy": accuracy_score(df["label"], df["prediction"]),
        "macro_f1": f1_score(
            df["label"],
            df["prediction"],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
    }


def main():
    prediction_dir = Path("data/processed/walk_forward")
    rows = []

    for fold_id in range(1, 6):
        path = prediction_dir / f"fold_{fold_id}_predictions.parquet"

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        metrics = evaluate_prediction_file(path)
        rows.append({"fold": fold_id, **metrics})

    if not rows:
        print("No fold prediction files found. Train/evaluate each fold first.")
        return

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    result.to_csv(
        prediction_dir / "walk_forward_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()