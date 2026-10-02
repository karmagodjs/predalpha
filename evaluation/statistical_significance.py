from pathlib import Path
import numpy as np
import pandas as pd

PREDICTION_DIR = Path("data/processed/walk_forward")
N_BOOTSTRAPS = 2000
RANDOM_SEED = 42

def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def block_bootstrap_accuracy(y_true, y_pred, block_size=5):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Invalid or empty predictions.")

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    scores = []

    for _ in range(N_BOOTSTRAPS):
        sampled = []
        while len(sampled) < n:
            start = rng.integers(0, n)
            sampled.extend((start + i) % n for i in range(block_size))
        idx = np.asarray(sampled[:n])
        scores.append(accuracy(y_true[idx], y_pred[idx]))

    return np.quantile(scores, [0.025, 0.50, 0.975])

def main():
    files = sorted(PREDICTION_DIR.glob("fold_*_predictions.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No OOS prediction files found in {PREDICTION_DIR}"
        )

    frames = []
    for path in files:
        df = pd.read_parquet(path)
        required = {"label", "prediction"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} must contain {required}")
        frames.append(df[["label", "prediction"]])

    data = pd.concat(frames, ignore_index=True)
    low, median, high = block_bootstrap_accuracy(
        data["label"], data["prediction"]
    )

    print(f"Out-of-sample rows: {len(data)}")
    print(f"Accuracy: {accuracy(data.label, data.prediction):.4f}")
    print(f"Bootstrap 95% interval: [{low:.4f}, {high:.4f}]")
    print("Note: interval is descriptive; it is not proof of trading profitability.")

if __name__ == "__main__":
    main()