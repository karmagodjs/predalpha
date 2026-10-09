from pathlib import Path
import json
import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from pipeline_v2.models.recurrent_models import SmallLSTM


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/train_scaled.npz"
VAL_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/validation_scaled.npz"

OUT_DIR = ROOT / "data/models/phase21"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["DOWN", "FLAT", "UP"]
LABEL_TO_ID = {"DOWN": 0, "FLAT": 1, "UP": 2}

WEIGHTS = {
    "baseline": [1.0, 1.0, 1.0],
    "flat_1.25": [1.0, 1.25, 1.0],
    "flat_1.50": [1.0, 1.50, 1.0],
    "flat_1.75": [1.0, 1.75, 1.0],
    "flat_2.00": [1.0, 2.00, 1.0],
}

SEEDS = [42, 123, 999]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path):
    z = np.load(path, allow_pickle=False)

    X = z["X_imputed"].astype(np.float32)
    y_raw = z["y"].astype(str)

    y = np.array(
        [LABEL_TO_ID[label] for label in y_raw],
        dtype=np.int64,
    )

    return X, y


def train_one(X_train, y_train, X_val, y_val, weights, seed):

    set_seed(seed)

    model = SmallLSTM(
        input_size=11,
        hidden_size=32,
        num_layers=1,
        num_classes=3,
        dropout_p=0.10,
        seed=seed,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=150,
    )

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        patience=25,
    )

    model.model.eval()

    with torch.no_grad():
        logits = model.model(
            torch.from_numpy(X_val)
        )

        predictions = logits.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_val, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_val,
        predictions,
    )
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
    )

    cm = confusion_matrix(
        y_val,
        predictions,
    )

    return {
        "seed": seed,
        "weights": weights,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }


def main():

    print("=" * 70)
    print("PHASE 21 — CLASS-WEIGHTED LSTM")
    print("=" * 70)

    X_train, y_train = load_data(TRAIN_PATH)
    X_val, y_val = load_data(VAL_PATH)

    print(f"\nTrain: {X_train.shape}")
    print(f"Validation: {X_val.shape}")

    all_results = []

    total = len(WEIGHTS) * len(SEEDS)
    current = 0

    for name, weights in WEIGHTS.items():

        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {name}")
        print(f"Class weights: {weights}")
        print("=" * 70)

        experiment_results = []

        for seed in SEEDS:

            current += 1

            print(
                f"\n[{current}/{total}] "
                f"{name} | seed={seed}"
            )

            result = train_one(
                X_train,
                y_train,
                X_val,
                y_val,
                weights,
                seed,
            )

            experiment_results.append(result)
            all_results.append({
                "experiment": name,
                **result,
            })

            print(
                f"Accuracy: {result['accuracy']*100:.2f}% | "
                f"Balanced: {result['balanced_accuracy']*100:.2f}% | "
                f"Macro F1: {result['macro_f1']:.4f}"
            )

        mean_accuracy = np.mean(
            [r["accuracy"] for r in experiment_results]
        )

        std_accuracy = np.std(
            [r["accuracy"] for r in experiment_results]
        )

        mean_balanced = np.mean(
            [r["balanced_accuracy"] for r in experiment_results]
        )

        mean_macro_f1 = np.mean(
            [r["macro_f1"] for r in experiment_results]
        )

        std_macro_f1 = np.std(
            [r["macro_f1"] for r in experiment_results]
        )

        print("\nAGGREGATE")
        print(
            f"Accuracy: "
            f"{mean_accuracy*100:.2f}% ± "
            f"{std_accuracy*100:.2f}%"
        )

        print(
            f"Balanced Accuracy: "
            f"{mean_balanced*100:.2f}%"
        )

        print(
            f"Macro F1: "
            f"{mean_macro_f1:.4f} ± "
            f"{std_macro_f1:.4f}"
        )

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    summary = []

    for name in WEIGHTS:

        rows = [
            r for r in all_results
            if r["experiment"] == name
        ]

        summary.append({
            "experiment": name,
            "weights": WEIGHTS[name],
            "mean_accuracy": float(
                np.mean([r["accuracy"] for r in rows])
            ),
            "std_accuracy": float(
                np.std([r["accuracy"] for r in rows])
            ),
            "mean_balanced_accuracy": float(
                np.mean(
                    [r["balanced_accuracy"] for r in rows]
                )
            ),
            "mean_macro_f1": float(
                np.mean([r["macro_f1"] for r in rows])
            ),
            "std_macro_f1": float(
                np.std([r["macro_f1"] for r in rows])
            ),
        })

    summary.sort(
        key=lambda x: x["mean_macro_f1"],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("PHASE 21 FINAL VALIDATION COMPARISON")
    print("=" * 70)

    for rank, row in enumerate(summary, 1):

        print(
            f"{rank}. {row['experiment']:12s} | "
            f"Acc={row['mean_accuracy']*100:.2f}% | "
            f"BalAcc={row['mean_balanced_accuracy']*100:.2f}% | "
            f"MacroF1={row['mean_macro_f1']:.4f}"
        )

    results = {
        "phase": "21",
        "purpose": "class_weighted_lstm",
        "test_used": False,
        "experiments": all_results,
        "summary_ranked_by_macro_f1": summary,
    }

    with open(
        OUT_DIR / "class_weighted_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\nArtifact:")
    print(OUT_DIR / "class_weighted_results.json")

    print("\n" + "=" * 70)
    print("PHASE 21 COMPLETE")
    print("TEST SET REMAINED LOCKED")
    print("=" * 70)


if __name__ == "__main__":
    main()