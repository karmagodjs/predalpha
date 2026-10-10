from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from pipeline_v2.models.recurrent_models import SmallLSTM


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "data/models/phase19/recurrent/best_lstm.pt"
TRAIN_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/train_scaled.npz"
VAL_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/validation_scaled.npz"

OUT_DIR = ROOT / "data/models/phase20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["DOWN", "FLAT", "UP"]
LABEL_TO_ID = {"DOWN": 0, "FLAT": 1, "UP": 2}


def load_npz(path):
    z = np.load(path, allow_pickle=False)

    X = z["X_imputed"].astype(np.float32)
    y_raw = z["y"].astype(str)
    endpoints = z["endpoints"]

    y = np.array([LABEL_TO_ID[x] for x in y_raw], dtype=np.int64)

    return X, y, y_raw, endpoints


def evaluate(model, X, y):
    model.model.eval()

    with torch.no_grad():
        logits = model.model(
            torch.from_numpy(X)
        )

        probs = torch.softmax(logits, dim=1).numpy()

    preds = probs.argmax(axis=1)

    return preds, probs


def main():

    print("=" * 70)
    print("PHASE 20 — VALIDATION DIAGNOSTICS")
    print("=" * 70)

    print("\nLoading validation data...")
    X_val, y_val, y_val_raw, val_endpoints = load_npz(VAL_PATH)

    print("Validation:", X_val.shape)

    print("\nLoading train data...")
    X_train, y_train, y_train_raw, train_endpoints = load_npz(TRAIN_PATH)

    print("Train:", X_train.shape)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = SmallLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout_p=checkpoint["dropout_p"],
        seed=checkpoint["seed"],
        lr=checkpoint["lr"],
        weight_decay=checkpoint["weight_decay"],
        batch_size=checkpoint["batch_size"],
        max_epochs=checkpoint.get("max_epochs", 150),
    )

    model.load_checkpoint(MODEL_PATH)

    # ------------------------------------------------------------
    # VALIDATION PREDICTIONS
    # ------------------------------------------------------------

    preds, probs = evaluate(model, X_val, y_val)

    confidence = probs.max(axis=1)

    print("\n" + "=" * 70)
    print("VALIDATION PERFORMANCE")
    print("=" * 70)

    print(
        classification_report(
            y_val,
            preds,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_val, preds))

    # ------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION CONFIDENCE")
    print("=" * 70)

    print(f"Mean confidence   : {confidence.mean():.4f}")
    print(f"Median confidence : {np.median(confidence):.4f}")
    print(f"P10 confidence    : {np.percentile(confidence, 10):.4f}")
    print(f"P90 confidence    : {np.percentile(confidence, 90):.4f}")

    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:

        coverage = np.mean(confidence >= threshold)

        if coverage > 0:
            accuracy = np.mean(
                preds[confidence >= threshold]
                == y_val[confidence >= threshold]
            )
        else:
            accuracy = 0.0

        print(
            f"Confidence >= {threshold:.2f}: "
            f"coverage={coverage*100:.2f}% "
            f"accuracy={accuracy*100:.2f}%"
        )

    # ------------------------------------------------------------
    # CLASS PREDICTION DISTRIBUTION
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTIONS")
    print("=" * 70)

    print("Actual:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name}: "
            f"{np.sum(y_val == i)} "
            f"({np.mean(y_val == i)*100:.2f}%)"
        )

    print("\nPredicted:")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name}: "
            f"{np.sum(preds == i)} "
            f"({np.mean(preds == i)*100:.2f}%)"
        )

    # ------------------------------------------------------------
    # PER-CLASS CONFIDENCE
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFIDENCE BY ACTUAL CLASS")
    print("=" * 70)

    for i, name in enumerate(CLASS_NAMES):

        mask = y_val == i

        print(
            f"{name}: "
            f"mean={confidence[mask].mean():.4f}, "
            f"median={np.median(confidence[mask]):.4f}"
        )

    # ------------------------------------------------------------
    # MARKET-LEVEL ANALYSIS
    # ------------------------------------------------------------

    # endpoints are unique sequence endpoints.
    # Market IDs are not guaranteed to be present in NPZ,
    # so this diagnostic intentionally does not invent them.
    #
    # We therefore save sequence-level predictions/confidence
    # for subsequent market mapping from upstream metadata.

    results = {
        "validation_shape": list(X_val.shape),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "confidence_p10": float(np.percentile(confidence, 10)),
        "confidence_p90": float(np.percentile(confidence, 90)),
        "actual_distribution": {
            CLASS_NAMES[i]: int(np.sum(y_val == i))
            for i in range(3)
        },
        "predicted_distribution": {
            CLASS_NAMES[i]: int(np.sum(preds == i))
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
        "confidence_analysis": {}
    }

    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:

        mask = confidence >= threshold

        results["confidence_analysis"][str(threshold)] = {
            "coverage": float(mask.mean()),
            "accuracy": (
                float(np.mean(preds[mask] == y_val[mask]))
                if mask.any()
                else None
            )
        }

    np.savez(
        OUT_DIR / "validation_predictions.npz",
        y=y_val,
        predictions=preds,
        probabilities=probs,
        confidence=confidence,
        endpoints=val_endpoints
    )

    with open(
        OUT_DIR / "diagnostic_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nArtifacts:")
    print(OUT_DIR / "diagnostic_results.json")
    print(OUT_DIR / "validation_predictions.npz")

    print("\n" + "=" * 70)
    print("PHASE 20 DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()