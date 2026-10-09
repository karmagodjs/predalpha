from pathlib import Path
import json
import hashlib

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from pipeline_v2.models.recurrent_models import SmallLSTM


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "data" / "models" / "phase19" / "recurrent" / "best_lstm.pt"
TEST_PATH = ROOT / "data" / "clean_v2" / "07_scaled" / "expanded_collection" / "test_scaled.npz"
OUTPUT_DIR = ROOT / "data" / "models" / "phase19" / "final_test"

CLASS_NAMES = ["DOWN", "FLAT", "UP"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("PHASE 19C — FINAL LOCKED TEST EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Verify artifacts
    # ------------------------------------------------------------------
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
    assert TEST_PATH.exists(), f"Missing test set: {TEST_PATH}"

    print(f"Model: {MODEL_PATH}")
    print(f"Test : {TEST_PATH}")

    # ------------------------------------------------------------------
    # 2. Load TEST ONLY — X_imputed, exactly as Phase 19B used train/val
    # ------------------------------------------------------------------
    data = np.load(TEST_PATH, allow_pickle=False)

    X_test = data["X_imputed"].astype(np.float32)
    y_raw = data["y"]

    assert X_test.shape == (1582, 10, 11), X_test.shape
    assert len(y_raw) == 1582

    assert not np.isnan(X_test).any(), "NaN detected in X_test"
    assert not np.isinf(X_test).any(), "Inf detected in X_test"

    label_to_int = {
        "DOWN": 0,
        "FLAT": 1,
        "UP": 2,
    }

    y_test = np.array(
        [label_to_int[str(y)] for y in y_raw],
        dtype=np.int64,
    )

    print(f"Test shape: {X_test.shape}")
    print(f"Test labels: {len(y_test)}")

    # ------------------------------------------------------------------
    # 3. Load FROZEN LSTM checkpoint
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["model_type"] == "SmallLSTM"
    assert checkpoint["input_size"] == 11
    assert checkpoint["hidden_size"] == 32
    assert checkpoint["num_layers"] == 1
    assert checkpoint["num_classes"] == 3
    assert checkpoint["parameter_count"] == 5859

    model = SmallLSTM(
        input_size=11,
        hidden_size=32,
        num_layers=1,
        num_classes=3,
        dropout_p=0.10,
        seed=checkpoint["seed"],
    )

    model.load_checkpoint(MODEL_PATH)

    # ------------------------------------------------------------------
    # 4. FINAL inference — NO training, NO tuning
    # ------------------------------------------------------------------
    model.model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model.model(X_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2],
    )

    per_class = {}

    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    results = {
        "phase": "19C",
        "evaluation": "FINAL_LOCKED_TEST",
        "test_set_locked_before_evaluation": True,
        "test_samples": int(len(y_test)),
        "sequence_shape": list(X_test.shape),
        "model": {
            "type": checkpoint["model_type"],
            "seed": checkpoint["seed"],
            "input_size": checkpoint["input_size"],
            "hidden_size": checkpoint["hidden_size"],
            "num_layers": checkpoint["num_layers"],
            "dropout": checkpoint["dropout_p"],
            "parameters": checkpoint["parameter_count"],
            "best_epoch": checkpoint["best_epoch"],
            "best_val_loss": checkpoint["best_val_loss"],
        },
        "metrics": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "provenance": {
            "model_sha256": sha256(MODEL_PATH),
            "test_sha256": sha256(TEST_PATH),
        },
    }

    # ------------------------------------------------------------------
    # 6. Save separate final-test artifacts
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "final_test_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Print final result
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST RESULT")
    print("=" * 70)

    print(f"Accuracy           : {accuracy:.4%}")
    print(f"Balanced Accuracy  : {balanced_accuracy:.4%}")
    print(f"Macro F1           : {macro_f1:.4f}")
    print(f"Weighted F1        : {weighted_f1:.4f}")

    print("\nPer-class:")
    for name in CLASS_NAMES:
        m = per_class[name]
        print(
            f"{name:>5}: "
            f"Precision={m['precision']:.4f} "
            f"Recall={m['recall']:.4f} "
            f"F1={m['f1']:.4f} "
            f"N={m['support']}"
        )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nArtifacts:")
    print(output_file)

    print("\nTEST EVALUATION COMPLETE.")
    print("No training or tuning was performed.")
    print("=" * 70)


if __name__ == "__main__":
    main()