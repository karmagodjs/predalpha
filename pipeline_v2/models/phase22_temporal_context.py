from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Ensure Windows Anaconda Library/bin is in DLL search path for scipy/sklearn
if sys.platform == "win32":
    for p in [r"C:\ProgramData\anaconda3\Library\bin", r"C:\ProgramData\anaconda3\Scripts"]:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except (AttributeError, OSError):
                pass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from pipeline_v2.models.recurrent_models import SmallLSTM


BASE = Path("data/clean_v2/07_scaled")
OUT = Path("data/models/phase22")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 999]
LOOKBACKS = [20, 30, 60]

LABEL_TO_ID = {"DOWN": 0, "FLAT": 1, "UP": 2}
ID_TO_LABEL = {0: "DOWN", 1: "FLAT", 2: "UP"}

# Fixed Phase 19B L10 Validation Baseline
L10_BASELINE = {
    "accuracy_mean": 0.5910,
    "accuracy_std": 0.0006,
    "balanced_accuracy_mean": 0.5385,
    "balanced_accuracy_std": 0.0031,
    "macro_f1_mean": 0.5543,
    "macro_f1_std": 0.0043,
}


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load scaled sequence NPZ split without touching test data.
    Ensures:
    - X is float32
    - labels are converted to int64 (DOWN=0, FLAT=1, UP=2)
    - NaNs/Infs in X are zero-imputed strictly at the model-input boundary
    - No rows deleted, no interpolation or backfill
    """
    assert "test" not in path.name.lower(), f"Forbidden: attempted to load test partition {path}"
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y_raw = z["y"]

    if y_raw.dtype.kind in ("U", "S", "O"):
        y = np.array([LABEL_TO_ID[str(v).strip().upper()] for v in y_raw], dtype=np.int64)
    else:
        y = np.asarray(y_raw, dtype=np.int64)

    # Causal warm-up handling: zero-impute ONLY at model-input boundary
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    assert not np.isnan(X).any(), f"NaN detected in {path}"
    assert not np.isinf(X).any(), f"Inf detected in {path}"
    assert len(X) == len(y), f"Length mismatch in {path}: {len(X)} != {len(y)}"

    return X, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: SmallLSTM, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate trained model on integer labels in evaluation mode.
    """
    pred_ids = model.predict(X)
    y_ids = np.asarray(y, dtype=np.int64)

    acc = float(accuracy_score(y_ids, pred_ids))
    bal_acc = float(balanced_accuracy_score(y_ids, pred_ids))
    m_f1 = float(f1_score(y_ids, pred_ids, average="macro"))

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(m_f1, 4),
    }


def generate_report(results: Dict[str, Any]) -> str:
    lines = [
        "# Phase 22 — Temporal Context Validation Report",
        "",
        "> [!IMPORTANT]",
        "> **VALIDATION-ONLY EVALUATION**: Test partition was NEVER loaded or evaluated.",
        "> **STATUS**: Phase 22 Validation Experiments Completed.",
        "",
        "## 1. Summary Comparison: L10 Baseline vs. Longer Lookback Contexts",
        "",
        "| Lookback Window | Model | Val Accuracy (Mean ± Std) | Val Balanced Accuracy (Mean ± Std) | Val Macro F1 (Mean ± Std) | Δ Macro F1 vs L10 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
        f"| **L10 Baseline** | SmallLSTM | {L10_BASELINE['accuracy_mean']*100:.2f}% ± {L10_BASELINE['accuracy_std']*100:.2f}% | {L10_BASELINE['balanced_accuracy_mean']*100:.2f}% ± {L10_BASELINE['balanced_accuracy_std']*100:.2f}% | {L10_BASELINE['macro_f1_mean']:.4f} ± {L10_BASELINE['macro_f1_std']:.4f} | baseline |",
    ]

    for L in LOOKBACKS:
        res = results[f"L{L}"]
        m_acc = res["mean"]["accuracy"]
        s_acc = res["std"]["accuracy"]
        m_bal = res["mean"]["balanced_accuracy"]
        s_bal = res["std"]["balanced_accuracy"]
        m_f1 = res["mean"]["macro_f1"]
        s_f1 = res["std"]["macro_f1"]
        delta_f1 = m_f1 - L10_BASELINE["macro_f1_mean"]

        lines.append(
            f"| **L{L}** | SmallLSTM | {m_acc*100:.2f}% ± {s_acc*100:.2f}% | {m_bal*100:.2f}% ± {s_bal*100:.2f}% | {m_f1:.4f} ± {s_f1:.4f} | {delta_f1:+.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Seed-by-Seed Performance Breakdown",
        "",
        "| Context | Seed | Val Accuracy | Val Balanced Accuracy | Val Macro F1 |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ])

    for L in LOOKBACKS:
        res = results[f"L{L}"]
        for s in res["seeds"]:
            lines.append(
                f"| L{L} | `{s['seed']}` | {s['accuracy']*100:.2f}% | {s['balanced_accuracy']*100:.2f}% | {s['macro_f1']:.4f} |"
            )

    # Find best across lookbacks
    best_acc_L = max(LOOKBACKS, key=lambda l: results[f"L{l}"]["mean"]["accuracy"])
    best_bal_L = max(LOOKBACKS, key=lambda l: results[f"L{l}"]["mean"]["balanced_accuracy"])
    best_f1_L = max(LOOKBACKS, key=lambda l: results[f"L{l}"]["mean"]["macro_f1"])

    lines.extend([
        "",
        "---",
        "",
        "## 3. Analysis & Key Findings",
        "",
        f"- **Best Mean Validation Accuracy**: **L{best_acc_L}** ({results[f'L{best_acc_L}']['mean']['accuracy']*100:.2f}%)",
        f"- **Best Mean Balanced Accuracy**: **L{best_bal_L}** ({results[f'L{best_bal_L}']['mean']['balanced_accuracy']*100:.2f}%)",
        f"- **Best Mean Macro F1**: **L{best_f1_L}** ({results[f'L{best_f1_L}']['mean']['macro_f1']:.4f})",
        "",
        "### Temporal Context Verdict:",
    ])

    best_f1_val = results[f"L{best_f1_L}"]["mean"]["macro_f1"]
    if best_f1_val > L10_BASELINE["macro_f1_mean"] + 0.005:
        verdict = f"Longer temporal context (L{best_f1_L}) provides a meaningful improvement over L10 (+{best_f1_val - L10_BASELINE['macro_f1_mean']:.4f} Macro F1)."
    elif best_f1_val >= L10_BASELINE["macro_f1_mean"]:
        verdict = f"Longer temporal context yields marginal or comparable performance relative to L10 baseline ({best_f1_val:.4f} vs {L10_BASELINE['macro_f1_mean']:.4f})."
    else:
        verdict = f"Longer temporal context does NOT improve over the L10 baseline ({best_f1_val:.4f} vs {L10_BASELINE['macro_f1_mean']:.4f}). L10 remains the superior or more parsimonious lookback window."

    lines.append(f"- {verdict}")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Verification Check",
        "- **Test partition touched**: **NO** (test_scaled.npz was completely locked and unread)",
        "- **Phase 22 Status**: **PASS**",
    ])

    return "\n".join(lines)


def main() -> None:
    all_results = {}

    for L in LOOKBACKS:
        scaled = BASE / f"expanded_L{L}"

        train_path = scaled / "train_scaled.npz"
        val_path = scaled / "validation_scaled.npz"

        X_train, y_train = load_split(train_path)
        X_val, y_val = load_split(val_path)

        print(f"\n{'=' * 70}")
        print(f"L={L}")
        print(f"Train: {X_train.shape}, y: {y_train.shape}, class counts: {np.bincount(y_train)}")
        print(f"Val:   {X_val.shape}, y: {y_val.shape}, class counts: {np.bincount(y_val)}")
        print(f"{'=' * 70}")

        seed_results = []

        for seed in SEEDS:
            print(f"\n--- Fitting SmallLSTM L={L} (Seed {seed}) ---")

            set_seed(seed)

            model = SmallLSTM(
                input_size=11,
                hidden_size=32,
                num_layers=1,
                num_classes=3,
                dropout_p=0.10,
                lr=1e-3,
                weight_decay=1e-4,
                batch_size=256,
                max_epochs=150,
                seed=seed,
            )

            model.fit(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                patience=25,
            )

            metrics = evaluate(model, X_val, y_val)
            metrics["seed"] = seed
            seed_results.append(metrics)

            print(
                f"Seed {seed} -> Accuracy: {metrics['accuracy']:.4f} | "
                f"Balanced: {metrics['balanced_accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f}"
            )

        mean_acc = float(np.mean([r["accuracy"] for r in seed_results]))
        std_acc = float(np.std([r["accuracy"] for r in seed_results]))
        mean_bal = float(np.mean([r["balanced_accuracy"] for r in seed_results]))
        std_bal = float(np.std([r["balanced_accuracy"] for r in seed_results]))
        mean_f1 = float(np.mean([r["macro_f1"] for r in seed_results]))
        std_f1 = float(np.std([r["macro_f1"] for r in seed_results]))

        all_results[f"L{L}"] = {
            "seeds": seed_results,
            "mean": {
                "accuracy": round(mean_acc, 4),
                "balanced_accuracy": round(mean_bal, 4),
                "macro_f1": round(mean_f1, 4),
            },
            "std": {
                "accuracy": round(std_acc, 4),
                "balanced_accuracy": round(std_bal, 4),
                "macro_f1": round(std_f1, 4),
            },
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "mean_balanced_accuracy": round(mean_bal, 4),
            "std_balanced_accuracy": round(std_bal, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
        }

        print(f"\n--- L={L} Aggregate (3 Seeds) ---")
        print(f"Accuracy:          {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Balanced Accuracy: {mean_bal:.4f} ± {std_bal:.4f}")
        print(f"Macro F1:          {mean_f1:.4f} ± {std_f1:.4f}")

    # Save JSON results
    json_path = OUT / "temporal_context_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON results to {json_path}")

    # Generate and save Markdown report
    report_md = generate_report(all_results)
    report_path = OUT / "temporal_context_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()