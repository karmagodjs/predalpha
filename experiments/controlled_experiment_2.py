"""
Controlled Experiment 2: Feature Engineering (Microprice Collinearity Fix).

Compares:
- Arm A: Reproduced baseline with original 11 features.
- Arm B: Causal feature replacement (microprice replaced by microprice - mid_price).

Evaluated across seeds 42, 123, and 999 on validation set (N=1,428).
Strict train-only preprocessing fitting.
Locked test set is untouched.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from pipeline_v2.models.metrics import evaluate_predictions
from pipeline_v2.models.recurrent_models import SmallLSTM, set_seed
from pipeline_v2.scaling.train_scaler import CausalSequenceScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("controlled_exp_2")

TRAIN_RAW_PATH = ROOT / "data/clean_v2/06_sequences/expanded_collection/train_sequences.npz"
VAL_RAW_PATH = ROOT / "data/clean_v2/06_sequences/expanded_collection/validation_sequences.npz"
TRAIN_SCALED_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/train_scaled.npz"
VAL_SCALED_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/validation_scaled.npz"

OUT_DIR = ROOT / "data/models/experiment2_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TO_ID = {"DOWN": 0, "FLAT": 1, "UP": 2}
CLASS_NAMES = ["DOWN", "FLAT", "UP"]
SEEDS = [42, 123, 999]


def compute_collinearity_diagnostics() -> Dict[str, Any]:
    """Verify collinearity between mid_price and microprice strictly on permitted training data."""
    logger.info("Computing collinearity diagnostics on training sequences...")
    with np.load(TRAIN_RAW_PATH, allow_pickle=False) as tr:
        X_train = tr["X"]
        feat_names = [str(s) for s in tr["feature_names"]]

    mid_idx = feat_names.index("mid_price")
    micro_idx = feat_names.index("microprice")
    spread_idx = feat_names.index("spread")
    depth_idx = feat_names.index("depth_imbalance")

    # Flatten (N, L)
    mid_flat = X_train[:, :, mid_idx].ravel()
    micro_flat = X_train[:, :, micro_idx].ravel()
    spread_flat = X_train[:, :, spread_idx].ravel()
    depth_flat = X_train[:, :, depth_idx].ravel()

    valid_mask = ~np.isnan(mid_flat) & ~np.isnan(micro_flat)
    mid_v = mid_flat[valid_mask]
    micro_v = micro_flat[valid_mask]
    spread_v = spread_flat[valid_mask]
    depth_v = depth_flat[valid_mask]

    diff_v = micro_v - mid_v
    pearson_r = float(np.corrcoef(mid_v, micro_v)[0, 1])
    r2 = pearson_r ** 2
    mean_abs_diff = float(np.mean(np.abs(diff_v)))
    max_abs_diff = float(np.max(np.abs(diff_v)))

    # Step 9 endpoint correlation
    mid_ep = X_train[:, 9, mid_idx]
    micro_ep = X_train[:, 9, micro_idx]
    pearson_r_ep = float(np.corrcoef(mid_ep, micro_ep)[0, 1])

    # Condition number of correlation matrices
    X_flat = X_train.reshape(-1, 11)
    row_valid = ~np.isnan(X_flat).any(axis=1)
    X_clean = X_flat[row_valid]

    C_orig = np.corrcoef(X_clean, rowvar=False)
    cond_orig = float(np.linalg.cond(C_orig))

    X_mod = X_clean.copy()
    X_mod[:, micro_idx] = X_clean[:, micro_idx] - X_clean[:, mid_idx]
    C_mod = np.corrcoef(X_mod, rowvar=False)
    cond_mod = float(np.linalg.cond(C_mod))

    r_delta_mid = float(np.corrcoef(diff_v, mid_v)[0, 1])
    r_delta_depth = float(np.corrcoef(diff_v, depth_v)[0, 1])
    analytic_identity = 0.5 * spread_v * depth_v
    r_identity = float(np.corrcoef(diff_v, analytic_identity)[0, 1])

    diagnostics = {
        "training_sample_count": int(len(mid_v)),
        "pearson_r_mid_micro": round(pearson_r, 7),
        "r2_mid_micro": round(r2, 7),
        "pearson_r_endpoint_step9": round(pearson_r_ep, 7),
        "mean_abs_difference": round(mean_abs_diff, 6),
        "max_abs_difference": round(max_abs_diff, 6),
        "condition_number_original_11": round(cond_orig, 2),
        "condition_number_modified_11": round(cond_mod, 2),
        "condition_number_reduction_factor": round(cond_orig / cond_mod, 1),
        "pearson_r_delta_micro_mid": round(r_delta_mid, 6),
        "pearson_r_delta_micro_depth_imbalance": round(r_delta_depth, 6),
        "pearson_r_delta_micro_analytic_formula": round(r_identity, 6),
    }

    logger.info(
        f"Collinearity diagnostics: r={pearson_r:.6f}, R2={r2:.6f}, "
        f"CondOrig={cond_orig:.1f} -> CondMod={cond_mod:.1f} ({cond_orig/cond_mod:.1f}x reduction)"
    )
    return diagnostics


def prepare_datasets() -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CausalSequenceScaler],
]:
    """
    Prepare datasets for Arm A and Arm B strictly without looking at test data.

    Returns:
        (arm_a_datasets, arm_b_datasets)
        where arm_a_datasets = (X_train_a, y_train, X_val_a, y_val)
        and arm_b_datasets = (X_train_b, y_train, X_val_b, y_val, scaler_b)
    """
    # 1. Arm A: original scaled features
    with np.load(TRAIN_SCALED_PATH, allow_pickle=False) as tr:
        X_train_a = tr["X_imputed"].astype(np.float32)
        y_train_raw = tr["y"].astype(str)
        y_train = np.array([LABEL_TO_ID[s] for s in y_train_raw], dtype=np.int64)

    with np.load(VAL_SCALED_PATH, allow_pickle=False) as va:
        X_val_a = va["X_imputed"].astype(np.float32)
        y_val_raw = va["y"].astype(str)
        y_val = np.array([LABEL_TO_ID[s] for s in y_val_raw], dtype=np.int64)

    # 2. Arm B: raw sequences transformed with feature 9 = microprice - mid_price
    with np.load(TRAIN_RAW_PATH, allow_pickle=False) as tr_raw:
        X_train_b_raw = tr_raw["X"].astype(np.float64).copy()
        feat_names = [str(s) for s in tr_raw["feature_names"]]

    with np.load(VAL_RAW_PATH, allow_pickle=False) as va_raw:
        X_val_b_raw = va_raw["X"].astype(np.float64).copy()

    mid_idx = feat_names.index("mid_price")
    micro_idx = feat_names.index("microprice")
    assert mid_idx == 0 and micro_idx == 9, f"Unexpected feature indices: mid={mid_idx}, micro={micro_idx}"

    # Replace feature 9 with microprice - mid_price
    delta_micro_train = X_train_b_raw[:, :, micro_idx] - X_train_b_raw[:, :, mid_idx]
    delta_micro_val = X_val_b_raw[:, :, micro_idx] - X_val_b_raw[:, :, mid_idx]

    X_train_b_raw[:, :, micro_idx] = delta_micro_train
    X_val_b_raw[:, :, micro_idx] = delta_micro_val

    # Fit CausalSequenceScaler strictly on training partition
    mod_feature_names = list(feat_names)
    mod_feature_names[micro_idx] = "microprice_minus_mid"
    scaler_b = CausalSequenceScaler(scaler_type="standard", feature_names=mod_feature_names)
    scaler_b.fit(X_train_b_raw, split_name="train")

    # Transform train and val with train-learned parameters
    X_train_b_scaled = scaler_b.transform(X_train_b_raw)
    X_val_b_scaled = scaler_b.transform(X_val_b_raw)

    # Detect warmup NaNs and apply zero imputation post-normalization
    tr_nan_mask = np.isnan(X_train_b_raw)
    val_nan_mask = np.isnan(X_val_b_raw)

    X_train_b = np.where(tr_nan_mask, 0.0, X_train_b_scaled).astype(np.float32)
    X_val_b = np.where(val_nan_mask, 0.0, X_val_b_scaled).astype(np.float32)

    # Strict invariant assertions
    assert not np.isnan(X_train_b).any(), "NaN detected in Arm B X_train!"
    assert not np.isnan(X_val_b).any(), "NaN detected in Arm B X_val!"
    assert not np.isinf(X_train_b).any(), "Inf detected in Arm B X_train!"
    assert not np.isinf(X_val_b).any(), "Inf detected in Arm B X_val!"
    assert X_train_b.shape == (7514, 10, 11), f"Arm B train shape mismatch: {X_train_b.shape}"
    assert X_val_b.shape == (1428, 10, 11), f"Arm B val shape mismatch: {X_val_b.shape}"

    logger.info(
        f"Arm B Scaler fitted on Train only: Feature 9 ({mod_feature_names[micro_idx]}) "
        f"center={scaler_b.center_[micro_idx]:.6f}, scale={scaler_b.scale_[micro_idx]:.6f}"
    )

    arm_a = (X_train_a, y_train, X_val_a, y_val)
    arm_b = (X_train_b, y_train, X_val_b, y_val, scaler_b)
    return arm_a, arm_b


def run_single_model(
    arm_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    """Train and evaluate SmallLSTM on validation set with identical configuration."""
    logger.info(f"Training {arm_name} (Seed {seed})...")
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
        class_weight=None,
        seed=seed,
    )

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, patience=25)
    val_preds = model.predict(X_val)
    eval_dict = evaluate_predictions(y_val, val_preds, class_names=CLASS_NAMES)

    cm = eval_dict["confusion_matrix"]
    directional_inversions = int(cm[0][2] + cm[2][0])

    run_result = {
        "arm": arm_name,
        "seed": seed,
        "best_epoch": int(model.best_epoch),
        "best_val_loss": round(float(model.best_val_loss), 5),
        "train_duration_sec": round(float(model.train_duration_sec), 2),
        "accuracy": eval_dict["accuracy"],
        "balanced_accuracy": eval_dict["balanced_accuracy"],
        "macro_f1": eval_dict["macro_f1"],
        "weighted_f1": eval_dict["weighted_f1"],
        "per_class": eval_dict["per_class"],
        "confusion_matrix": cm,
        "directional_inversions": directional_inversions,
    }

    logger.info(
        f"{arm_name} Seed {seed}: Acc={run_result['accuracy']:.4f}, "
        f"BalAcc={run_result['balanced_accuracy']:.4f}, "
        f"MacroF1={run_result['macro_f1']:.4f}, Inversions={directional_inversions}, "
        f"BestEpoch={model.best_epoch}"
    )
    return run_result


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and standard deviation across seeds."""
    accs = [r["accuracy"] for r in runs]
    bal_accs = [r["balanced_accuracy"] for r in runs]
    macro_f1s = [r["macro_f1"] for r in runs]
    weighted_f1s = [r["weighted_f1"] for r in runs]
    inversions = [r["directional_inversions"] for r in runs]
    best_epochs = [r["best_epoch"] for r in runs]
    val_losses = [r["best_val_loss"] for r in runs]

    # Per-class aggregates
    per_class_agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for c in CLASS_NAMES:
        per_class_agg[c] = {
            "precision": {
                "mean": round(float(np.mean([r["per_class"][c]["precision"] for r in runs])), 4),
                "std": round(float(np.std([r["per_class"][c]["precision"] for r in runs])), 4),
            },
            "recall": {
                "mean": round(float(np.mean([r["per_class"][c]["recall"] for r in runs])), 4),
                "std": round(float(np.std([r["per_class"][c]["recall"] for r in runs])), 4),
            },
            "f1": {
                "mean": round(float(np.mean([r["per_class"][c]["f1"] for r in runs])), 4),
                "std": round(float(np.std([r["per_class"][c]["f1"] for r in runs])), 4),
            },
            "support": runs[0]["per_class"][c]["support"],
        }

    return {
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "balanced_accuracy_mean": round(float(np.mean(bal_accs)), 4),
        "balanced_accuracy_std": round(float(np.std(bal_accs)), 4),
        "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
        "macro_f1_std": round(float(np.std(macro_f1s)), 4),
        "weighted_f1_mean": round(float(np.mean(weighted_f1s)), 4),
        "weighted_f1_std": round(float(np.std(weighted_f1s)), 4),
        "directional_inversions_mean": round(float(np.mean(inversions)), 1),
        "directional_inversions_std": round(float(np.std(inversions)), 2),
        "best_epoch_mean": round(float(np.mean(best_epochs)), 1),
        "val_loss_mean": round(float(np.mean(val_losses)), 5),
        "per_class": per_class_agg,
        "runs": runs,
    }


def main() -> None:
    t_start_total = time.time()
    logger.info("Starting PredAlpha Controlled Experiment 2: Feature Engineering...")

    # 1. Collinearity Diagnostics on Permitted Training Data
    diagnostics = compute_collinearity_diagnostics()

    # 2. Data Preparation
    arm_a_data, arm_b_data = prepare_datasets()
    X_train_a, y_train, X_val_a, y_val = arm_a_data
    X_train_b, _, X_val_b, _, scaler_b = arm_b_data

    # 3. Execution of Arm A across seeds
    logger.info("==========================================")
    logger.info("Executing Arm A: Original 11 Features")
    logger.info("==========================================")
    runs_a = []
    for s in SEEDS:
        res = run_single_model("Arm_A_Original11", X_train_a, y_train, X_val_a, y_val, seed=s)
        runs_a.append(res)
    agg_a = aggregate_runs(runs_a)

    # 4. Execution of Arm B across seeds
    logger.info("==========================================")
    logger.info("Executing Arm B: Microprice Collinearity Replacement")
    logger.info("==========================================")
    runs_b = []
    for s in SEEDS:
        res = run_single_model("Arm_B_MicropriceMinusMid", X_train_b, y_train, X_val_b, y_val, seed=s)
        runs_b.append(res)
    agg_b = aggregate_runs(runs_b)

    # 5. Delta Comparison
    comparison = {
        "accuracy_diff": round(agg_b["accuracy_mean"] - agg_a["accuracy_mean"], 4),
        "balanced_accuracy_diff": round(agg_b["balanced_accuracy_mean"] - agg_a["balanced_accuracy_mean"], 4),
        "macro_f1_diff": round(agg_b["macro_f1_mean"] - agg_a["macro_f1_mean"], 4),
        "directional_inversions_diff": round(agg_b["directional_inversions_mean"] - agg_a["directional_inversions_mean"], 1),
        "val_loss_diff": round(agg_b["val_loss_mean"] - agg_a["val_loss_mean"], 5),
    }

    # Scaler B metadata
    scaler_metadata = scaler_b.to_dict()

    full_results = {
        "experiment_name": "Controlled_Experiment_2_Feature_Engineering",
        "description": "Comparison of original 11 features (Arm A) vs causal microprice replacement with microprice - mid_price (Arm B)",
        "collinearity_diagnostics": diagnostics,
        "scaler_b_parameters": {
            "center": scaler_metadata["centers"],
            "scale": scaler_metadata["scales"],
        },
        "arm_a_summary": agg_a,
        "arm_b_summary": agg_b,
        "delta_comparison": comparison,
        "total_elapsed_sec": round(time.time() - t_start_total, 2),
    }

    out_file = OUT_DIR / "controlled_experiment_2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"Results successfully saved to {out_file}")
    logger.info("Controlled Experiment 2 complete.")


if __name__ == "__main__":
    main()
