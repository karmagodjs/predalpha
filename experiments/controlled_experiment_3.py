"""
Controlled Experiment 3: Quote-Based Order Flow Imbalance (OFI).

Compares:
- Arm A: Reproduced Baseline with original 11 features.
- Arm B: Baseline + OFI (12 features: original 11 + causal quote-based OFI).
- Arm C (Ablation): OFI replacing static depth imbalance (11 features).

Evaluated across seeds 42, 123, and 999 on the validation split (N=1,428).
Strict train-only preprocessing fitting.
Locked test set is untouched.
"""

from __future__ import annotations

import copy
import hashlib
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
import pandas as pd
import torch

from pipeline_v2.models.metrics import evaluate_predictions
from pipeline_v2.models.recurrent_models import SmallLSTM, set_seed
from pipeline_v2.scaling.train_scaler import CausalSequenceScaler
from pipeline_v2.sequences.sequence_builder import SAFE_FEATURE_COLUMNS, SequenceBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("controlled_exp_3")

TRAIN_SPLIT_PARQUET = ROOT / "data/clean_v2/05_splits/expanded_collection/train.parquet"
VAL_SPLIT_PARQUET = ROOT / "data/clean_v2/05_splits/expanded_collection/validation.parquet"
TEST_LOCKED_NPZ = ROOT / "data/clean_v2/07_scaled/expanded_collection/test_scaled.npz"
PHASE19_CHECKPOINT = ROOT / "data/models/phase19/recurrent/best_lstm.pt"

OUT_DIR = ROOT / "data/models/experiment3_ofi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TO_ID = {"DOWN": 0, "FLAT": 1, "UP": 2}
CLASS_NAMES = ["DOWN", "FLAT", "UP"]
SEEDS = [42, 123, 999]


def verify_safety_invariants() -> None:
    """Verify that protected artifacts and immutable checkpoints remain untouched."""
    logger.info("Verifying safety invariants...")
    # 1. Phase 19 checkpoint hash
    if not PHASE19_CHECKPOINT.exists():
        raise FileNotFoundError(f"Phase 19 checkpoint missing: {PHASE19_CHECKPOINT}")
    chk_bytes = PHASE19_CHECKPOINT.read_bytes()
    chk_hash = hashlib.sha256(chk_bytes).hexdigest()
    expected_hash = "d8b75bdbaffbb7f2c4b398beea484f83d851b37380bccd93ca07199a3d9c025d"
    assert chk_hash == expected_hash, f"Phase 19 hash mismatch! {chk_hash} != {expected_hash}"

    # 2. Test dataset size guard (must not open or inspect contents)
    if not TEST_LOCKED_NPZ.exists():
        raise FileNotFoundError(f"Locked test dataset missing: {TEST_LOCKED_NPZ}")
    test_size = TEST_LOCKED_NPZ.stat().st_size
    assert test_size == 182401, f"Locked test dataset size mutated! {test_size} != 182401"
    logger.info("Safety invariants verified: Phase 19 SHA-256 and locked test set intact.")


def compute_causal_quote_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute causal quote-based Order Flow Imbalance (OFI) on order book streams.

    Definition (Cont, Kukanov, and Stoikov 2014):
    For quotes (bid, ask, bid_size, ask_size) at 1-second grid timestep t within (market_id, asset_id):
    - Bid Flow:
        db = qb_t               if b_t > b_{t-1}
        db = qb_t - qb_{t-1}    if b_t == b_{t-1}
        db = -qb_{t-1}          if b_t < b_{t-1}
    - Ask Flow:
        da = -qa_t              if a_t < a_{t-1}
        da = -(qa_t - qa_{t-1}) if a_t == a_{t-1}
        da = qa_{t-1}           if a_t > a_{t-1}
    - Total Event OFI:
        ofi_raw_t = db + da
    - Depth-Normalized OFI (dimensionless, scale-invariant across regimes):
        avg_depth_t = 0.5 * (qb_t + qa_t + qb_{t-1} + qa_{t-1})
        quote_ofi_norm_t = ofi_raw_t / (avg_depth_t + 1e-6)

    Event Ordering & Missing Quote Handling:
    - Streams partitioned strictly by (market_id, asset_id).
    - Rows sorted chronologically by grid_timestamp_ms.
    - If gap between timestamps > 1,000 ms, quote is discontinuous: ofi = NaN.
    - At start of stream (t=0): ofi = NaN.
    - All calculations depend ONLY on data observed at or before timestamp t.
    """
    stream_results: List[pd.DataFrame] = []

    for (mkt, ast), group in df.groupby(["market_id", "asset_id"], sort=False):
        g = group.sort_values("grid_timestamp_ms", ascending=True).copy()
        b = g["bid"].to_numpy(dtype=np.float64)
        a = g["ask"].to_numpy(dtype=np.float64)
        qb = g["bid_size"].to_numpy(dtype=np.float64)
        qa = g["ask_size"].to_numpy(dtype=np.float64)
        ts = g["grid_timestamp_ms"].to_numpy(dtype=np.int64)
        n = len(g)

        ofi_raw = np.full(n, np.nan, dtype=np.float64)
        ofi_norm = np.full(n, np.nan, dtype=np.float64)

        for i in range(1, n):
            delta_ms = ts[i] - ts[i - 1]
            if delta_ms != 1000:
                # Discontinuous gap; cannot assume consecutive 1s transition
                continue

            # Bid flow
            if b[i] > b[i - 1]:
                db = qb[i]
            elif b[i] == b[i - 1]:
                db = qb[i] - qb[i - 1]
            else:
                db = -qb[i - 1]

            # Ask flow
            if a[i] < a[i - 1]:
                da = -qa[i]
            elif a[i] == a[i - 1]:
                da = -(qa[i] - qa[i - 1])
            else:
                da = qa[i - 1]

            flow = db + da
            depth = 0.5 * (qb[i] + qa[i] + qb[i - 1] + qa[i - 1])
            ofi_raw[i] = flow
            ofi_norm[i] = flow / (depth + 1e-6)

        g["quote_ofi_raw"] = ofi_raw
        g["quote_ofi_norm"] = ofi_norm
        stream_results.append(g)

    return pd.concat(stream_results, axis=0).sort_index()


def run_statistical_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical feature ablation to establish whether OFI adds
    information beyond existing returns and depth imbalance.
    """
    logger.info("Computing statistical feature ablation on permitted training split...")
    lbl_map = {"DOWN": -1.0, "FLAT": 0.0, "UP": 1.0}

    # Clean subset of train
    cols = ["mid_return_1s", "mid_return_3s", "mid_return_5s", "depth_imbalance", "quote_ofi_norm", "label"]
    clean_tr = train_df[cols].dropna().copy()
    clean_tr["y"] = clean_tr["label"].map(lbl_map)

    # Clean subset of val
    clean_va = val_df[cols].dropna().copy()
    clean_va["y"] = clean_va["label"].map(lbl_map)

    # Correlations on training set
    ofi = clean_tr["quote_ofi_norm"].values
    r1 = clean_tr["mid_return_1s"].values
    r3 = clean_tr["mid_return_3s"].values
    r5 = clean_tr["mid_return_5s"].values
    di = clean_tr["depth_imbalance"].values
    y_tr = clean_tr["y"].values

    corr_r1 = float(np.corrcoef(ofi, r1)[0, 1])
    corr_r3 = float(np.corrcoef(ofi, r3)[0, 1])
    corr_r5 = float(np.corrcoef(ofi, r5)[0, 1])
    corr_di = float(np.corrcoef(ofi, di)[0, 1])
    corr_y = float(np.corrcoef(ofi, y_tr)[0, 1])

    # OLS Regression on Train:
    # Model 1: returns + depth imbalance
    # Model 2: returns + depth imbalance + OFI
    X1_tr = clean_tr[["mid_return_1s", "mid_return_3s", "mid_return_5s", "depth_imbalance"]].values
    X2_tr = clean_tr[["mid_return_1s", "mid_return_3s", "mid_return_5s", "depth_imbalance", "quote_ofi_norm"]].values

    X1_va = clean_va[["mid_return_1s", "mid_return_3s", "mid_return_5s", "depth_imbalance"]].values
    X2_va = clean_va[["mid_return_1s", "mid_return_3s", "mid_return_5s", "depth_imbalance", "quote_ofi_norm"]].values
    y_va = clean_va["y"].values

    # Standardize inputs using training statistics only
    mu1, sig1 = X1_tr.mean(0), X1_tr.std(0)
    mu2, sig2 = X2_tr.mean(0), X2_tr.std(0)

    X1_tr_z = (X1_tr - mu1) / sig1
    X2_tr_z = (X2_tr - mu2) / sig2
    X1_va_z = (X1_va - mu1) / sig1
    X2_va_z = (X2_va - mu2) / sig2

    # Add constant
    C1_tr = np.column_stack([np.ones(len(y_tr)), X1_tr_z])
    C2_tr = np.column_stack([np.ones(len(y_tr)), X2_tr_z])
    C1_va = np.column_stack([np.ones(len(y_va)), X1_va_z])
    C2_va = np.column_stack([np.ones(len(y_va)), X2_va_z])

    beta1, _, _, _ = np.linalg.lstsq(C1_tr, y_tr, rcond=None)
    beta2, _, _, _ = np.linalg.lstsq(C2_tr, y_tr, rcond=None)

    def calc_r2(y_true, y_hat):
        return float(1.0 - np.sum((y_true - y_hat) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    r2_tr_base = calc_r2(y_tr, C1_tr @ beta1)
    r2_tr_ofi = calc_r2(y_tr, C2_tr @ beta2)
    r2_va_base = calc_r2(y_va, C1_va @ beta1)
    r2_va_ofi = calc_r2(y_va, C2_va @ beta2)

    # Condition number check
    C_base_corr = np.corrcoef(X1_tr, rowvar=False)
    C_ofi_corr = np.corrcoef(X2_tr, rowvar=False)
    cond_base = float(np.linalg.cond(C_base_corr))
    cond_ofi = float(np.linalg.cond(C_ofi_corr))

    ablation_report = {
        "correlations": {
            "r(ofi, mid_return_1s)": round(corr_r1, 4),
            "r(ofi, mid_return_3s)": round(corr_r3, 4),
            "r(ofi, mid_return_5s)": round(corr_r5, 4),
            "r(ofi, depth_imbalance)": round(corr_di, 4),
            "r(ofi, target_direction)": round(corr_y, 4),
        },
        "ols_train_r2_base": round(r2_tr_base, 5),
        "ols_train_r2_ofi": round(r2_tr_ofi, 5),
        "ols_val_r2_base": round(r2_va_base, 5),
        "ols_val_r2_ofi": round(r2_va_ofi, 5),
        "ols_beta_base": [round(float(b), 4) for b in beta1],
        "ols_beta_ofi": [round(float(b), 4) for b in beta2],
        "ofi_beta_coefficient": round(float(beta2[-1]), 4),
        "condition_number_returns_and_depth": round(cond_base, 2),
        "condition_number_with_ofi": round(cond_ofi, 2),
    }

    logger.info(
        f"Statistical ablation: r(ofi, depth_imb)={corr_di:.4f}, r(ofi, r1s)={corr_r1:.4f}, "
        f"Beta OFI={beta2[-1]:.4f}, Train R2: {r2_tr_base:.5f} -> {r2_tr_ofi:.5f}"
    )
    return ablation_report


def build_and_scale_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CausalSequenceScaler]:
    """
    Build sequences and apply train-only CausalSequenceScaler.
    """
    builder = SequenceBuilder(
        sequence_length=10,
        feature_columns=feature_cols,
        timestamp_col="grid_timestamp_ms",
        label_col="label",
    )

    _, X_tr_raw, y_tr_raw, _ = builder.build_sequences_from_df(train_df, "train")
    _, X_va_raw, y_va_raw, _ = builder.build_sequences_from_df(val_df, "validation")

    y_train = np.array([LABEL_TO_ID[s] for s in y_tr_raw], dtype=np.int64)
    y_val = np.array([LABEL_TO_ID[s] for s in y_va_raw], dtype=np.int64)

    # Fit scaler strictly on training split
    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=feature_cols)
    scaler.fit(X_tr_raw, split_name="train")

    X_tr_scaled = scaler.transform(X_tr_raw)
    X_va_scaled = scaler.transform(X_va_raw)

    # Impute warm-up NaNs post-normalization
    X_train = np.where(np.isnan(X_tr_raw), 0.0, X_tr_scaled).astype(np.float32)
    X_val = np.where(np.isnan(X_va_raw), 0.0, X_va_scaled).astype(np.float32)

    # Strict invariant assertions
    assert not np.isnan(X_train).any(), "NaN in X_train!"
    assert not np.isnan(X_val).any(), "NaN in X_val!"
    assert not np.isinf(X_train).any(), "Inf in X_train!"
    assert not np.isinf(X_val).any(), "Inf in X_val!"
    assert len(X_train) == 7514, f"Train count mismatch: {len(X_train)}"
    assert len(X_val) == 1428, f"Val count mismatch: {len(X_val)}"

    return X_train, y_train, X_val, y_val, scaler


def run_single_model(
    arm_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Train SmallLSTM with AdamW and validate with identical patience/hyperparameters."""
    logger.info(f"Training {arm_name} (Seed {seed}, input_size={input_size})...")
    model = SmallLSTM(
        input_size=input_size,
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

    res = {
        "arm": arm_name,
        "seed": seed,
        "input_size": input_size,
        "parameter_count": model.parameter_count,
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
        f"{arm_name} Seed {seed}: Acc={res['accuracy']:.4f}, "
        f"BalAcc={res['balanced_accuracy']:.4f}, MacroF1={res['macro_f1']:.4f}, "
        f"Inversions={directional_inversions}, BestEpoch={model.best_epoch}"
    )
    return res


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across seeds."""
    accs = [r["accuracy"] for r in runs]
    bal_accs = [r["balanced_accuracy"] for r in runs]
    macro_f1s = [r["macro_f1"] for r in runs]
    weighted_f1s = [r["weighted_f1"] for r in runs]
    inversions = [r["directional_inversions"] for r in runs]
    best_epochs = [r["best_epoch"] for r in runs]
    val_losses = [r["best_val_loss"] for r in runs]

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
    t_start = time.time()
    logger.info("Starting PredAlpha Controlled Experiment 3: Quote-Based Order Flow Imbalance...")

    # 1. Safety verification
    verify_safety_invariants()

    # 2. Ingest splits and generate causal OFI feature
    logger.info("Ingesting train and validation splits...")
    raw_train_df = pd.read_parquet(TRAIN_SPLIT_PARQUET)
    raw_val_df = pd.read_parquet(VAL_SPLIT_PARQUET)

    logger.info("Computing causal quote-based OFI feature on splits...")
    train_df = compute_causal_quote_ofi(raw_train_df)
    val_df = compute_causal_quote_ofi(raw_val_df)

    # 3. Statistical feature ablation
    ablation_stats = run_statistical_ablation(train_df, val_df)

    # 4. Define feature sets
    cols_a = list(SAFE_FEATURE_COLUMNS)  # 11 features
    cols_b = list(SAFE_FEATURE_COLUMNS) + ["quote_ofi_norm"]  # 12 features
    cols_c = [c for c in SAFE_FEATURE_COLUMNS if c != "depth_imbalance"] + ["quote_ofi_norm"]  # 11 features (ablation)

    # Build and scale datasets
    logger.info("Building Arm A dataset (Original 11 features)...")
    X_tr_a, y_tr_a, X_va_a, y_va_a, scaler_a = build_and_scale_dataset(train_df, val_df, cols_a)

    logger.info("Building Arm B dataset (Baseline + OFI, 12 features)...")
    X_tr_b, y_tr_b, X_va_b, y_va_b, scaler_b = build_and_scale_dataset(train_df, val_df, cols_b)

    logger.info("Building Arm C dataset (Ablation: OFI replacing Depth Imbalance, 11 features)...")
    X_tr_c, y_tr_c, X_va_c, y_va_c, scaler_c = build_and_scale_dataset(train_df, val_df, cols_c)

    # 5. Execute Arm A (Baseline)
    logger.info("==========================================")
    logger.info("Executing Arm A: Original 11-Feature Baseline")
    logger.info("==========================================")
    runs_a = []
    for s in SEEDS:
        runs_a.append(run_single_model("Arm_A_Baseline11", X_tr_a, y_tr_a, X_va_a, y_va_a, input_size=11, seed=s))
    agg_a = aggregate_runs(runs_a)

    # 6. Execute Arm B (Baseline + OFI)
    logger.info("==========================================")
    logger.info("Executing Arm B: Baseline + OFI (12 Features)")
    logger.info("==========================================")
    runs_b = []
    for s in SEEDS:
        runs_b.append(run_single_model("Arm_B_BaselinePlusOFI", X_tr_b, y_tr_b, X_va_b, y_va_b, input_size=12, seed=s))
    agg_b = aggregate_runs(runs_b)

    # 7. Execute Arm C (Ablation: OFI without depth imbalance)
    logger.info("==========================================")
    logger.info("Executing Arm C: Ablation (OFI Replacing Depth Imbalance)")
    logger.info("==========================================")
    runs_c = []
    for s in SEEDS:
        runs_c.append(run_single_model("Arm_C_OFI_NoDepthImb", X_tr_c, y_tr_c, X_va_c, y_va_c, input_size=11, seed=s))
    agg_c = aggregate_runs(runs_c)

    # 8. Comparison deltas
    delta_b_vs_a = {
        "accuracy_diff": round(agg_b["accuracy_mean"] - agg_a["accuracy_mean"], 4),
        "balanced_accuracy_diff": round(agg_b["balanced_accuracy_mean"] - agg_a["balanced_accuracy_mean"], 4),
        "macro_f1_diff": round(agg_b["macro_f1_mean"] - agg_a["macro_f1_mean"], 4),
        "directional_inversions_diff": round(agg_b["directional_inversions_mean"] - agg_a["directional_inversions_mean"], 1),
        "val_loss_diff": round(agg_b["val_loss_mean"] - agg_a["val_loss_mean"], 5),
    }

    delta_c_vs_a = {
        "accuracy_diff": round(agg_c["accuracy_mean"] - agg_a["accuracy_mean"], 4),
        "balanced_accuracy_diff": round(agg_c["balanced_accuracy_mean"] - agg_a["balanced_accuracy_mean"], 4),
        "macro_f1_diff": round(agg_c["macro_f1_mean"] - agg_a["macro_f1_mean"], 4),
        "directional_inversions_diff": round(agg_c["directional_inversions_mean"] - agg_a["directional_inversions_mean"], 1),
        "val_loss_diff": round(agg_c["val_loss_mean"] - agg_a["val_loss_mean"], 5),
    }

    scaler_b_info = scaler_b.to_dict()

    full_results = {
        "experiment_name": "Controlled_Experiment_3_Order_Flow_Imbalance",
        "description": "Controlled comparison of baseline (11 features) vs baseline + quote-based OFI (12 features) and ablation (11 features)",
        "statistical_ablation": ablation_stats,
        "scaler_b_parameters": {
            "center": scaler_b_info["centers"],
            "scale": scaler_b_info["scales"],
        },
        "arm_a_baseline": agg_a,
        "arm_b_plus_ofi": agg_b,
        "arm_c_ablation": agg_c,
        "delta_b_vs_a": delta_b_vs_a,
        "delta_c_vs_a": delta_c_vs_a,
        "total_elapsed_sec": round(time.time() - t_start, 2),
    }

    out_file = OUT_DIR / "controlled_experiment_3_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"Controlled Experiment 3 completed successfully. Results saved to {out_file}")


if __name__ == "__main__":
    main()
