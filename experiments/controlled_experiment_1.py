"""
Controlled Experiment 1: Hyperparameter tuning and comparison runner.
Selects optimal hyperparameter on validation for Arm B and Arm C,
then runs full 3-seed evaluation (42, 123, 999) across all three arms.
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline_v2.models.metrics import evaluate_predictions
from pipeline_v2.models.recurrent_models import SmallLSTM, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("controlled_exp_1")

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/train_scaled.npz"
VAL_PATH = ROOT / "data/clean_v2/07_scaled/expanded_collection/validation_scaled.npz"
OUT_DIR = ROOT / "data/models/experiment1_accuracy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TO_ID = {"DOWN": 0, "FLAT": 1, "UP": 2}
CLASS_NAMES = ["DOWN", "FLAT", "UP"]
SEEDS = [42, 123, 999]


def load_datasets() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-scaled training and validation sequences without touching test set."""
    with np.load(TRAIN_PATH, allow_pickle=False) as tr:
        X_train = tr["X_imputed"].astype(np.float32)
        y_train_raw = tr["y"].astype(str)
        y_train = np.array([LABEL_TO_ID[s] for s in y_train_raw], dtype=np.int64)

    with np.load(VAL_PATH, allow_pickle=False) as va:
        X_val = va["X_imputed"].astype(np.float32)
        y_val_raw = va["y"].astype(str)
        y_val = np.array([LABEL_TO_ID[s] for s in y_val_raw], dtype=np.int64)

    return X_train, y_train, X_val, y_val


class InversionPenaltyLoss(nn.Module):
    """
    Cost-sensitive loss adding an explicit penalty for DOWN <-> UP directional inversions.
    When true label is DOWN (0), allocates penalty proportional to predicted UP prob (p2).
    When true label is UP (2), allocates penalty proportional to predicted DOWN prob (p0).
    """

    def __init__(self, inversion_penalty: float = 0.5, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.inversion_penalty = float(inversion_penalty)
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        if self.inversion_penalty <= 0.0:
            return ce_loss

        probs = torch.softmax(logits, dim=-1)
        down_mask = (targets == 0).float()
        up_mask = (targets == 2).float()
        inversion_penalty_term = (down_mask * probs[:, 2] + up_mask * probs[:, 0]).mean()
        return ce_loss + self.inversion_penalty * inversion_penalty_term


class InversionMarginLoss(nn.Module):
    """
    Cost-sensitive margin loss: introduces an additive logit penalty margin m
    on the opposing direction logit before softmax normalization.
    For DOWN (0): z2 is augmented by +m.
    For UP (2): z0 is augmented by +m.
    """

    def __init__(self, margin: float = 0.25, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.margin = float(margin)
        self.class_weight = class_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits shape: (B, 3)
        augmented_logits = logits.clone()
        down_mask = targets == 0
        up_mask = targets == 2

        if down_mask.any():
            augmented_logits[down_mask, 2] += self.margin
        if up_mask.any():
            augmented_logits[up_mask, 0] += self.margin

        # Compute CrossEntropy with augmented logits
        return nn.functional.cross_entropy(augmented_logits, targets, weight=self.class_weight)


class FlexibleSmallLSTM(SmallLSTM):
    """
    SmallLSTM trainer supporting custom cost-sensitive loss criteria while
    maintaining exact architecture and AdamW training budget.
    """

    def __init__(
        self,
        seed: int = 42,
        class_weight: Optional[np.ndarray] = None,
        custom_criterion: Optional[nn.Module] = None,
    ):
        super().__init__(
            input_size=11,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout_p=0.10,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            max_epochs=150,
            class_weight=class_weight,
            seed=seed,
        )
        self.custom_criterion = custom_criterion

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        patience: int = 25,
    ) -> FlexibleSmallLSTM:
        set_seed(self.seed)
        t_start = time.time()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.custom_criterion is not None:
            criterion = self.custom_criterion
        else:
            weight_t = (
                torch.tensor(self.class_weight, dtype=torch.float32)
                if self.class_weight is not None
                else None
            )
            criterion = nn.CrossEntropyLoss(weight=weight_t)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        patience_counter = 0
        self.history = []

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)
            epoch_info = {"epoch": epoch, "train_loss": round(train_loss, 5)}

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_logits = self.model(X_val_t)
                    v_loss = float(criterion(v_logits, y_val_t).item())
                    v_preds = v_logits.argmax(dim=-1).cpu().numpy()
                    v_eval = evaluate_predictions(y_val, v_preds, class_names=CLASS_NAMES)

                epoch_info["val_loss"] = round(v_loss, 5)
                epoch_info["val_accuracy"] = v_eval["accuracy"]
                epoch_info["val_balanced_accuracy"] = v_eval["balanced_accuracy"]
                epoch_info["val_macro_f1"] = v_eval["macro_f1"]

                if v_loss < self.best_val_loss - 1e-4:
                    self.best_val_loss = v_loss
                    self.best_epoch = epoch
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.history.append(epoch_info)
                    break
            else:
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch

            self.history.append(epoch_info)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.train_duration_sec = round(time.time() - t_start, 4)
        return self


def compute_inversion_stats(cm: List[List[int]]) -> Dict[str, Any]:
    """Compute inversion counts and rates from a 3x3 confusion matrix."""
    # cm[row][col] -> row=actual, col=pred
    down_to_up = int(cm[0][2])
    up_to_down = int(cm[2][0])
    total_inversions = down_to_up + up_to_down

    total_down = int(sum(cm[0]))
    total_up = int(sum(cm[2]))
    total_directional = total_down + total_up

    inversion_rate = float(total_inversions / total_directional) if total_directional > 0 else 0.0

    return {
        "down_to_up": down_to_up,
        "up_to_down": up_to_down,
        "total_inversions": total_inversions,
        "total_directional": total_directional,
        "inversion_rate": round(inversion_rate, 4),
    }


def evaluate_model_seed(
    model_factory: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    """Train and evaluate on validation for one deterministic seed."""
    model = model_factory(seed)
    model.fit(X_train, y_train, X_val, y_val, patience=25)

    preds = model.predict(X_val)
    metrics = evaluate_predictions(y_val, preds, class_names=CLASS_NAMES)
    inversion_stats = compute_inversion_stats(metrics["confusion_matrix"])

    return {
        "seed": seed,
        "best_epoch": model.best_epoch,
        "best_val_loss": round(model.best_val_loss, 5),
        "train_duration_sec": model.train_duration_sec,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "inversions": inversion_stats,
    }


def aggregate_seed_results(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std across random seeds."""
    accuracies = [r["accuracy"] for r in seed_results]
    bal_accs = [r["balanced_accuracy"] for r in seed_results]
    macro_f1s = [r["macro_f1"] for r in seed_results]
    weighted_f1s = [r["weighted_f1"] for r in seed_results]
    inv_rates = [r["inversions"]["inversion_rate"] for r in seed_results]
    total_invs = [r["inversions"]["total_inversions"] for r in seed_results]

    agg = {
        "accuracy_mean": round(float(np.mean(accuracies)), 4),
        "accuracy_std": round(float(np.std(accuracies)), 4),
        "balanced_accuracy_mean": round(float(np.mean(bal_accs)), 4),
        "balanced_accuracy_std": round(float(np.std(bal_accs)), 4),
        "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
        "macro_f1_std": round(float(np.std(macro_f1s)), 4),
        "weighted_f1_mean": round(float(np.mean(weighted_f1s)), 4),
        "weighted_f1_std": round(float(np.std(weighted_f1s)), 4),
        "inversion_rate_mean": round(float(np.mean(inv_rates)), 4),
        "inversion_count_mean": round(float(np.mean(total_invs)), 1),
    }

    # Per-class means
    per_class_agg = {}
    for c in CLASS_NAMES:
        precs = [r["per_class"][c]["precision"] for r in seed_results]
        recs = [r["per_class"][c]["recall"] for r in seed_results]
        f1s = [r["per_class"][c]["f1"] for r in seed_results]
        per_class_agg[c] = {
            "precision_mean": round(float(np.mean(precs)), 4),
            "precision_std": round(float(np.std(precs)), 4),
            "recall_mean": round(float(np.mean(recs)), 4),
            "recall_std": round(float(np.std(recs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
        }
    agg["per_class"] = per_class_agg
    return agg


def main():
    logger.info("=" * 70)
    logger.info("PREDALPHA ACCURACY IMPROVEMENT — CONTROLLED EXPERIMENT 1")
    logger.info("=" * 70)

    X_train, y_train, X_val, y_val = load_datasets()
    logger.info(f"Loaded Train: {X_train.shape}, Validation: {X_val.shape}")

    # =========================================================================
    # STEP 1: HYPERPARAMETER TUNING ON VALIDATION (SEED 42)
    # =========================================================================
    logger.info("\n--- Tuning Arm B: Class Weight Candidates on Validation (Seed 42) ---")
    weight_candidates = {
        "flat_1.25": np.array([1.0, 1.25, 1.0], dtype=np.float32),
        "flat_1.50": np.array([1.0, 1.50, 1.0], dtype=np.float32),
        "flat_1.75": np.array([1.0, 1.75, 1.0], dtype=np.float32),
        "flat_2.00": np.array([1.0, 2.00, 1.0], dtype=np.float32),
        "balanced": np.array([0.8212, 1.7713, 0.8212], dtype=np.float32),
    }

    best_arm_b_name = "flat_1.50"
    best_arm_b_f1 = -1.0
    arm_b_tuning_records = {}

    for name, w in weight_candidates.items():
        res = evaluate_model_seed(
            lambda s: FlexibleSmallLSTM(seed=s, class_weight=w),
            X_train, y_train, X_val, y_val, seed=42,
        )
        arm_b_tuning_records[name] = {
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "macro_f1": res["macro_f1"],
            "flat_recall": res["per_class"]["FLAT"]["recall"],
        }
        logger.info(
            f"Weight '{name}': Acc={res['accuracy']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, "
            f"MacroF1={res['macro_f1']:.4f}, FLAT_Recall={res['per_class']['FLAT']['recall']:.4f}"
        )
        if res["macro_f1"] > best_arm_b_f1:
            best_arm_b_f1 = res["macro_f1"]
            best_arm_b_name = name

    logger.info(f"--> Selected Best Arm B Configuration: {best_arm_b_name} (MacroF1={best_arm_b_f1:.4f})")
    best_class_weight = weight_candidates[best_arm_b_name]

    # --- Tuning Arm C: Cost-Sensitive Penalty Candidates on Validation (Seed 42) ---
    logger.info("\n--- Tuning Arm C: Cost-Sensitive Candidates on Validation (Seed 42) ---")
    penalty_candidates = {
        "penalty_0.25": InversionPenaltyLoss(inversion_penalty=0.25),
        "penalty_0.50": InversionPenaltyLoss(inversion_penalty=0.50),
        "penalty_0.75": InversionPenaltyLoss(inversion_penalty=0.75),
        "penalty_1.00": InversionPenaltyLoss(inversion_penalty=1.00),
        "margin_0.25": InversionMarginLoss(margin=0.25),
        "margin_0.50": InversionMarginLoss(margin=0.50),
    }

    best_arm_c_name = "penalty_0.50"
    best_arm_c_score = -1.0
    arm_c_tuning_records = {}

    for name, loss_fn in penalty_candidates.items():
        res = evaluate_model_seed(
            lambda s: FlexibleSmallLSTM(seed=s, custom_criterion=loss_fn),
            X_train, y_train, X_val, y_val, seed=42,
        )
        # Score prioritizing inversion reduction while maintaining macro F1:
        # Score = MacroF1 - 0.5 * inversion_rate
        score = res["macro_f1"] - 0.2 * res["inversions"]["inversion_rate"]
        arm_c_tuning_records[name] = {
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "macro_f1": res["macro_f1"],
            "inversion_rate": res["inversions"]["inversion_rate"],
            "total_inversions": res["inversions"]["total_inversions"],
        }
        logger.info(
            f"Candidate '{name}': Acc={res['accuracy']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, "
            f"MacroF1={res['macro_f1']:.4f}, Inversions={res['inversions']['total_inversions']} ({res['inversions']['inversion_rate']*100:.2f}%)"
        )
        if score > best_arm_c_score:
            best_arm_c_score = score
            best_arm_c_name = name

    logger.info(f"--> Selected Best Arm C Configuration: {best_arm_c_name}")
    best_arm_c_criterion = penalty_candidates[best_arm_c_name]

    # =========================================================================
    # STEP 2: FULL 3-SEED EVALUATION ACROSS ALL THREE ARMS
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING 3-SEED CONTROLLED COMPARISON (Seeds: 42, 123, 999)")
    logger.info("=" * 70)

    # Arm A: Existing Unweighted Baseline
    logger.info("\n[Arm A] Running Existing Unweighted Baseline...")
    arm_a_seeds = []
    for s in SEEDS:
        res = evaluate_model_seed(
            lambda seed: FlexibleSmallLSTM(seed=seed, class_weight=None),
            X_train, y_train, X_val, y_val, seed=s,
        )
        arm_a_seeds.append(res)
        logger.info(f"  Seed {s}: Acc={res['accuracy']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, MacroF1={res['macro_f1']:.4f}, Inversions={res['inversions']['total_inversions']}")
    arm_a_agg = aggregate_seed_results(arm_a_seeds)

    # Arm B: Correctly Wired Class-Weighted Loss
    logger.info(f"\n[Arm B] Running Class-Weighted Loss ({best_arm_b_name})...")
    arm_b_seeds = []
    for s in SEEDS:
        res = evaluate_model_seed(
            lambda seed: FlexibleSmallLSTM(seed=seed, class_weight=best_class_weight),
            X_train, y_train, X_val, y_val, seed=s,
        )
        arm_b_seeds.append(res)
        logger.info(f"  Seed {s}: Acc={res['accuracy']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, MacroF1={res['macro_f1']:.4f}, Inversions={res['inversions']['total_inversions']}")
    arm_b_agg = aggregate_seed_results(arm_b_seeds)

    # Arm C: Cost-Sensitive Loss
    logger.info(f"\n[Arm C] Running Cost-Sensitive Inversion Loss ({best_arm_c_name})...")
    arm_c_seeds = []
    for s in SEEDS:
        # Clone criterion for each seed to ensure fresh state
        loss_instance = copy.deepcopy(best_arm_c_criterion)
        res = evaluate_model_seed(
            lambda seed: FlexibleSmallLSTM(seed=seed, custom_criterion=loss_instance),
            X_train, y_train, X_val, y_val, seed=s,
        )
        arm_c_seeds.append(res)
        logger.info(f"  Seed {s}: Acc={res['accuracy']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, MacroF1={res['macro_f1']:.4f}, Inversions={res['inversions']['total_inversions']}")
    arm_c_agg = aggregate_seed_results(arm_c_seeds)

    # =========================================================================
    # STEP 3: ASSEMBLE REPORT & ARTIFACTS
    # =========================================================================
    final_output = {
        "metadata": {
            "experiment_name": "Controlled Experiment 1: Loss Formulations",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_train": str(TRAIN_PATH),
            "dataset_validation": str(VAL_PATH),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "seeds": SEEDS,
            "architecture": "SmallLSTM(input=11, hidden=32, num_layers=1, num_classes=3, dropout=0.10)",
            "selected_arm_b_config": best_arm_b_name,
            "selected_arm_b_weights": best_class_weight.tolist(),
            "selected_arm_c_config": best_arm_c_name,
        },
        "tuning_exploration": {
            "arm_b_weights": arm_b_tuning_records,
            "arm_c_penalties": arm_c_tuning_records,
        },
        "results": {
            "arm_a_unweighted": {
                "aggregate": arm_a_agg,
                "seeds": arm_a_seeds,
            },
            "arm_b_class_weighted": {
                "aggregate": arm_b_agg,
                "seeds": arm_b_seeds,
            },
            "arm_c_cost_sensitive": {
                "aggregate": arm_c_agg,
                "seeds": arm_c_seeds,
            },
        },
    }

    out_file = OUT_DIR / "controlled_experiment_1_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info(f"EXPERIMENT 1 COMPLETE — Artifacts saved to: {out_file}")
    logger.info("=" * 70)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("CONTROLLED EXPERIMENT 1 SUMMARY (Validation Mean across Seeds 42, 123, 999)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Arm A (Unweighted)':<16} | {'Arm B (Class-Wtd)':<16} | {'Arm C (Cost-Sens)':<16}")
    print("-" * 80)
    print(f"{'Accuracy':<28} | {arm_a_agg['accuracy_mean']:.4f} ± {arm_a_agg['accuracy_std']:.4f}  | {arm_b_agg['accuracy_mean']:.4f} ± {arm_b_agg['accuracy_std']:.4f}  | {arm_c_agg['accuracy_mean']:.4f} ± {arm_c_agg['accuracy_std']:.4f}")
    print(f"{'Balanced Accuracy':<28} | {arm_a_agg['balanced_accuracy_mean']:.4f} ± {arm_a_agg['balanced_accuracy_std']:.4f}  | {arm_b_agg['balanced_accuracy_mean']:.4f} ± {arm_b_agg['balanced_accuracy_std']:.4f}  | {arm_c_agg['balanced_accuracy_mean']:.4f} ± {arm_c_agg['balanced_accuracy_std']:.4f}")
    print(f"{'Macro F1':<28} | {arm_a_agg['macro_f1_mean']:.4f} ± {arm_a_agg['macro_f1_std']:.4f}  | {arm_b_agg['macro_f1_mean']:.4f} ± {arm_b_agg['macro_f1_std']:.4f}  | {arm_c_agg['macro_f1_mean']:.4f} ± {arm_c_agg['macro_f1_std']:.4f}")
    print(f"{'Weighted F1':<28} | {arm_a_agg['weighted_f1_mean']:.4f} ± {arm_a_agg['weighted_f1_std']:.4f}  | {arm_b_agg['weighted_f1_mean']:.4f} ± {arm_b_agg['weighted_f1_std']:.4f}  | {arm_c_agg['weighted_f1_mean']:.4f} ± {arm_c_agg['weighted_f1_std']:.4f}")
    print(f"{'Inversion Rate':<28} | {arm_a_agg['inversion_rate_mean']*100:.2f}%           | {arm_b_agg['inversion_rate_mean']*100:.2f}%           | {arm_c_agg['inversion_rate_mean']*100:.2f}%")
    print(f"{'Inversion Count (mean)':<28} | {arm_a_agg['inversion_count_mean']:.1f}            | {arm_b_agg['inversion_count_mean']:.1f}            | {arm_c_agg['inversion_count_mean']:.1f}")
    print("-" * 80)
    for c in CLASS_NAMES:
        f1_a = arm_a_agg['per_class'][c]['f1_mean']
        f1_b = arm_b_agg['per_class'][c]['f1_mean']
        f1_c = arm_c_agg['per_class'][c]['f1_mean']
        print(f"F1 {c:<25} | {f1_a:.4f}           | {f1_b:.4f}           | {f1_c:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
