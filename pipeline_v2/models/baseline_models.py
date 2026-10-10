"""
Baseline Models for Phase 19A Signal Validation.

Contains:
1. MajorityBaseline: Empirical class-mode benchmark (the absolute performance floor).
2. LogisticRegressionModel: Linear multinomial softmax classifier (Linear(in_features -> 3)).
3. SmallMLP: 2-hidden-layer feedforward network matching Phase 19 specification:
   Linear(110 -> 64) -> ReLU -> Dropout(0.10) -> Linear(64 -> 32) -> ReLU -> Dropout(0.10) -> Linear(32 -> 3).

All models are deterministic given a random seed, enforce strict causality,
and support early stopping based on validation metrics.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pipeline_v2.models.metrics import evaluate_predictions


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across python, numpy, and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MajorityBaseline:
    """
    Model 0: Predicts the most frequent class in the training partition for all samples.
    Represents the statistical performance floor.
    """

    def __init__(self, num_classes: int = 3, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["DOWN", "FLAT", "UP"]
        self.majority_class: Optional[int] = None
        self.majority_class_name: Optional[str] = None
        self.class_counts: Dict[int, int] = {}
        self.class_priors: np.ndarray = np.zeros(num_classes, dtype=np.float64)
        self.n_samples_seen: int = 0

    def fit(self, y: np.ndarray) -> MajorityBaseline:
        """Fit empirical class frequencies from training labels."""
        y_arr = np.asarray(y, dtype=np.int64)
        self.n_samples_seen = len(y_arr)
        counts = np.bincount(y_arr, minlength=self.num_classes)
        self.class_counts = {int(i): int(counts[i]) for i in range(self.num_classes)}
        self.class_priors = counts / float(len(y_arr))

        # In case of tie, pick the highest class index deterministically
        max_count = np.max(counts)
        tied_candidates = np.where(counts == max_count)[0]
        self.majority_class = int(tied_candidates[-1])
        self.majority_class_name = self.class_names[self.majority_class]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the majority class for all input rows."""
        if self.majority_class is None:
            raise RuntimeError("MajorityBaseline must be fitted before predict.")
        n = len(X)
        return np.full(n, self.majority_class, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return empirical training class priors for all input rows."""
        n = len(X)
        return np.tile(self.class_priors, (n, 1))

    def get_summary(self) -> Dict[str, Any]:
        """Return metadata summary."""
        return {
            "model_type": "MajorityBaseline",
            "majority_class": self.majority_class,
            "majority_class_name": self.majority_class_name,
            "class_counts": self.class_counts,
            "class_priors": self.class_priors.tolist(),
            "parameter_count": 0,
        }


class LogisticRegressionModule(nn.Module):
    """Single linear layer for multinomial softmax regression."""

    def __init__(self, in_features: int = 110, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LogisticRegressionModel:
    """
    Model 1: Multinomial Logistic Regression trained via AdamW on CrossEntropyLoss.
    """

    def __init__(
        self,
        in_features: int = 110,
        num_classes: int = 3,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        class_weight: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.in_features = in_features
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.class_weight = class_weight
        self.seed = seed

        set_seed(seed)
        self.model = LogisticRegressionModule(in_features=in_features, num_classes=num_classes)
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.train_duration_sec: float = 0.0
        self.inference_duration_sec: float = 0.0

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        patience: int = 25,
    ) -> LogisticRegressionModel:
        """Fit model strictly on training data with optional early stopping on validation."""
        set_seed(self.seed)
        t_start = time.time()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)

        weight_t = None
        if self.class_weight is not None:
            weight_t = torch.tensor(self.class_weight, dtype=torch.float32)

        criterion = nn.CrossEntropyLoss(weight=weight_t)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        self.history = []
        self.model.train()

        for epoch in range(1, self.max_epochs + 1):
            optimizer.zero_grad()
            logits = self.model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            train_loss = float(loss.item())

            # Evaluate epoch
            epoch_info = {"epoch": epoch, "train_loss": round(train_loss, 5)}

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_logits = self.model(X_val_t)
                    v_loss = float(criterion(v_logits, y_val_t).item())
                    v_preds = v_logits.argmax(dim=-1).numpy()
                    v_eval = evaluate_predictions(y_val, v_preds)

                epoch_info["val_loss"] = round(v_loss, 5)
                epoch_info["val_accuracy"] = v_eval["accuracy"]
                epoch_info["val_balanced_accuracy"] = v_eval["balanced_accuracy"]
                epoch_info["val_macro_f1"] = v_eval["macro_f1"]

                if v_loss < best_val_loss - 1e-4:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                self.model.train()
                if patience_counter >= patience:
                    break
            else:
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch

            self.history.append(epoch_info)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.train_duration_sec = time.time() - t_start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices."""
        t0 = time.time()
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_t)
            preds = logits.argmax(dim=-1).numpy()
        self.inference_duration_sec = time.time() - t0
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=-1).numpy()
        return probs


class SmallMLPModule(nn.Module):
    """
    Model 2 Architecture as specified:
    Input: in_features (110)
    Linear(110 -> 64) -> ReLU -> Dropout(0.10)
    Linear(64 -> 32) -> ReLU -> Dropout(0.10)
    Linear(32 -> 3)
    """

    def __init__(
        self,
        in_features: int = 110,
        hidden1: int = 64,
        hidden2: int = 32,
        num_classes: int = 3,
        dropout_p: float = 0.10,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallMLP:
    """
    Model 2: Small MLP trained via AdamW on CrossEntropyLoss with validation early stopping.
    """

    def __init__(
        self,
        in_features: int = 110,
        hidden1: int = 64,
        hidden2: int = 32,
        num_classes: int = 3,
        dropout_p: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 150,
        class_weight: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.in_features = in_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.class_weight = class_weight
        self.seed = seed

        set_seed(seed)
        self.model = SmallMLPModule(
            in_features=in_features,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.train_duration_sec: float = 0.0
        self.inference_duration_sec: float = 0.0

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        patience: int = 25,
    ) -> SmallMLP:
        """Fit MLP using mini-batch AdamW on training set only."""
        set_seed(self.seed)
        t_start = time.time()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        weight_t = None
        if self.class_weight is not None:
            weight_t = torch.tensor(self.class_weight, dtype=torch.float32)

        criterion = nn.CrossEntropyLoss(weight=weight_t)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

        best_val_loss = float("inf")
        best_epoch = 0
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

            # Evaluate epoch
            epoch_info = {"epoch": epoch, "train_loss": round(train_loss, 5)}

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_logits = self.model(X_val_t)
                    v_loss = float(criterion(v_logits, y_val_t).item())
                    v_preds = v_logits.argmax(dim=-1).numpy()
                    v_eval = evaluate_predictions(y_val, v_preds)

                epoch_info["val_loss"] = round(v_loss, 5)
                epoch_info["val_accuracy"] = v_eval["accuracy"]
                epoch_info["val_balanced_accuracy"] = v_eval["balanced_accuracy"]
                epoch_info["val_macro_f1"] = v_eval["macro_f1"]

                # Monitor validation loss for early stopping
                if v_loss < best_val_loss - 1e-4:
                    best_val_loss = v_loss
                    best_epoch = epoch
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break
            else:
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch

            self.history.append(epoch_info)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.train_duration_sec = time.time() - t_start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices."""
        t0 = time.time()
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_t)
            preds = logits.argmax(dim=-1).numpy()
        self.inference_duration_sec = time.time() - t0
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=-1).numpy()
        return probs
