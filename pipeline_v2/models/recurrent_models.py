"""
Recurrent Sequence Models for Phase 19B Small Recurrent Sequence Modeling.

Contains:
1. SmallGRUModule / SmallGRU: 1-layer GRU (input 11 -> hidden 32) -> Dropout(0.10) -> Linear(32 -> 3).
2. SmallLSTMModule / SmallLSTM: 1-layer LSTM (input 11 -> hidden 32) -> Dropout(0.10) -> Linear(32 -> 3).

Both architectures consume causal sequence tensors of shape (N, L=10, D=11)
and strictly enforce temporal causality without future lookahead.
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

from pipeline_v2.models.baseline_models import set_seed
from pipeline_v2.models.metrics import evaluate_predictions


class SmallGRUModule(nn.Module):
    """
    Model A Architecture:
    Input: (B, 10, 11)
    GRU(input_size=11, hidden_size=32, num_layers=1, batch_first=True)
    Dropout(p=0.10)
    Linear(32 -> 3)
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout_p: float = 0.10,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over sequence.
        x: (B, L=10, D=11)
        Returns logits: (B, 3)
        """
        # out: (B, L, H), h_n: (num_layers, B, H)
        out, h_n = self.gru(x)
        # Extract endpoint hidden state at final step
        last_hidden = h_n[-1]  # (B, H)
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits


class SmallLSTMModule(nn.Module):
    """
    Model B Architecture:
    Input: (B, 10, 11)
    LSTM(input_size=11, hidden_size=32, num_layers=1, batch_first=True)
    Dropout(p=0.10)
    Linear(32 -> 3)
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout_p: float = 0.10,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over sequence.
        x: (B, L=10, D=11)
        Returns logits: (B, 3)
        """
        # out: (B, L, H), h_n: (num_layers, B, H), c_n: (num_layers, B, H)
        out, (h_n, c_n) = self.lstm(x)
        # Extract endpoint hidden state at final step
        last_hidden = h_n[-1]  # (B, H)
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits


class SmallGRU:
    """
    Model A Trainer: Small GRU trained via AdamW on CrossEntropyLoss with validation early stopping.
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout_p: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 150,
        class_weight: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.class_weight = class_weight
        self.seed = seed

        set_seed(seed)
        self.model = SmallGRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.train_duration_sec: float = 0.0
        self.inference_duration_sec: float = 0.0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

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
    ) -> SmallGRU:
        """Fit GRU using mini-batch AdamW on training set only."""
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
                    v_preds = v_logits.argmax(dim=-1).numpy()
                    v_eval = evaluate_predictions(y_val, v_preds)

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

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.train_duration_sec = round(time.time() - t_start, 4)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices (0, 1, 2) in eval mode."""
        self.model.eval()
        t_start = time.time()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu().numpy()
        self.inference_duration_sec = round(time.time() - t_start, 4)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via softmax."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def evaluate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate cross-entropy loss without grad."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        weight_t = torch.tensor(self.class_weight, dtype=torch.float32) if self.class_weight is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight_t)
        with torch.no_grad():
            logits = self.model(X_t)
            loss = float(criterion(logits, y_t).item())
        return loss

    def save_checkpoint(self, path: str | Any) -> None:
        """Save PyTorch state dict and model hyperparameters."""
        checkpoint = {
            "model_type": "SmallGRU",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout_p": self.dropout_p,
            "state_dict": self.model.state_dict(),
            "parameter_count": self.parameter_count,
            "seed": self.seed,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_duration_sec": self.train_duration_sec,
        }
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: str | Any) -> SmallGRU:
        """Load PyTorch state dict from path."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return self


class SmallLSTM:
    """
    Model B Trainer: Small LSTM trained via AdamW on CrossEntropyLoss with validation early stopping.
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout_p: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 150,
        class_weight: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.class_weight = class_weight
        self.seed = seed

        set_seed(seed)
        self.model = SmallLSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.train_duration_sec: float = 0.0
        self.inference_duration_sec: float = 0.0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

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
    ) -> SmallLSTM:
        """Fit LSTM using mini-batch AdamW on training set only."""
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
                    v_preds = v_logits.argmax(dim=-1).numpy()
                    v_eval = evaluate_predictions(y_val, v_preds)

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

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.train_duration_sec = round(time.time() - t_start, 4)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices (0, 1, 2) in eval mode."""
        self.model.eval()
        t_start = time.time()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu().numpy()
        self.inference_duration_sec = round(time.time() - t_start, 4)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via softmax."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def evaluate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate cross-entropy loss without grad."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        weight_t = torch.tensor(self.class_weight, dtype=torch.float32) if self.class_weight is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight_t)
        with torch.no_grad():
            logits = self.model(X_t)
            loss = float(criterion(logits, y_t).item())
        return loss

    def save_checkpoint(self, path: str | Any) -> None:
        """Save PyTorch state dict and model hyperparameters."""
        checkpoint = {
            "model_type": "SmallLSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout_p": self.dropout_p,
            "state_dict": self.model.state_dict(),
            "parameter_count": self.parameter_count,
            "seed": self.seed,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_duration_sec": self.train_duration_sec,
        }
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: str | Any) -> SmallLSTM:
        """Load PyTorch state dict from path."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return self
