"""
Causal Sequence Dataset and DataLoader with Warm-up NaN Handling.

Implements Phase 17/18 Tensor Collation & DataLoader Level Warm-Up NaN Handling:
1. Authentic Data Preservation:
   - Does NOT delete valid sequences.
   - Does NOT modify raw or scaled stored dataset artifacts to hide NaNs.
   - Sequences preserve authentic microstructure values including causal warm-up NaNs in steps 0-4.
2. Tensor Collation Level Warm-Up NaN Handling:
   - Causal Valid Mask: `causal_mask = ~torch.isnan(x)` indicating valid observations vs lookback warmup NaNs.
   - Safe Zero-Imputation: `x_imputed = torch.nan_to_num(x, nan=0.0)` strictly zero-imputing unobserved warm-up features.
   - Step-Level Valid Mask: `step_mask = causal_mask.all(dim=-1)` indicating timesteps where all features are available.
   - Key Padding Mask: `key_padding_mask = ~step_mask` for standard PyTorch Transformer attention masking.
   - Zero-NaN Guarantee: Output tensor `x_imputed` is guaranteed to contain strictly 0 NaNs and 0 Infs before model forward pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("pipeline_v2.causal_dataloader")

# Standard class mappings for PredAlpha-HFT
CLASS_TO_IDX = {"DOWN": 0, "FLAT": 1, "UP": 2}
IDX_TO_CLASS = {0: "DOWN", 1: "FLAT", 2: "UP"}


class CausalSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping causal 3D sequence tensors.
    
    Preserves authentic feature values with zero modification or filtering.
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, List[str], pd.Series],
        endpoints: Optional[Union[np.ndarray, torch.Tensor, List[int], pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X.copy()).float()
        else:
            self.X = X.clone().detach().float()

        # Handle class label encoding if string
        if isinstance(y, (list, tuple)) or isinstance(y, pd.Series):
            y_arr = np.asarray(y)
        elif isinstance(y, np.ndarray):
            y_arr = y
        elif isinstance(y, torch.Tensor):
            y_arr = y.cpu().numpy()
        else:
            raise TypeError(f"Unsupported label type: {type(y)}")

        if y_arr.dtype.kind in ("U", "S", "O"):
            y_idx = np.array([CLASS_TO_IDX.get(str(label).strip().upper(), 1) for label in y_arr], dtype=np.int64)
            self.y = torch.from_numpy(y_idx).long()
        else:
            self.y = torch.as_tensor(y_arr, dtype=torch.long)

        if endpoints is not None:
            ep_arr = np.asarray(endpoints, dtype=np.int64)
            self.endpoints = torch.from_numpy(ep_arr).long()
        else:
            self.endpoints = torch.arange(len(self.X), dtype=torch.long)

        self.feature_names = list(feature_names) if feature_names is not None else []

        assert len(self.X) == len(self.y), f"X and y length mismatch: {len(self.X)} != {len(self.y)}"
        assert len(self.X) == len(self.endpoints), f"X and endpoints length mismatch: {len(self.X)} != {len(self.endpoints)}"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "endpoint": self.endpoints[idx],
        }


def causal_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function implementing causal mask + safe zero-imputation at batch collation level.
    
    Guarantees:
    1. Input tensors `x` are stacked into batch `(B, L, D)`.
    2. `causal_mask` of shape `(B, L, D)` captures exact valid vs warm-up NaN locations.
    3. `x_imputed` replaces all NaNs with 0.0, strictly ensuring 0 NaNs and 0 Infs in forward pass.
    4. `step_valid_mask` of shape `(B, L)` indicates which steps are completely free of warm-up NaNs.
    5. `key_padding_mask` of shape `(B, L)` provides standard PyTorch Transformer attention masks.
    """
    x_raw = torch.stack([item["x"] for item in batch], dim=0)  # Shape: (B, L, D)
    y_batch = torch.stack([item["y"] for item in batch], dim=0)  # Shape: (B,)
    ep_batch = torch.stack([item["endpoint"] for item in batch], dim=0)  # Shape: (B,)

    # 1. Compute causal valid mask (True = valid finite value, False = warmup NaN)
    causal_mask = ~torch.isnan(x_raw)

    # 2. Safe zero-imputation
    x_imputed = torch.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Invariant assertions
    assert torch.isnan(x_imputed).sum() == 0, "Collation failed: NaN detected in imputed tensor!"
    assert torch.isinf(x_imputed).sum() == 0, "Collation failed: Inf detected in imputed tensor!"
    assert torch.isnan(y_batch).sum() == 0, "Collation failed: NaN detected in target labels!"

    # 3. Timestep level valid mask (True if all features at step t are valid)
    step_valid_mask = causal_mask.all(dim=-1)  # Shape: (B, L)

    # 4. PyTorch Transformer key_padding_mask (True = ignored/masked position)
    key_padding_mask = ~step_valid_mask  # Shape: (B, L)

    return {
        "x": x_imputed,                         # (B, L, D) float32 zero-imputed
        "x_raw": x_raw,                         # (B, L, D) float32 raw (with warmup NaNs)
        "causal_mask": causal_mask,             # (B, L, D) bool valid mask
        "step_valid_mask": step_valid_mask,     # (B, L) bool
        "key_padding_mask": key_padding_mask,   # (B, L) bool
        "y": y_batch,                           # (B,) int64
        "endpoint": ep_batch,                   # (B,) int64
    }


def create_causal_dataloaders(
    scaled_data_source: Union[Path, str, Dict[str, np.ndarray]],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train, validation, and test DataLoaders equipped with causal warm-up NaN handling.
    """
    if isinstance(scaled_data_source, (str, Path)):
        source_p = Path(scaled_data_source)
        if (source_p / "scaled_production.npz").exists():
            npz = np.load(source_p / "scaled_production.npz", allow_pickle=True)
            X_train, y_train, ep_train = npz["train_X"], npz["train_y"], npz["train_endpoints"]
            X_val, y_val, ep_val = npz["val_X"], npz["val_y"], npz["val_endpoints"]
            X_test, y_test, ep_test = npz["test_X"], npz["test_y"], npz["test_endpoints"]
            feature_names = list(npz["feature_names"]) if "feature_names" in npz else []
        elif (source_p / "train_scaled.npz").exists():
            tr = np.load(source_p / "train_scaled.npz", allow_pickle=True)
            va = np.load(source_p / "validation_scaled.npz", allow_pickle=True)
            te = np.load(source_p / "test_scaled.npz", allow_pickle=True)
            X_train, y_train, ep_train = tr["X"], tr["y"], tr["endpoints"]
            X_val, y_val, ep_val = va["X"], va["y"], va["endpoints"]
            X_test, y_test, ep_test = te["X"], te["y"], te["endpoints"]
            feature_names = list(tr["feature_names"]) if "feature_names" in tr else []
        elif source_p.suffix == ".npz":
            npz = np.load(source_p, allow_pickle=True)
            X_train, y_train, ep_train = npz["train_X"], npz["train_y"], npz["train_endpoints"]
            X_val, y_val, ep_val = npz["val_X"], npz["val_y"], npz["val_endpoints"]
            X_test, y_test, ep_test = npz["test_X"], npz["test_y"], npz["test_endpoints"]
            feature_names = list(npz["feature_names"]) if "feature_names" in npz else []
        else:
            raise FileNotFoundError(f"Could not locate scaled NPZ artifacts under {source_p}")
    elif isinstance(scaled_data_source, dict):
        X_train, y_train, ep_train = scaled_data_source["train_X"], scaled_data_source["train_y"], scaled_data_source["train_endpoints"]
        X_val, y_val, ep_val = scaled_data_source["val_X"], scaled_data_source["val_y"], scaled_data_source["val_endpoints"]
        X_test, y_test, ep_test = scaled_data_source["test_X"], scaled_data_source["test_y"], scaled_data_source["test_endpoints"]
        feature_names = list(scaled_data_source.get("feature_names", []))
    else:
        raise TypeError(f"Unsupported scaled data source: {type(scaled_data_source)}")

    train_ds = CausalSequenceDataset(X_train, y_train, ep_train, feature_names=feature_names)
    val_ds = CausalSequenceDataset(X_val, y_val, ep_val, feature_names=feature_names)
    test_ds = CausalSequenceDataset(X_test, y_test, ep_test, feature_names=feature_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=causal_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=causal_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=causal_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    metadata = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "total_samples": len(train_ds) + len(val_ds) + len(test_ds),
        "batch_size": batch_size,
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "sequence_length": X_train.shape[1] if len(X_train) > 0 else 10,
    }

    return train_loader, val_loader, test_loader, metadata
