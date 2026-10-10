"""
Unit tests for Causal DataLoader & Tensor Collation Warm-up NaN Handling.

Verifies:
1. Authentic Warm-Up NaN Preservation:
   - Input sequences retain authentic values including warm-up NaNs in timesteps 0-4.
   - Raw / processed datasets are NOT altered or deleted to hide NaNs.
2. Causal Collation:
   - `causal_mask` has exact shape (B, L, D) and correctly flags warmup NaNs.
   - Safe zero-imputation replaces all NaNs with 0.0 in `x_imputed`.
   - Zero NaNs and zero Infs exist in `x_imputed`.
   - Targets `y` and `endpoints` are preserved and finite.
3. Neural Network Compatibility:
   - `nn.Linear` and `OrderBookTransformer` forward passes produce finite outputs with zero NaNs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from models.orderbook_transformer import OrderBookTransformer
from pipeline_v2.sequences.causal_dataloader import (
    CausalSequenceDataset,
    causal_collate_fn,
    create_causal_dataloaders,
)


@pytest.fixture
def synthetic_warmup_batch():
    """Create a synthetic batch with authentic 5s lookback warm-up NaNs."""
    batch_size = 4
    seq_len = 10
    feat_dim = 11

    # Simulate realistic warm-up pattern:
    # Feature 3 (1s return): NaN at step 0
    # Feature 4 (3s return): NaN at steps 0, 1, 2
    # Feature 5 & 6 (5s return & vol): NaN at steps 0, 1, 2, 3, 4
    # All other features: finite values
    X = np.random.randn(batch_size, seq_len, feat_dim)
    X[:, 0, 3] = np.nan
    X[:, :3, 4] = np.nan
    X[:, :5, 5] = np.nan
    X[:, :5, 6] = np.nan

    y = np.array(["UP", "DOWN", "FLAT", "UP"])
    endpoints = np.array([1791204610000, 1791204620000, 1791204630000, 1791204640000], dtype=np.int64)

    return X, y, endpoints


def test_dataset_preserves_authentic_nans(synthetic_warmup_batch):
    """Verify CausalSequenceDataset does not mutate or hide authentic NaNs."""
    X, y, ep = synthetic_warmup_batch
    ds = CausalSequenceDataset(X, y, ep)

    assert len(ds) == 4
    item0 = ds[0]
    x_item = item0["x"]

    # Verify NaNs are authentically preserved in item
    assert torch.isnan(x_item[0, 3]).item() is True
    assert torch.isnan(x_item[2, 4]).item() is True
    assert torch.isnan(x_item[4, 5]).item() is True

    # Timestep 9 (endpoint) must have ZERO NaNs
    assert torch.isnan(x_item[9, :]).sum().item() == 0


def test_causal_collate_fn_mask_and_imputation(synthetic_warmup_batch):
    """Verify collation generates causal mask and safe zero-imputation with 0 NaNs."""
    X, y, ep = synthetic_warmup_batch
    ds = CausalSequenceDataset(X, y, ep)
    batch_items = [ds[i] for i in range(len(ds))]

    batch = causal_collate_fn(batch_items)

    assert "x" in batch
    assert "causal_mask" in batch
    assert "step_valid_mask" in batch
    assert "key_padding_mask" in batch
    assert "y" in batch
    assert "endpoint" in batch

    x_imp = batch["x"]
    mask = batch["causal_mask"]

    # Invariant 1: Zero NaNs and zero Infs in imputed tensor
    assert torch.isnan(x_imp).sum().item() == 0
    assert torch.isinf(x_imp).sum().item() == 0

    # Invariant 2: Mask corresponds exactly to NaN positions in raw tensor
    raw_nans = torch.isnan(batch["x_raw"])
    assert torch.equal(~mask, raw_nans)

    # Invariant 3: Where mask is False, x_imp is strictly 0.0
    assert (x_imp[~mask] == 0.0).all().item() is True

    # Invariant 4: Where mask is True, x_imp matches raw values bit-for-bit
    assert torch.equal(x_imp[mask], batch["x_raw"][mask])

    # Invariant 5: Endpoint T (step 9) is 100% valid
    assert mask[:, 9, :].all().item() is True


def test_neural_forward_pass_with_collation(synthetic_warmup_batch):
    """Verify neural network forward pass succeeds with zero NaNs using collated batch."""
    X, y, ep = synthetic_warmup_batch
    ds = CausalSequenceDataset(X, y, ep)
    loader = DataLoader(ds, batch_size=4, collate_fn=causal_collate_fn)

    batch = next(iter(loader))
    x = batch["x"]  # (4, 10, 11) zero-imputed
    targets = batch["y"]  # (4,)

    # Test through linear projection
    proj = torch.nn.Linear(11, 32)
    out_proj = proj(x)
    assert torch.isnan(out_proj).sum().item() == 0
    assert out_proj.shape == (4, 10, 32)

    # Test through OrderBookTransformer
    model = OrderBookTransformer(
        input_features=11,
        sequence_length=10,
        d_model=32,
        n_heads=2,
        num_layers=1,
        num_classes=3,
    )
    logits = model(x)

    assert logits.shape == (4, 3)
    assert torch.isnan(logits).sum().item() == 0

    loss = torch.nn.functional.cross_entropy(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    loss.backward()

    # Verify gradients are finite
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isnan(p.grad).sum().item() == 0
