"""
Unit Tests for Phase 19B Recurrent Models (SmallGRU, SmallLSTM).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline_v2.models.recurrent_models import (
    SmallGRU,
    SmallGRUModule,
    SmallLSTM,
    SmallLSTMModule,
)


def test_gru_module_shapes_and_parameters():
    """Verify GRU forward pass shapes and exact parameter counts."""
    B, L, D = 16, 10, 11
    model = SmallGRUModule(input_size=D, hidden_size=32, num_layers=1, num_classes=3, dropout_p=0.10)
    x = torch.randn(B, L, D)
    out = model(x)
    assert out.shape == (B, 3)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Exact parameter calculation
    # GRU: 3 * (input_size*hidden_size + hidden_size*hidden_size + hidden_size + hidden_size)
    # 3 * (11*32 + 32*32 + 32 + 32) = 3 * (352 + 1024 + 64) = 3 * 1440 = 4320
    # FC: 32*3 + 3 = 99
    # Total = 4419
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 4419


def test_lstm_module_shapes_and_parameters():
    """Verify LSTM forward pass shapes and exact parameter counts."""
    B, L, D = 16, 10, 11
    model = SmallLSTMModule(input_size=D, hidden_size=32, num_layers=1, num_classes=3, dropout_p=0.10)
    x = torch.randn(B, L, D)
    out = model(x)
    assert out.shape == (B, 3)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Exact parameter calculation
    # LSTM: 4 * (input_size*hidden_size + hidden_size*hidden_size + hidden_size + hidden_size)
    # 4 * (11*32 + 32*32 + 32 + 32) = 4 * (352 + 1024 + 64) = 4 * 1440 = 5760
    # FC: 32*3 + 3 = 99
    # Total = 5859
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 5859


def test_gru_trainer_fit_and_predict():
    """Verify SmallGRU training, prediction, and probability outputs."""
    np.random.seed(42)
    N, L, D = 100, 10, 11
    X_train = np.random.randn(N, L, D).astype(np.float32)
    y_train = np.random.choice([0, 1, 2], size=N).astype(np.int64)

    X_val = np.random.randn(30, L, D).astype(np.float32)
    y_val = np.random.choice([0, 1, 2], size=30).astype(np.int64)

    trainer = SmallGRU(
        input_size=11,
        hidden_size=32,
        max_epochs=5,
        batch_size=32,
        seed=42,
    )
    trainer.fit(X_train, y_train, X_val, y_val, patience=3)

    preds = trainer.predict(X_val)
    probs = trainer.predict_proba(X_val)

    assert preds.shape == (30,)
    assert set(preds).issubset({0, 1, 2})
    assert probs.shape == (30, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert len(trainer.history) > 0


def test_lstm_trainer_fit_and_predict():
    """Verify SmallLSTM training, prediction, and probability outputs."""
    np.random.seed(42)
    N, L, D = 100, 10, 11
    X_train = np.random.randn(N, L, D).astype(np.float32)
    y_train = np.random.choice([0, 1, 2], size=N).astype(np.int64)

    X_val = np.random.randn(30, L, D).astype(np.float32)
    y_val = np.random.choice([0, 1, 2], size=30).astype(np.int64)

    trainer = SmallLSTM(
        input_size=11,
        hidden_size=32,
        max_epochs=5,
        batch_size=32,
        seed=42,
    )
    trainer.fit(X_train, y_train, X_val, y_val, patience=3)

    preds = trainer.predict(X_val)
    probs = trainer.predict_proba(X_val)

    assert preds.shape == (30,)
    assert set(preds).issubset({0, 1, 2})
    assert probs.shape == (30, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert len(trainer.history) > 0


def test_recurrent_checkpoint_save_and_load():
    """Verify saving and loading checkpoints for GRU and LSTM."""
    np.random.seed(42)
    N, L, D = 50, 10, 11
    X = np.random.randn(N, L, D).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=N).astype(np.int64)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gru_path = tmp_path / "gru.pt"
        lstm_path = tmp_path / "lstm.pt"

        # GRU
        gru = SmallGRU(input_size=11, hidden_size=32, max_epochs=2, seed=42)
        gru.fit(X, y)
        gru.save_checkpoint(gru_path)
        assert gru_path.exists()

        gru_loaded = SmallGRU(input_size=11, hidden_size=32, seed=42)
        gru_loaded.load_checkpoint(gru_path)
        preds1 = gru.predict(X)
        preds2 = gru_loaded.predict(X)
        assert np.array_equal(preds1, preds2)

        # LSTM
        lstm = SmallLSTM(input_size=11, hidden_size=32, max_epochs=2, seed=42)
        lstm.fit(X, y)
        lstm.save_checkpoint(lstm_path)
        assert lstm_path.exists()

        lstm_loaded = SmallLSTM(input_size=11, hidden_size=32, seed=42)
        lstm_loaded.load_checkpoint(lstm_path)
        preds1 = lstm.predict(X)
        preds2 = lstm_loaded.predict(X)
        assert np.array_equal(preds1, preds2)


def test_recurrent_seed_reproducibility():
    """Verify identical random seeds produce bitwise identical model weights and predictions."""
    np.random.seed(123)
    X = np.random.randn(50, 10, 11).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=50).astype(np.int64)

    m1 = SmallGRU(seed=42, max_epochs=3)
    m1.fit(X, y)
    p1 = m1.predict_proba(X)

    m2 = SmallGRU(seed=42, max_epochs=3)
    m2.fit(X, y)
    p2 = m2.predict_proba(X)

    assert np.allclose(p1, p2, atol=1e-6)
