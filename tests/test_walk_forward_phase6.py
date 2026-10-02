import pandas as pd

from training.walk_forward import walk_forward_split
from training.fold_preprocessing import prepare_fold


def test_purged_walk_forward_boundaries():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=1000, freq="s"),
        "feature": range(1000),
    })

    folds = walk_forward_split(
        df,
        n_splits=3,
        initial_train_ratio=0.40,
        validation_ratio=0.10,
        test_ratio=0.10,
        purge_gap=20,
    )

    for fold in folds:
        train = fold["train"]
        val = fold["validation"]
        test = fold["test"]

        assert train.timestamp.max() < val.timestamp.min()
        assert val.timestamp.max() < test.timestamp.min()

        # Ensure the requested row gap is respected.
        assert val.index.min() - train.index.max() - 1 == 20
        assert test.index.min() - val.index.max() - 1 == 20


def test_scaler_fits_training_only():
    from training.preprocessing import FEATURES

    rows = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="s"),
        **{feature: [float(i) for i in range(rows)] for feature in FEATURES},
    })

    fold = {
        "train": df.iloc[:60].copy(),
        "validation": df.iloc[60:80].copy(),
        "test": df.iloc[80:].copy(),
    }

    train, val, test, scaler = prepare_fold(fold)

    assert scaler.mean[FEATURES[0]] == df.iloc[:60][FEATURES[0]].mean()
    assert abs(train[FEATURES].mean().max()) < 1e-9
    assert val[FEATURES].iloc[0, 0] > 0
    assert test[FEATURES].iloc[0, 0] > val[FEATURES].iloc[0, 0]