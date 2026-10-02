import pandas as pd


def walk_forward_split(
    df,
    n_splits=3,
    initial_train_ratio=0.40,
    validation_ratio=0.10,
    test_ratio=0.10,
    purge_gap=20,
):
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    n = len(df)
    initial_train = int(n * initial_train_ratio)
    val_size = int(n * validation_ratio)
    test_size = int(n * test_ratio)

    if min(initial_train, val_size, test_size) < 1:
        raise ValueError("Train, validation, and test sizes must be at least 1.")

    required = (
        initial_train
        + val_size
        + test_size
        + (n_splits - 1) * test_size
        + 2 * purge_gap
    )

    if required > n:
        raise ValueError(
            f"Not enough rows for {n_splits} folds. "
            f"Need {required}, have {n}."
        )

    folds = []

    for fold_id in range(n_splits):
        train_end = initial_train + fold_id * test_size

        val_start = train_end + purge_gap
        val_end = val_start + val_size

        test_start = val_end + purge_gap
        test_end = test_start + test_size

        train = df.iloc[:train_end].copy()
        validation = df.iloc[val_start:val_end].copy()
        test = df.iloc[test_start:test_end].copy()

        if train.empty or validation.empty or test.empty:
            raise ValueError(f"Fold {fold_id + 1} contains an empty split.")

        folds.append({
            "fold": fold_id + 1,
            "train": train,
            "validation": validation,
            "test": test,
            "purge_gap": purge_gap,
        })

    return folds