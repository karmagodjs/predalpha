def temporal_split(
    df,
    train_ratio=0.70,
    val_ratio=0.15
):

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    val_end = int(
        n * (train_ratio + val_ratio)
    )

    train = df.iloc[
        :train_end
    ].copy()

    val = df.iloc[
        train_end:val_end
    ].copy()

    test = df.iloc[
        val_end:
    ].copy()

    return train, val, test