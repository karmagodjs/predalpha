import pandas as pd


def add_future_return(
    df: pd.DataFrame,
    horizon: int = 10,
):
    """
    Calculate future microprice return.

    Creates:
        future_return
        future_return_<horizon>
    """

    df = df.copy()

    future_price = (
        df["microprice"]
        .shift(-horizon)
    )

    future_return = (
        future_price - df["microprice"]
    ) / df["microprice"]

    # Main/backward-compatible column
    df["future_return"] = future_return

    # Horizon-specific column
    df[f"future_return_{horizon}"] = future_return

    return df


def create_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 0.00005,
):
    """
    Create 3-class directional labels.

    Labels:
        0 = DOWN
        1 = FLAT
        2 = UP
    """

    df = add_future_return(
        df,
        horizon=horizon,
    )

    # Default = FLAT
    df["label"] = 1

    # DOWN
    df.loc[
        df["future_return"] < -threshold,
        "label",
    ] = 0

    # UP
    df.loc[
        df["future_return"] > threshold,
        "label",
    ] = 2

    # Remove rows where future price doesn't exist
    df = df.dropna(
        subset=["future_return"]
    ).reset_index(drop=True)

    return df