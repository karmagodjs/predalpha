import pandas as pd
import numpy as np


def add_regime_features(df, volatility_window=50, trend_window=50):
    df = df.sort_values("timestamp").copy()

    returns = df["mid_price"].pct_change()

    df["regime_volatility"] = (
        returns.rolling(
            volatility_window,
            min_periods=volatility_window,
        ).std()
    )

    df["regime_trend"] = (
        df["mid_price"]
        / df["mid_price"].shift(trend_window)
        - 1.0
    )

    return df.replace([np.inf, -np.inf], np.nan)


def fit_regime_thresholds(train_df):
    valid_vol = train_df["regime_volatility"].dropna()
    valid_trend = train_df["regime_trend"].dropna()

    if valid_vol.empty or valid_trend.empty:
        raise ValueError("Not enough training data to fit regime thresholds.")

    return {
        "volatility_median": valid_vol.quantile(0.50),
        "trend_deadband": valid_trend.abs().quantile(0.50),
    }


def apply_regimes(df, thresholds):
    result = df.copy()

    high_vol = (
        result["regime_volatility"]
        > thresholds["volatility_median"]
    )

    trend_band = thresholds["trend_deadband"]
    up = result["regime_trend"] > trend_band
    down = result["regime_trend"] < -trend_band

    result["regime"] = "RANGE_LOW_VOL"
    result.loc[high_vol, "regime"] = "RANGE_HIGH_VOL"
    result.loc[up & ~high_vol, "regime"] = "TREND_UP_LOW_VOL"
    result.loc[down & ~high_vol, "regime"] = "TREND_DOWN_LOW_VOL"
    result.loc[up & high_vol, "regime"] = "TREND_UP_HIGH_VOL"
    result.loc[down & high_vol, "regime"] = "TREND_DOWN_HIGH_VOL"

    result.loc[
        result["regime_volatility"].isna()
        | result["regime_trend"].isna(),
        "regime",
    ] = "INSUFFICIENT_HISTORY"

    return result


def main():
    path = "data/processed/feature_market_data.parquet"
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)

    # Chronological initial training period only.
    train_end = int(len(df) * 0.70)
    train = df.iloc[:train_end]

    df = add_regime_features(df)
    train_features = df.iloc[:train_end]

    thresholds = fit_regime_thresholds(train_features)
    df = apply_regimes(df, thresholds)

    print("Regime thresholds fitted on training period:")
    print(thresholds)

    print("\nRegime counts:")
    print(df["regime"].value_counts(dropna=False))

    # Optional: compare future returns by regime, if labels/returns exist.
    if "future_return_50" in df.columns:
        print("\nFuture return by regime:")
        print(
            df.groupby("regime")["future_return_50"]
            .agg(["count", "mean", "median", "std"])
            .round(6)
        )

    output = "data/processed/market_regimes.parquet"
    df.to_parquet(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()