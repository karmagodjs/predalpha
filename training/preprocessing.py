import pandas as pd

FEATURES = [
    "best_bid",
    "best_ask",
    "mid_price",
    "spread",
    "bid_depth",
    "ask_depth",
    "imbalance",
    "microprice",
]


class FeatureScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, df):

        self.mean = df[FEATURES].mean()

        self.std = (
            df[FEATURES]
            .std()
            .replace(0, 1.0)
        )

        return self

    def transform(self, df):

        if self.mean is None:
            raise RuntimeError(
                "Scaler must be fitted first."
            )

        result = df.copy()

        # Preserve raw values for backtesting
        for feature in FEATURES:

            result[f"{feature}_raw"] = (
                result[feature]
            )

        # Apply TRAIN statistics
        result[FEATURES] = (
            result[FEATURES] - self.mean
        ) / self.std

        return result

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)