import pandas as pd

from training.preprocessing import FEATURES


class FoldScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, train_df):
        missing = [c for c in FEATURES if c not in train_df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        self.mean = train_df[FEATURES].mean()
        self.std = train_df[FEATURES].std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, df):
        if self.mean is None:
            raise RuntimeError("Call fit() on training data first.")

        result = df.copy()
        result[FEATURES] = (result[FEATURES] - self.mean) / self.std
        return result

    def fit_transform(self, train_df):
        self.fit(train_df)
        return self.transform(train_df)


def prepare_fold(fold):
    scaler = FoldScaler()

    train = scaler.fit_transform(fold["train"])
    validation = scaler.transform(fold["validation"])
    test = scaler.transform(fold["test"])

    return train, validation, test, scaler