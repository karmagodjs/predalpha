import pandas as pd
import torch

from torch.utils.data import Dataset


class OrderBookDataset(Dataset):

    def __init__(self, path):

        self.data = pd.read_parquet(path)

        self.features = [
            "best_bid",
            "best_ask",
            "mid_price",
            "spread",
            "bid_depth",
            "ask_depth",
            "imbalance",
            "microprice",
        ]

        self.data = self.data.dropna(
            subset=self.features
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        x = torch.tensor(
            row[self.features].values,
            dtype=torch.float32
        )

        return x