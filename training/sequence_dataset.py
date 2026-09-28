import torch
from torch.utils.data import Dataset

from training.preprocessing import FEATURES


class OrderBookSequenceDataset(Dataset):

    def __init__(
        self,
        df,
        sequence_length=100
    ):

        self.df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        self.sequence_length = (
            sequence_length
        )

        if len(self.df) <= sequence_length:

            raise ValueError(
                f"Need more than "
                f"{sequence_length} rows, "
                f"got {len(self.df)}"
            )

        # Validate required features
        missing = [
            feature
            for feature in FEATURES
            if feature not in self.df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing features: {missing}"
            )

    def __len__(self):

        return (
            len(self.df)
            - self.sequence_length
        )

    def __getitem__(self, index):

        start = index
        end = (
            index
            + self.sequence_length
        )

        sequence = self.df.iloc[
            start:end
        ]

        target = self.df.iloc[
            end
        ]["label"]

        x = torch.tensor(
            sequence[FEATURES].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            target,
            dtype=torch.long
        )

        return x, y