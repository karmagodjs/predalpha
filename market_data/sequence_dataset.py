import torch
from torch.utils.data import Dataset

from training.preprocessing import FEATURES


class OrderBookSequenceDataset(Dataset):

    def __init__(self, df, sequence_length=100):

        self.df = df.reset_index(drop=True)
        self.sequence_length = sequence_length

        if len(self.df) <= sequence_length:
            raise ValueError(
                f"Need more than {sequence_length} rows, "
                f"got {len(self.df)}"
            )

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, index):

        end = index + self.sequence_length

        sequence = self.df.iloc[index:end]

        x = torch.tensor(
            sequence[FEATURES].values,
            dtype=torch.float32
        )

        # Label immediately after the sequence
        y = torch.tensor(
            self.df.iloc[end]["label"],
            dtype=torch.long
        )

        return x, y