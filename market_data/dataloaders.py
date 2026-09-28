import torch
from torch.utils.data import DataLoader

from training.sequence_dataset import SequenceDataset


def create_dataloaders(
    train_path="data/processed/train.parquet",
    val_path="data/processed/val.parquet",
    batch_size=32,
    num_workers=0,
):
    train_dataset = SequenceDataset(
        train_path
    )

    val_dataset = SequenceDataset(
        val_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader