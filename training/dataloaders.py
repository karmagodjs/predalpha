from torch.utils.data import DataLoader

from training.sequence_dataset import OrderBookSequenceDataset


def create_dataloaders(
    train_path,
    val_path,
    test_path,
    sequence_length=10,
    batch_size=32,
    num_workers=0,
):

    sequence_length = int(sequence_length)
    batch_size = int(batch_size)
    num_workers = int(num_workers)

    print(
        f"Sequence length: {sequence_length}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    train_dataset = OrderBookSequenceDataset(
        train_path,
        sequence_length=sequence_length,
    )

    val_dataset = OrderBookSequenceDataset(
        val_path,
        sequence_length=sequence_length,
    )

    test_dataset = OrderBookSequenceDataset(
        test_path,
        sequence_length=sequence_length,
    )

    print(
        f"Train sequences: {len(train_dataset)}"
    )

    print(
        f"Validation sequences: {len(val_dataset)}"
    )

    print(
        f"Test sequences: {len(test_dataset)}"
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )