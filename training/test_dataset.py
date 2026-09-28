import pandas as pd

from training.sequence_dataset import (
    OrderBookSequenceDataset
)


def main():

    train = pd.read_parquet(
        "data/processed/train_scaled.parquet"
    )

    dataset = OrderBookSequenceDataset(
        train,
        sequence_length=100
    )

    print(
        "Dataset size:",
        len(dataset)
    )

    x, y = dataset[0]

    print(
        "X shape:",
        x.shape
    )

    print(
        "Y:",
        y.item()
    )

    print(
        "X dtype:",
        x.dtype
    )

    print(
        "Y dtype:",
        y.dtype
    )


if __name__ == "__main__":
    main()