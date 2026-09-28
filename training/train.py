from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.orderbook_transformer import (
    OrderBookTransformer
)

from training.sequence_dataset import (
    OrderBookSequenceDataset
)

from training.class_weights import (
    calculate_class_weights
)


TRAIN_PATH = (
    "data/processed/train_scaled.parquet"
)

VAL_PATH = (
    "data/processed/val_scaled.parquet"
)

SEQUENCE_LENGTH = 10
BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

MODEL_DIR = Path(
    "checkpoints"
)

MODEL_PATH = (
    MODEL_DIR
    / "best_orderbook_transformer.pt"
)


def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item()
                * x.size(0)
            )

            predictions = (
                logits.argmax(dim=1)
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

    return (
        total_loss / total,
        correct / total
    )


def main():

    print("=" * 60)
    print("PREDALPHA TRANSFORMER TRAINING")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    import pandas as pd

    train_df = pd.read_parquet(
        TRAIN_PATH
    )

    val_df = pd.read_parquet(
        VAL_PATH
    )

    print(
        "\nTrain rows:",
        len(train_df)
    )

    print(
        "Validation rows:",
        len(val_df)
    )

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = (
        OrderBookSequenceDataset(
            train_df,
            sequence_length=SEQUENCE_LENGTH
        )
    )

    val_dataset = (
        OrderBookSequenceDataset(
            val_df,
            sequence_length=SEQUENCE_LENGTH
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "\nTrain sequences:",
        len(train_dataset)
    )

    print(
        "Validation sequences:",
        len(val_dataset)
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = OrderBookTransformer(
        input_features=8,
        sequence_length=SEQUENCE_LENGTH,
        d_model=64,
        n_heads=4,
        num_layers=2,
        num_classes=3,
        dropout=0.1,
    ).to(device)

    print(
        "\nModel parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # --------------------------------------------------
    # Class weights
    # --------------------------------------------------

    weights = calculate_class_weights(
        train_df
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")

    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * x.size(0)
            )

            predictions = (
                logits.argmax(dim=1)
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

        train_loss = (
            total_loss / total
        )

        train_accuracy = (
            correct / total
        )

        val_loss, val_accuracy = (
            evaluate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        scheduler.step(
            val_loss
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.6f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss":
                        val_loss,
                    "val_accuracy":
                        val_accuracy,
                },
                MODEL_PATH,
            )

            print(
                "✓ Best model saved."
            )

        else:

            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

            if (
                patience_counter
                >= PATIENCE
            ):

                print(
                    "\nEarly stopping."
                )

                break

    print(
        "\nTraining complete."
    )

    print(
        "Checkpoint:",
        MODEL_PATH
    )


if __name__ == "__main__":
    main()