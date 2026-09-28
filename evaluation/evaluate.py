from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from models.orderbook_transformer import (
    OrderBookTransformer
)

from training.sequence_dataset import (
    OrderBookSequenceDataset
)


TEST_PATH = (
    "data/processed/test_scaled.parquet"
)

MODEL_PATH = Path(
    "checkpoints/best_orderbook_transformer.pt"
)

SEQUENCE_LENGTH = 10
BATCH_SIZE = 64


def main():

    print("=" * 60)
    print("PREDALPHA OUT-OF-SAMPLE EVALUATION")
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
    # Load test data
    # --------------------------------------------------

    test_df = pd.read_parquet(
        TEST_PATH
    )

    print(
        "Test rows:",
        len(test_df)
    )

    print(
        "\nTest label distribution:"
    )

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    test_dataset = (
        OrderBookSequenceDataset(
            test_df,
            sequence_length=SEQUENCE_LENGTH
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "\nTest sequences:",
        len(test_dataset)
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

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "\nLoaded checkpoint:"
    )

    print(
        "Epoch:",
        checkpoint.get("epoch")
    )

    print(
        "Validation loss:",
        checkpoint.get("val_loss")
    )

    print(
        "Validation accuracy:",
        checkpoint.get("val_accuracy")
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)

            logits = model(x)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = (
                probabilities.argmax(
                    dim=1
                )
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.asarray(
        all_labels
    )

    y_pred = np.asarray(
        all_predictions
    )

    probabilities = np.asarray(
        all_probabilities
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[
                "DOWN",
                "FLAT",
                "UP",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    print(
        "CONFUSION MATRIX"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2],
        )
    )

    # --------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------

    print(
        "\nPrediction distribution:"
    )

    unique, counts = np.unique(
        y_pred,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        name = {
            0: "DOWN",
            1: "FLAT",
            2: "UP",
        }.get(
            int(label),
            "UNKNOWN"
        )

        print(
            f"{name}: {count}"
        )

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    confidence = probabilities.max(
        axis=1
    )

    print(
        "\nConfidence:"
    )

    print(
        pd.Series(
            confidence
        ).describe()
    )

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    output = test_df.iloc[
        SEQUENCE_LENGTH:
    ].copy()

    output["prediction"] = y_pred

    output["prob_down"] = (
        probabilities[:, 0]
    )

    output["prob_flat"] = (
        probabilities[:, 1]
    )

    output["prob_up"] = (
        probabilities[:, 2]
    )

    output["confidence"] = confidence

    output_path = Path(
        "data/processed/"
        "test_predictions.parquet"
    )

    output.to_parquet(
        output_path,
        index=False
    )

    print(
        "\nPredictions saved:"
    )

    print(output_path)


if __name__ == "__main__":
    main()