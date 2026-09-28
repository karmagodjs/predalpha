import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average=None,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,

        "down_precision": precision[0],
        "flat_precision": precision[1],
        "up_precision": precision[2],

        "down_recall": recall[0],
        "flat_recall": recall[1],
        "up_recall": recall[2],

        "down_f1": f1[0],
        "flat_f1": f1[1],
        "up_f1": f1[2],
    }


def print_metrics(
    y_true,
    y_pred
):

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
                "UP"
            ],
            zero_division=0
        )
    )

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )

    print(matrix)

    return matrix