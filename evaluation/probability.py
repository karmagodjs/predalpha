import numpy as np


def confidence_statistics(
    probabilities
):

    probabilities = np.asarray(
        probabilities
    )

    confidence = (
        probabilities.max(axis=1)
    )

    return {
        "mean_confidence":
            float(confidence.mean()),

        "median_confidence":
            float(np.median(confidence)),

        "high_confidence_ratio":
            float(
                (confidence >= 0.70).mean()
            ),

        "very_high_confidence_ratio":
            float(
                (confidence >= 0.90).mean()
            )
    }