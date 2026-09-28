import torch


def calculate_class_weights(df):

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nClass counts:")

    print(counts)

    total = counts.sum()

    num_classes = 3

    weights = []

    for class_id in range(num_classes):

        count = counts.get(
            class_id,
            0
        )

        if count == 0:

            weight = 0.0

        else:

            weight = total / (
                num_classes * count
            )

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    print(
        "\nClass weights:",
        weights
    )

    return weights