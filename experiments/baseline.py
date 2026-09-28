from collections import Counter


def majority_class_predict(
    y_train,
    y_test
):

    majority = Counter(
        y_train
    ).most_common(1)[0][0]

    predictions = [
        majority
        for _ in y_test
    ]

    return predictions