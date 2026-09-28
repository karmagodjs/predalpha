import torch


class AlphaSignal:
    """
    Converts model probabilities into trading signals.

    Classes:
        0 -> DOWN  -> SHORT
        1 -> FLAT  -> NO_TRADE
        2 -> UP    -> LONG
    """

    def __init__(self, confidence_threshold=0.60):
        self.confidence_threshold = confidence_threshold

    def generate(self, probabilities):
        predictions = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1).values

        signals = []

        for prediction, confidence in zip(
            predictions,
            confidences
        ):
            confidence = confidence.item()
            prediction = prediction.item()

            if confidence < self.confidence_threshold:
                signals.append("NO_TRADE")

            elif prediction == 2:
                signals.append("LONG")

            elif prediction == 0:
                signals.append("SHORT")

            else:
                signals.append("NO_TRADE")

        return signals


def generate_signal(
    probabilities,
    min_confidence=0.60
):
    """
    Backward-compatible helper function.
    """

    signal_engine = AlphaSignal(
        confidence_threshold=min_confidence
    )

    return signal_engine.generate(probabilities)