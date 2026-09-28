import torch

from models.orderbook_transformer import (
    OrderBookTransformer
)


def test_transformer_shape():

    model = OrderBookTransformer(
        input_features=8,
        sequence_length=100,
        d_model=64,
        n_heads=4,
        num_layers=2,
        num_classes=3
    )

    x = torch.randn(
        16,
        100,
        8
    )

    output = model(x)

    assert output.shape == (
        16,
        3
    )


def test_prediction_classes():

    model = OrderBookTransformer()

    x = torch.randn(
        4,
        100,
        8
    )

    logits = model(x)

    predictions = logits.argmax(
        dim=1
    )

    assert predictions.shape == (4,)

    assert torch.all(
        (predictions >= 0)
        &
        (predictions <= 2)
    )