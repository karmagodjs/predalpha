import torch
import torch.nn as nn


class OrderBookTransformer(nn.Module):

    def __init__(
        self,
        input_features=8,
        d_model=64,
        n_heads=4,
        num_layers=2,
        num_classes=3,
        dropout=0.1
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_features,
            d_model
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout,
                batch_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                num_classes
            )
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        return self.classifier(x)