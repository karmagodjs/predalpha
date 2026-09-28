import torch
import torch.nn as nn


class OrderBookTransformer(nn.Module):

    def __init__(
        self,
        input_features=8,
        sequence_length=100,
        d_model=64,
        n_heads=4,
        num_layers=2,
        num_classes=3,
        dropout=0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_features,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                sequence_length,
                d_model
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),

            nn.Linear(
                d_model,
                32
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                32,
                num_classes
            ),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = (
            x
            + self.position_embedding[
                :, :x.size(1), :
            ]
        )

        x = self.encoder(x)

        # Last timestep representation
        x = x[:, -1, :]

        return self.head(x)