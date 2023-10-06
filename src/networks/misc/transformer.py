import torch
from torch import nn

from src.networks.misc.cln import ConditionalLayerNormalization
from src.networks.misc.positional_encoding import PositionalEncoding


class TransformerBlock(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        channels: int,
        conditionning_channels: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        """Initializer."""
        super().__init__()
        self.pos_encoder = PositionalEncoding(channels, dropout=dropout)
        self.transformer = nn.Transformer(
            channels,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
        )
        self.dropout = nn.Dropout(dropout)
        self.cond1 = ConditionalLayerNormalization(conditionning_channels, channels)
        self.linear = nn.Linear(channels, channels)
        self.cond2 = ConditionalLayerNormalization(conditionning_channels, channels)

    def forward(
        self, features: torch.Tensor, style_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [B, C, N]
            style_embedding: [B, C_style, 1]
        """
        x_transposed = features.transpose(1, 2)
        x_transposed = self.pos_encoder(x_transposed)
        x = self.transformer(x_transposed, x_transposed).transpose(1, 2)
        x = self.dropout(x)
        x = self.cond1(x, style_embedding)
        x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        x = self.cond2(x, style_embedding)
        return x + features
