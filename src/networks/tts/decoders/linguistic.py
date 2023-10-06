import torch
from torch import nn

from src.networks.misc.cln import ConditionalLayerNormalization
from src.networks.misc.convrelunorm import ConvReLUNorm
from src.networks.misc.transformer import TransformerBlock


class LinguisticDecoder(nn.Module):
    """Linguistic decoder."""

    def __init__(
        self,
        first_conv_in_channels: int,
        first_conv_out_channels: int,
        first_conv_kernel_size: int,
        first_conv_stride: int,
        dropout: float,
        n_transformer_blocks: int,
        transformer_nhead: int,
        transformer_num_encoder_layers: int,
        transformer_num_decoder_layers: int,
        transformer_dim_feedforward: int,
        cln_convs_middle_channels: int,
        cln_convs_out_channels: int,
        cln_convs_kernel_size: int,
        cln_convs_stride: int,
        style_dim: int,
        n_cln_convs: int,
    ):
        """Initializer.
        Args:
            First ConvReLUNorm layers params,
            Transformer Blocks params,
            Second ConvReLUNorm layers params,
            ConditionalLayerNormalization layers params,
            Conv1d params.
        """
        super().__init__()
        self.first_conv = ConvReLUNorm(
            first_conv_in_channels,
            first_conv_out_channels,
            first_conv_kernel_size,
            first_conv_stride,
            dropout,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    first_conv_out_channels,
                    style_dim,
                    transformer_nhead,
                    transformer_num_encoder_layers,
                    transformer_num_decoder_layers,
                    transformer_dim_feedforward,
                    dropout,
                )
                for _ in range(n_transformer_blocks)
            ]
        )
        self.cln_convs = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv": ConvReLUNorm(
                            first_conv_out_channels
                            if i == 0
                            else cln_convs_middle_channels
                            if i == n_cln_convs - 1
                            else cln_convs_middle_channels,
                            cln_convs_out_channels,
                            cln_convs_kernel_size,
                            cln_convs_stride,
                            dropout,
                        ),
                        "cln": ConditionalLayerNormalization(
                            style_dim,
                            cln_convs_out_channels
                            if i == n_cln_convs - 1
                            else cln_convs_middle_channels,
                        ),
                    }
                )
                for i in range(n_cln_convs)
            ]
        )
        self.conv1d = nn.Conv1d(cln_convs_out_channels, 128, 1, 1)

    def forward(
        self, phoneme_features: torch.Tensor, style_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            phoneme_features: [torch.float32, [B, 128, N]], upsampled phoneme features.
            style_embedding: [torch.float32, [B, 128, N]].
        Returns:
            linguistic_features: [torch.float32, [B, 128, N]].
        """
        embeddings = self.first_conv(phoneme_features)
        for transformer_block in self.transformer_blocks:
            embeddings = transformer_block(embeddings, style_embedding)
        for conv in self.cln_convs:
            embeddings = conv["conv"](embeddings)
            embeddings = conv["cln"](embeddings, style_embedding)
        linguistic_features: torch.Tensor = self.conv1d(embeddings)
        return linguistic_features
