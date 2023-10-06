import torch
from torch import nn

from src.networks.misc.convrelunorm import ConvReLUNorm
from src.networks.misc.transformer import TransformerBlock


class PhonemeEncoder(nn.Module):
    """
    Phoneme encoder.
    """

    def __init__(
        self,
        num_labels: int,
        conv_in_channels: int,
        conv_middle_channels: int,
        conv_out_channels: int,
        conv_kernel_size: int,
        conv_stride: int,
        dropout: float,
        n_convs: int,
        conditionning_channels: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        n_transformers: int,
        negative_slope: float,
        out_linear: int,
    ):
        """Initializer.
        Args:
            num_labels: size of phonemtic dictionary,
            ConvReLUNorm layers params,
            Transformer Blocks params,
            Linear layer params,
            LeakyReLU params.
        """
        super().__init__()
        # Lookup embedding table
        self.lookup_embedder = nn.Embedding(num_labels, conv_in_channels)
        # 3 ConvReLUNorm
        self.convs = nn.ModuleList(
            [
                ConvReLUNorm(
                    conv_in_channels if i == 0 else conv_middle_channels,
                    conv_out_channels if i == n_convs - 1 else conv_middle_channels,
                    conv_kernel_size,
                    conv_stride,
                    dropout,
                )
                for i in range(n_convs)
            ]
        )
        # 3 transformer blocks
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    conv_out_channels,
                    conditionning_channels,
                    nhead,
                    num_encoder_layers,
                    num_decoder_layers,
                    dim_feedforward,
                    dropout,
                )
                for _ in range(n_transformers)
            ]
        )
        # Linear with LeakyReLU
        self.linear = nn.Linear(in_features=conv_out_channels, out_features=out_linear)
        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, phoneme: torch.Tensor, style_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phoneme: [torch.float32; [B, 1, Ntext]]
            style_embs: [torch.float32; [B, Nstyle, 1]]
        Returns:
            phoneme_features: [torch.float32; [B, 128, Ntext]]
        """
        # [B, C, N]
        embedding = self.lookup_embedder(phoneme).transpose(1, 2)
        for layer in self.convs:
            embedding = layer(embedding)
        for transformer in self.transformers:
            embedding = transformer(embedding, style_embs)
        linear_out: torch.Tensor = self.linear(embedding.transpose(1, 2)).transpose(
            1, 2
        )
        result = self.leakyrelu(linear_out)
        return result
