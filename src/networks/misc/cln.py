import torch
from torch import nn


class ConditionalLayerNormalization(nn.Module):
    """Conditional Layer Normalization / Style-Adaptive Layer Normalization
    [1]: Meta-StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation,
         Min et al., 2021, arxiv.2106.03153
    """

    def __init__(self, style_dim: int, vector_dim: int):
        """Initializer.
        Args:
            style_dim:
            vector_dim:
        """
        super().__init__()

        self.linear = nn.Linear(in_features=style_dim, out_features=vector_dim)

    def forward(
        self, vector: torch.Tensor, style_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vector: [B, C, N]
            style_embedding: [B, C_style, 1]
        """
        features = self.linear(style_embedding.squeeze(2))
        mean, std = features.mean(), features.std()
        normalized_vector: torch.Tensor = (vector - mean) / std
        return normalized_vector
