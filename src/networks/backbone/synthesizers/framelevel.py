from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer


class ConvGLU(nn.Module):
    """Dropout - Conv1d - GLU and conditional layer normalization."""

    def __init__(
        self,
        channels: int,
        kernels: int,
        dilations: int,
        dropout: float,
        conditionning_channels: Optional[int] = None,
    ):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dilations: dilation rate of the convolution.
            dropout: dropout rate.
            conditionning_channels: size of the condition channels, if provided.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels * 2,
                kernels,
                dilation=dilations,
                padding=(kernels - 1) * dilations // 2,
            ),
            nn.GLU(dim=1),
        )

        self.conditionning_channels = conditionning_channels
        if self.conditionning_channels is not None:
            self.cond = nn.Conv1d(self.conditionning_channels, channels * 2, 1)

    def forward(
        self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform the inputs with given conditions.
        Args:
            inputs: [torch.float32; [B, channels, T]], input channels.
            cond: [torch.float32; [B, cond, T]], if provided.
        Returns:
            [torch.float32; [B, channels, T]], transformed.
        """
        # [B, channels, T]
        x = inputs + self.conv(inputs)
        if cond is not None:
            assert self.cond is not None, "condition module does not exists"
            # [B, channels, T]
            x = F.instance_norm(x, use_input_stats=True)
            # [B, channels, T]
            weight, bias = self.cond(cond).chunk(2, dim=1)
            # [B, channels, T]
            x = x * weight + bias
        return x


class CondSequential(nn.Module):
    """Sequential pass with conditional inputs."""

    def __init__(self, modules: Sequence[nn.Module]):
        """Initializer.
        Args:
            modules: list of torch modules.
        """
        super().__init__()
        self.lists = nn.ModuleList(modules)

    # def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    def forward(
        self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pass the inputs to modules.
        Args:
            inputs: arbitary input tensors.
            args, kwargs: positional, keyword arguments.
        Returns:
            output tensor.
        """
        x = inputs
        for module in self.lists:
            # x = module.forward(x, *args, **kwargs)
            x = module.forward(x, cond)
        return x


class FrameLevelSynthesizer(nn.Module):
    """Frame-level synthesizer."""

    def __init__(
        self,
        in_channels: int,
        kernels: int,
        dilations: List[int],
        blocks: int,
        leak: float,
        dropout_rate: float,
        timbre_embedding_channels: Optional[int],
    ):
        """Initializer.
        Args:
            in_channels: The size of the input channels.
            kernels: The size of the convolutional kernels.
            dilations: The dilation rates.
            blocks: The number of the ConvGLU blocks after dilated ConvGLU.
            leak: The negative slope of the leaky relu.
            dropout_rate: The dropout rate.
            timbre_embedding_channels: The size of the time-varying timbre embeddings.
        """
        super().__init__()

        self.in_channels = in_channels
        self.timbre_embedding_channels = timbre_embedding_channels
        self.kernels = kernels
        self.dilations = dilations
        self.blocks = blocks
        self.leak = leak
        self.dropout_rate = dropout_rate

        # channels=1024
        # unknown `leak`, `dropout`
        self.preconv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(dropout_rate),
        )
        # kernels=3, dilations=[1, 3, 9, 27, 1, 3, 9, 27], blocks=2
        self.convglus = CondSequential(
            [
                ConvGLU(
                    in_channels,
                    kernels,
                    dilation,
                    dropout_rate,
                    conditionning_channels=timbre_embedding_channels,
                )
                for dilation in dilations
            ]
            + [
                ConvGLU(
                    in_channels,
                    1,
                    1,
                    dropout_rate,
                    conditionning_channels=timbre_embedding_channels,
                )
                for _ in range(blocks)
            ]
        )

        self.proj = nn.Conv1d(in_channels, in_channels, 1)

    @cuda_synchronized_timer(DO_PROFILING, prefix="FrameLevelSynthesizer")
    def forward(
        self, inputs: torch.Tensor, timbre: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Synthesize in frame-level.
        Args:
            inputs [torch.float32; [B, channels, T]]:
                Input features.
            timbre [torch.float32; [B, embed, T]], Optional:
                Time-varying timbre embeddings.
        Returns;
            [torch.float32; [B, channels, T]], outputs.
        """
        # [B, channels, T]
        x: torch.Tensor = self.preconv(inputs)
        # [B, channels, T]
        x = self.convglus(x, timbre)
        # [B, channels, T]
        return self.proj(x)
