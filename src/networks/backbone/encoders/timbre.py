from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.networks.misc.transform import MelSpectrogram
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature


class Res2Block(nn.Module):
    """Multi-scale residual blocks."""

    def __init__(self, channels: int, scale: int, kernels: int, dilation: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            scale: the number of the blocks.
            kenels: size of the convolutional kernels.
            dilation: dilation factors.
        """
        super().__init__()
        assert channels % scale == 0, (
            f"size of the input channels(={channels})"
            f" should be factorized by scale(={scale})"
        )
        width = channels // scale
        self.scale = scale
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        width,
                        width,
                        kernels,
                        padding=(kernels - 1) * dilation // 2,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(width),
                )
                for _ in range(scale - 1)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1D tensor,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        # [B, W, T], (S - 1) x [B, W, T] where W = C // S
        straight, *xs = inputs.chunk(self.scale, dim=1)
        # [B, W, T]
        base = torch.zeros_like(straight)
        # S x [B, W, T]
        outs = [straight]
        for x, conv in zip(xs, self.convs):
            # [B, W, T], increasing receptive field progressively
            base = conv(x + base)
            outs.append(base)
        # [B, C, T]
        return torch.cat(outs, dim=1)


class SERes2Block(nn.Module):
    """Multiscale residual block with Squeeze-Excitation modules."""

    def __init__(
        self, channels: int, scale: int, kernels: int, dilation: int, bottleneck: int
    ):
        """Initializer.
        Args:
            channels: size of the input channels.
            scale: the number of the resolutions, for res2block.
            kernels: size of the convolutional kernels.
            dilation: dilation factor.
            bottleneck: size of the bottleneck layers for squeeze and excitation.
        """
        super().__init__()
        self.preblock = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.BatchNorm1d(channels)
        )

        self.res2block = Res2Block(channels, scale, kernels, dilation)

        self.postblock = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.BatchNorm1d(channels)
        )

        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        # [B, C, T]
        x = self.preblock(inputs)
        # [B, C, T], res2net, multi-scale architecture
        x = self.res2block(x)
        # [B, C, T]
        x = self.postblock(x)
        # [B, C], squeeze and excitation
        scale = self.excitation(x.mean(dim=-1))
        # [B, C, T]
        x = x * scale[..., None]
        # residual connection
        return x + inputs


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling."""

    def __init__(self, channels: int, bottleneck: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            bottleneck: size of the bottleneck.
        """
        super().__init__()
        # nonlinear=Tanh
        # ref: https://github.com/KrishnaDN/Attentive-Statistics-Pooling-for-Deep-Speaker-Embedding
        # ref: https://github.com/TaoRuijie/ECAPA-TDNN
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pooling with weighted statistics.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C x 2]], weighted statistics.
        """
        # [B, C, T]
        weights = self.attention(inputs)
        # [B, C]
        mean = torch.sum(weights * inputs, dim=-1)
        # var = torch.sum(weights * inputs**2, dim=-1) - mean**2
        var = torch.sum(weights * inputs**2, dim=-1) - mean**2
        # [B, C x 2], for numerical stability of square root
        return torch.cat([mean, (var + 1e-7).sqrt()], dim=-1)

class TTSAttentiveStatisticsPooling(AttentiveStatisticsPooling):
    """Fixed Attentive statistics pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pooling with weighted statistics.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C x 2]], weighted statistics.
        """
        # [B, C, T]
        weights = self.attention(inputs)
        # [B, C]
        mean = torch.sum(weights * inputs, dim=-1)
        # var = torch.sum(weights * inputs**2, dim=-1) - mean**2
        var = torch.sqrt((weights * (inputs - mean.unsqueeze(-1))).pow(2).sum(dim=-1))
        # [B, C x 2], for numerical stability of square root
        return torch.cat([mean, var], dim=-1)


class MultiheadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(
        self,
        keys: int,
        values: int,
        queries: int,
        out_channels: int,
        hidden_channels: int,
        heads: int,
    ):
        """Initializer.
        Args:
            keys, valeus, queries: size of the input channels.
            out_channels: size of the output channels.
            hidden_channels: size of the hidden channels.
            heads: the number of the attention heads.
        """
        super().__init__()
        assert (
            hidden_channels % heads == 0
        ), f"size of hidden_channels channels(={hidden_channels}) should be factorized by heads(={heads})"
        self.channels, self.heads = hidden_channels // heads, heads
        self.proj_key = nn.Conv1d(keys, hidden_channels, 1)
        self.proj_value = nn.Conv1d(values, hidden_channels, 1)
        self.proj_query = nn.Conv1d(queries, hidden_channels, 1)
        self.proj_out = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform the inputs.
        Args:
            keys: [torch.float32; [B, keys, S]], attention key.
            values: [torch.float32; [B, values, S]], attention value.
            queries: [torch.float32; [B, queries, T]], attention query.
            mask: [torch.float32; [B, S, T]], attention mask, 0 for paddings.
        Returns:
            [torch.float32; [B, out_channels, T]], transformed outputs.
        """
        # B, T
        bsize, _, querylen = queries.shape
        # S
        keylen = keys.shape[-1]
        assert keylen == values.shape[-1], "lengths of key and value are not matched"
        # [B, H, hidden_channels // H, S]
        keys = self.proj_key(keys).view(bsize, self.heads, -1, keylen)
        values = self.proj_value(values).view(bsize, self.heads, -1, keylen)
        # [B, H, hidden_channels // H, T]
        queries = self.proj_query(queries).view(bsize, self.heads, -1, querylen)
        # [B, H, S, T]
        score = torch.matmul(keys.transpose(2, 3), queries) * (self.channels**-0.5)
        if mask is not None:
            score.masked_fill_(~mask[:, None, :, :1].to(torch.bool), -np.inf)
        # [B, H, S, T]
        weights = torch.softmax(score, dim=2)
        # [B, out_channels, T]
        out = self.proj_out(torch.matmul(values, weights).view(bsize, -1, querylen))
        if mask is not None:
            out = out * mask[:, :1]
        return out


class TimbreEncoder(nn.Module):
    """ECAPA-TDNN: Emphasized Channel Attention,
    [1] Propagation and Aggregation in TDNN Based Speaker Verification,
        Desplanques et al., 2020, arXiv:2005.07143.
    [2] Res2Net: A New Multi-scale Backbone architecture,
        Gao et al., 2019, arXiv:1904.01169.
    [3] Squeeze-and-Excitation Networks, Hu et al., 2017, arXiv:1709.01507.
    [4] Attentive Statistics Pooling for Deep Speaker Embedding,
        Okabe et al., 2018, arXiv:1803.10963.
    """

    in_channels: int
    """Size of the input channels."""
    out_channels: int
    """Size of the output embeddings."""
    channels: int
    """Size of the major states."""
    prekernels: int
    """Size of the convolutional kernels before feed to SERes2Block."""
    scale: int
    """The number of the resolutions, for SERes2Block."""
    kernels: int
    """Size of the convolutional kernels, for SERes2Block."""
    dilations: List[int]
    """Dilation factors."""
    bottleneck: int
    """Size of the bottleneck layers, both SERes2Block and AttentiveStatisticsPooling."""
    # NANSY++: 3072
    hidden_channels: int
    """Size of the hidden channels for attentive statistics pooling."""
    latent: int
    """Size of the timbre latent query."""
    timbre: int
    """Size of the timbre tokens."""
    tokens: int
    """The number of the timbre tokens."""
    # unknown
    heads: int
    """The number of the attention heads, for timbre token block."""
    linguistic_encoder_hidden_channels: int
    """The size of the features returned by the linguistic encoder."""
    # unknown
    slerp: float
    """Weight value for spherical interpolation."""
    mel_spectrogram_transform: MelSpectrogram
    """Utility class for Mel Spectrogram transform."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        prekernels: int,
        scale: int,
        kernels: int,
        dilations: List[int],
        bottleneck: int,
        hidden_channels: int,
        latent: int,
        timbre: int,
        tokens: int,
        heads: int,
        linguistic_encoder_hidden_channels: int,
        slerp: float,
        mel_spectrogram_transform: MelSpectrogram,
        attentive_statistics_pooling: AttentiveStatisticsPooling,
    ):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output embeddings.
            channels: size of the major states.
            prekernels: size of the convolutional kernels before feed to SERes2Block.
            scale: the number of the resolutions, for SERes2Block.
            kernels: size of the convolutional kernels, for SERes2Block.
            dilations: dilation factors.
            bottleneck: size of the bottleneck layers,
                both SERes2Block and AttentiveStatisticsPooling.
            hidden_channels: size of the hidden channels for attentive statistics pooling.
            latent: size of the timbre latent query.
            timbre: size of the timbre tokens.
            tokens: the number of the timbre tokens.
            heads: the number of the attention heads, for timbre token block.
            linguistic_encoder_hidden_channels:
                The size of the features returned by the linguistic encoder.
            slerp: weight value for spherical interpolation.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.prekernels = prekernels
        self.scale = scale
        self.kernels = kernels
        self.dilations = dilations
        self.bottleneck = bottleneck
        self.hidden_channels = hidden_channels
        self.latent = latent
        self.timbre = timbre
        self.tokens = tokens
        self.heads = heads
        self.linguistic_encoder_hidden_channels = linguistic_encoder_hidden_channels
        # unknown `slerp`
        assert 0 <= slerp <= 1, f"value slerp(={slerp:.2f}) should be in range [0, 1]"
        self.slerp = slerp

        self.mel_spectrogram_transform = mel_spectrogram_transform

        contents = self.contents = linguistic_encoder_hidden_channels + out_channels + 3

        # channels=512, prekernels=5
        # ref:[1], Figure2 and Page3, "architecture with either 512 or 1024 channels"
        self.preblock = nn.Sequential(
            nn.Conv1d(in_channels, channels, prekernels, padding=prekernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )
        # scale=8, kernels=3, dilations=[2, 3, 4], bottleneck=128
        self.blocks = nn.ModuleList(
            [
                SERes2Block(channels, scale, kernels, dilation, bottleneck)
                for dilation in dilations
            ]
        )
        # hidden_channels=1536
        # TODO(@revsic): hidden_channels=3072 on NANSY++
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(len(dilations) * channels, hidden_channels, 1), nn.ReLU()
        )
        # multi-head attention for time-varying timbre
        # NANSY++, latent=512, tokens=50
        self.timbre_query = nn.Parameter(torch.randn(1, latent, tokens))
        # NANSY++, timbre=128
        # unknown `heads`
        self.pre_mha = MultiheadAttention(
            keys=hidden_channels,
            values=hidden_channels,
            queries=latent,
            out_channels=latent,
            hidden_channels=latent,
            heads=heads,
        )
        self.post_mha = MultiheadAttention(
            keys=hidden_channels,
            values=hidden_channels,
            queries=latent,
            out_channels=timbre,
            hidden_channels=latent,
            heads=heads,
        )
        # attentive pooling and additional projector
        # out_channels=192
        self.pool = nn.Sequential(
            attentive_statistics_pooling,
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
        )

        # time-varying timbre encoder
        self.timbre_key = nn.Parameter(torch.randn(1, timbre, tokens))
        self.sampler = MultiheadAttention(
            keys=timbre,
            values=timbre,
            queries=contents,
            out_channels=timbre,
            hidden_channels=latent,
            heads=heads,
        )
        self.proj = nn.Conv1d(timbre, out_channels, 1)

    @cuda_synchronized_timer(DO_PROFILING, prefix="TimbreEncoder")
    def forward(
        self, inputs: torch.Tensor, precomputed_mel: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the x-vectors from the input sequence.
        Args:
            inputs: [torch.float32; [B, in_channels, T]], input sequences,
        Returns:
            [torch.float32; [B, out_channels]], global x-vectors,
            [torch.float32; [B, timbre, tokens]], timbre token bank.
        """
        if precomputed_mel is None:
            mel_spectrogram = self.mel_spectrogram_transform(inputs)
        else:
            mel_spectrogram = precomputed_mel

        # [B, C, T]
        x = self.preblock(mel_spectrogram)
        # N x [B, C, T]
        xs = []
        for block in self.blocks:
            # [B, C, T]
            x = block(x)
            xs.append(x)
        # [B, H, T]
        mfa = self.conv1x1(torch.cat(xs, dim=1))
        # [B, O]
        global_ = F.normalize(self.pool(mfa), p=2, dim=-1)
        # B
        bsize, _ = global_.shape
        # [B, latent, tokens]
        query = self.timbre_query.repeat(bsize, 1, 1)
        # [B, latent, tokens]
        query = self.pre_mha(mfa, mfa, query) + query
        # [B, timbre, tokens]
        local = self.post_mha(mfa, mfa, query)
        # [B, out_channels], [B, timbre, tokens]
        return global_, local

    def sample_timbre(
        self,
        contents: torch.Tensor,
        global_: torch.Tensor,
        tokens: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Sample the timbre tokens w.r.t. the contents.
        Args:
            contents: [torch.float32; [B, contents, T]], content queries.
            global_: [torch.float32; [B, out_channels]], global x-vectors, L2-normalized.
            tokens: [torch.float32; [B, timbre, tokens]], timbre token bank.
            eps: small value for preventing train instability of arccos in slerp.
        Returns:
            [torch.float32; [B, out_channels, T]], time-varying timbre embeddings.
        """
        # [B, timbre, tokens]
        key = self.timbre_key.repeat(contents.shape[0], 1, 1)
        # [B, timbre, T]
        sampled = self.sampler(key, tokens, contents)
        # [B, out_channels, T]
        sampled = F.normalize(self.proj(sampled), p=2, dim=1)
        # [B, 1, T]
        theta = torch.matmul(global_[:, None], sampled).clamp(-1 + eps, 1 - eps).acos()
        # [B, 1, T], slerp
        # clamp the theta is not necessary since cos(theta) is already clampped
        return (
            torch.sin(self.slerp * theta) * sampled
            + torch.sin((1 - self.slerp) * theta) * global_[..., None]
        ) / theta.sin()

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
