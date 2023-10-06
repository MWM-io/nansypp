from typing import Dict, List, Tuple, Union

import torch
from torch import nn

from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature

DiscriminatorLossMetrics = Dict[str, Union[float, torch.Tensor]]


class LeastSquaresDiscriminatorLoss(nn.Module):
    """LS-GAN's L2-based GAN discriminator-loss.

    Introduced in Mao et al. ICCV'2017, _Least Squares Generative Adversarial Networks_.
    """

    @cuda_synchronized_timer(DO_PROFILING, prefix="L2DiscriminatorLoss")
    def forward(
        self, logits_fake: List[torch.Tensor], logits_real: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, DiscriminatorLossMetrics]:
        if len(logits_fake) > 0 and len(logits_real) > 0:
            d_fake = logits_fake[0].new_zeros(1)
            d_real = logits_real[0].new_zeros(1)
        else:
            d_fake = torch.zeros(1)
            d_real = torch.zeros(1)

        for logit_fake, logit_real in zip(logits_fake, logits_real):
            d_fake = d_fake + logit_fake.square().mean()
            d_real = d_real + (1 - logit_real).square().mean()

        loss: torch.Tensor = d_fake + d_real
        metrics: DiscriminatorLossMetrics = {
            "disc/loss": loss.detach(),
            "disc/d-real": d_real.mean().detach(),
            "disc/d-fake": d_fake.mean().detach(),
        }
        return loss, metrics

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
