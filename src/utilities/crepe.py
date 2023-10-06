from typing import Optional

import torch
import torchcrepe


@torch.no_grad()
def crepe_pitch_estimation(
    audio: torch.Tensor,
    sample_rate: int,
    hop_len: int,
    device: torch.device,
    model: str = "tiny",
    fmin: int = 50,
    fmax: int = 550,
    batch_size: int = 2048,
) -> Optional[torch.Tensor]:
    try:
        result = torchcrepe.predict(
            audio,
            sample_rate,
            hop_len,
            fmin,
            fmax,
            model,
            batch_size=batch_size,
            device=device,
        )
    except:
        print("Crepe crashed.")
        result = None
    return result
