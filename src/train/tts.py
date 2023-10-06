from typing import Optional

import autoroot
import hydra
import hydra.utils as hu
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.datamodule.tts import TextToSpeechDataModule
from src.dataclasses.tts import Config
from src.losses.tts import TextToSpeechLoss
from src.models.tts import TextToSpeechModel
from src.networks.tts.tts import TextToSpeechNetwork


@hydra.main(
    config_path="../../configs/tts", config_name="baseline.yaml", version_base="1.3"
)
def main(cfg: Optional[Config] = None) -> None:
    if cfg is None:
        raise RuntimeError(
            "Did not receive a valid Hydra configuration, ",
            "did you mistakingly call the main function directly from another module?",
        )

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)
    datamodule: TextToSpeechDataModule = hu.instantiate(cfg.datamodule)
    network: TextToSpeechNetwork = hu.instantiate(cfg.network)
    optimizer: torch.optim.Optimizer = hu.instantiate(
        cfg.optimizer, filter(lambda p: p.requires_grad, network.parameters())
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler = hu.instantiate(
        cfg.scheduler, optimizer
    )
    loss: TextToSpeechLoss = hu.instantiate(cfg.loss)
    model: TextToSpeechModel = hu.instantiate(
        cfg.model,
        network=network,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    model_checkpoint: ModelCheckpoint = hu.instantiate(cfg.model_checkpoint)
    callbacks = []
    logger = hu.instantiate(cfg.logger)
    for _, cb_cfg in cfg.callbacks.items():
        callbacks.append(hu.instantiate(cb_cfg))
    callbacks.append(model_checkpoint)

    trainer: Trainer = hu.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.resume.checkpoint_path if cfg.resume is not None else None,
    )


if __name__ == "__main__":
    main()
