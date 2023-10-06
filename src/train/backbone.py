import copy
from typing import Optional

import autoroot  # PYTHON_PATH-setup, do not remove # pylint: disable=unused-import
import hydra
import hydra.utils as hu
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.datamodule.backbone import BackboneDataModule
from src.dataclasses.backbone import Config
from src.losses.backbone.discriminator import LeastSquaresDiscriminatorLoss
from src.losses.backbone.generator import GeneratorLoss
from src.models.backbone import Backbone
from src.networks.backbone.generator import Generator
from src.networks.misc.mpd import MultiPeriodDiscriminator


@hydra.main(
    config_path="../../configs/backbone",
    config_name="baseline.yaml",
    version_base="1.3",
)
def main(cfg: Optional[Config] = None) -> None:
    if cfg is None:
        raise RuntimeError(
            "Did not receive a valid Hydra configuration, ",
            "did you mistakingly call the main function directly from another module?",
        )

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    generator: Generator = hu.instantiate(cfg.generator)

    compute_augmentations_in_dataloader = True
    if compute_augmentations_in_dataloader:
        information_perturbator = copy.deepcopy(generator.information_perturbator)
        cqt = copy.deepcopy(generator.cqt)
        mel_spec = copy.deepcopy(generator.mel_spectrogram_transform)
    else:
        mel_spec = cqt = information_perturbator = None

    datamodule: BackboneDataModule = hu.instantiate(
        cfg.datamodule,
        information_perturbator=information_perturbator,
        cqt=cqt,
        mel_spec=mel_spec,
    )
    discriminator: MultiPeriodDiscriminator = hu.instantiate(cfg.discriminator)
    generator_optimizer: torch.optim.Optimizer = hu.instantiate(
        cfg.generator_optimizer,
        filter(lambda p: p.requires_grad, generator.parameters()),
    )
    discriminator_optimizer: torch.optim.Optimizer = hu.instantiate(
        cfg.discriminator_optimizer,
        filter(lambda p: p.requires_grad, discriminator.parameters()),
    )
    if cfg.resume is not None and cfg.resume.checkpoint_path is not None:
        optimizer_step = torch.load(cfg.resume.checkpoint_path, map_location="cpu")[
            "global_step"
        ]
        initial_training_step = optimizer_step // 2
    else:
        initial_training_step = 0
    generator_loss: GeneratorLoss = hu.instantiate(
        cfg.generator_loss,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        initial_training_step=initial_training_step,
        max_training_step=cfg.trainer.max_steps // 2,
    )
    discriminator_loss: LeastSquaresDiscriminatorLoss = hu.instantiate(
        cfg.discriminator_loss
    )

    model: Backbone = hu.instantiate(
        cfg.model,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
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
