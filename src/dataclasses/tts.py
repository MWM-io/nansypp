from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

from src.dataclasses.backbone import (
    CallbackConfig,
    LightiningTrainerPrecision,
    LoggerConfig,
    LRMonitorCallbackConfig,
    PretrainedWav2Vec2Config,
    PretrainedWav2Vec2XLSR53Config,
    ResumeConfig,
    RunDirBase,
    Wav2Vec2Config,
)

cs = ConfigStore.instance()


# -------- DATA -------- #


@dataclass
class CmuTextProcessorConfig:
    _target_: str = "src.data.text.text_processor.TextProcessor"
    sample_rate: int = II("datamodule.train_dataset.sample_rate")


@dataclass
class ValDatasetConfig:
    _target_: str = "src.data.dataset.tts_base.TTSBaseDataset"
    dataset_dir: str = MISSING
    descriptor_file: str = MISSING
    alignment_file: str = MISSING
    sample_rate: int = MISSING
    text_processor: CmuTextProcessorConfig = CmuTextProcessorConfig()


@dataclass
class TrainDatasetConfig(ValDatasetConfig):
    _target_: str = "src.data.dataset.tts_train.TTSTrainDataset"
    dataset_dir: str = II("data_path")
    text_processor: CmuTextProcessorConfig = CmuTextProcessorConfig()
    max_seconds_per_batch: int = 128
    features_lengths_file: str = "static/tts/features_lengths.json"


@dataclass
class DataModuleConfig:
    _target_: str = "src.data.datamodule.tts.TextToSpeechDataModule"
    train_dataset: TrainDatasetConfig = TrainDatasetConfig()
    val_dataset: Optional[ValDatasetConfig] = ValDatasetConfig()
    batch_size: int = 1
    pin_memory: bool = True
    num_workers: int = 16
    num_val_workers: int = 0
    shuffle: bool = False


cs.store(group="datamodule", name="base", node=DataModuleConfig)


# -------- NETWORK -------- #
@dataclass
class PhonemeEncoderConfig:
    _target_: str = "src.networks.tts.encoders.phoneme.PhonemeEncoder"
    conv_in_channels: int = 1024
    conv_out_channels: int = 1024
    conv_kernel_size: int = 5
    conv_stride: int = 1
    dropout: float = II("dropout")
    n_convs: int = 3
    conditionning_channels: int = II("network.style_encoder.out_linear")
    nhead: int = 2
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 1024
    n_transformers: int = 3
    negative_slope: float = 0.01
    out_linear: int = 128
    num_labels: int = 122
    conv_middle_channels: int = 1024


@dataclass
class NansyppTTSWav2Vec2Config(Wav2Vec2Config):
    layer_for_linguistic_features: int = 3  # cf. Figure 11.c in NANSY++ [choi2023]
    pretrained_model: PretrainedWav2Vec2Config = PretrainedWav2Vec2XLSR53Config()


cs.store(
    group="generator/linguistic_encoder/wav2vec2",
    name="nansypp-wav2vec2-tts",
    node=NansyppTTSWav2Vec2Config,
)


@dataclass
class StyleEncoderConfig:
    _target_: str = "src.networks.tts.encoders.style.StyleEncoder"
    conv_in_channels: int = 1024
    conv_out_channels: int = 1024
    conv_kernel_size: int = 5
    conv_stride: int = 2
    dropout: float = II("dropout")
    n_convs: int = 3
    bottleneck: int = 128
    wav2vec2: Wav2Vec2Config = NansyppTTSWav2Vec2Config(
        global_sample_rate=II("datamodule.train_dataset.sample_rate")
    )
    out_linear: int = 128
    conv_middle_channels: int = 1024


@dataclass
class DurationPredictorConfig:
    _target_: str = "src.networks.tts.decoders.duration.DurationPredictor"
    conv_in_channels: int = 128
    conv_out_channels: int = 128
    conv_kernel_size: int = 5
    conv_stride: int = 1
    style_dim: int = II("network.style_encoder.out_linear")
    dropout: float = II("dropout")
    n_convs: int = 2
    conv_middle_channels: int = 128


@dataclass
class LinguisticDecoderConfig:
    _target_: str = "src.networks.tts.decoders.linguistic.LinguisticDecoder"
    first_conv_in_channels: int = 128
    first_conv_out_channels: int = 128
    first_conv_kernel_size: int = 5
    first_conv_stride: int = 1
    dropout: float = II("dropout")
    n_transformer_blocks: int = 2
    transformer_nhead: int = 2
    transformer_num_encoder_layers: int = 2
    transformer_num_decoder_layers: int = 2
    transformer_dim_feedforward: int = 1024
    cln_convs_middle_channels: int = 128
    cln_convs_out_channels: int = 128
    cln_convs_kernel_size: int = 1
    cln_convs_stride: int = 1
    style_dim: int = II("network.style_encoder.out_linear")
    n_cln_convs: int = 3


@dataclass
class AmplitudeDecoderConfig:
    _target_: str = "src.networks.tts.decoders.amplitude.AmplitudeDecoder"
    conv_in_channels: int = 128
    conv_out_channels: int = 128
    conv_kernel_size: int = 7
    conv_stride: int = 1
    style_dim: int = II("network.style_encoder.out_linear")
    dropout: float = II("dropout")
    n_convs: int = 3
    conv_middle_channels: int = 128


@dataclass
class F0DecoderConfig:
    _target_: str = "src.networks.tts.decoders.f0.F0Decoder"
    conv_in_channels: int = 128
    conv_out_channels: int = 128
    conv_kernel_size: int = 7
    conv_stride: int = 1
    style_dim: int = II("network.style_encoder.out_linear")
    dropout: float = II("dropout")
    n_convs: int = 2
    gru_hidden_size: int = 64
    gru_num_layers: int = 2
    conv_middle_channels: int = 128


@dataclass
class PitchStatsConfig:
    _target_: str = "src.data.datamodule.tts.PitchStats"
    stats_dir: str = II("data_path")
    file: str = "pitch_stats.pt"


# -------- NETWORK -------- #
@dataclass
class NetworkConfig:
    _target_: str = "src.networks.tts.tts.TextToSpeechNetwork"
    phoneme_encoder: PhonemeEncoderConfig = PhonemeEncoderConfig()
    style_encoder: StyleEncoderConfig = StyleEncoderConfig()
    duration_predictor: DurationPredictorConfig = DurationPredictorConfig()
    linguistic_decoder: LinguisticDecoderConfig = LinguisticDecoderConfig()
    amplitude_decoder: AmplitudeDecoderConfig = AmplitudeDecoderConfig()
    f0_decoder: F0DecoderConfig = F0DecoderConfig()
    pitch_stats: PitchStatsConfig = PitchStatsConfig()
    backbone_info_file: str = field(
        default_factory=lambda: II("data_path") + "backbone_info.json"
    )
    sample_rate: int = II("datamodule.train_dataset.sample_rate")
    text_processor: CmuTextProcessorConfig = CmuTextProcessorConfig()


# -------- OPTIMIZERS -------- #
@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.Adam"

    lr: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = (1e-2) ** (1 / 20)


# -------- LOSS -------- #
@dataclass
class LossConfig:
    _target_: str = "src.losses.tts.TextToSpeechLoss"
    pitch_weight: float = 1
    duration_weight: float = 1
    linguistic_weight: float = 1
    linguistic_multiplicator: float = 100


# -------- MODEL -------- #
@dataclass
class ModelConfig:
    _target_: str = "src.models.tts.TextToSpeechModel"
    grad_norm_logging_interval_batches: int = 100
    cer_texts_file: Optional[str] = "static/tts/generation_test.json"


# -------- TRAIN -------- #
@dataclass
class ModelCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = "checkpoints/"
    filename: str = "steps={step}"
    every_n_train_steps: int = 1000
    save_top_k: int = -1
    verbose: bool = False


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    default_root_dir: str = "../static"

    val_check_interval: int = 1000
    max_steps: int = 200000

    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None

    benchmark: Optional[bool] = False
    precision: LightiningTrainerPrecision = LightiningTrainerPrecision.full_32

    # gradient clipping
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None


@dataclass
class Config:
    data_path: str
    model_name: str = "tts_baseline"
    experiment: str = "test"

    seed: Optional[int] = None
    datamodule: DataModuleConfig = DataModuleConfig()

    network: NetworkConfig = NetworkConfig()
    loss: LossConfig = LossConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: Optional[SchedulerConfig] = SchedulerConfig()

    model: ModelConfig = ModelConfig()
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()
    logger: LoggerConfig = LoggerConfig()
    callbacks: Dict[str, CallbackConfig] = field(
        default_factory=lambda: {"lr_monitor": LRMonitorCallbackConfig()}
    )
    trainer: TrainerConfig = TrainerConfig()
    resume: Optional[ResumeConfig] = None
    run_dir_base: RunDirBase = RunDirBase.RUNS_TTS
    dropout: float = 0.0


cs.store(name="nansy_tts_config", node=Config)
