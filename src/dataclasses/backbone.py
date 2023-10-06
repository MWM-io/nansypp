"""Typed hydra configurations for the whole project

Notes and gotchas:
    - Make sure to decorate each config class with @dataclass, even classes that are
    themselves derived from @dataclass-decorated configuration classes,
    otherwise Hydra fails at composing the configurations.

    - In the case of hierarchies of structured configurations such as:
    ```
    @dataclass
    class SubConfigSchema:
        foo: str = MISSING

    @dataclass
    class SubConfigA(SubConfigSchema):
        foo: str = 'A'
        
    @dataclass
    class SubConfigB(SubConfigSchema):
        foo: str = 'B'

    @dataclass
    class Config
        sub_config: SubConfig  # uninitialized
    ```
    
    Do not pass a default value of SubConfigA() or SubConfigB() to config.sub_config via a default class attribute,
    since this would prevent Hydra from overriding those:
    DON'T
    ```
    @dataclass
    class Config:
        sub_config: SubConfig = SubConfigA()
    ```
    Because trying to override config.sub_config with SubConfigB would yield:
    ```Merge error: SubConfigB is not a subclass of SubConfigA.```

    Default values should be provided via the Defaults list, either directly from the structured config,
    or through a yaml configuration file.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING, OmegaConf


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


cs = ConfigStore.instance()


# -------- DATA -------- #
@dataclass
class AdditiveBackgroundNoiseConfig:
    _target_: str = "src.data.preprocessing.augmentation.RandomBackgroundNoise"
    sample_rate: int = II("train_dataset.input_sample_rate")
    noise_dir: str = "/path/to/demand/decoded/audio/directory"
    min_snr_db: int = 0
    max_snr_db: int = 15
    noise_scale: float = MISSING
    augmentation_number: int = 1
    length_s: float = II("train_dataset.segment_length_s")


@dataclass
class SmallScaleAdditiveBackgroundNoiseConfig(AdditiveBackgroundNoiseConfig):
    noise_scale: float = 0.8


cs.store(
    group="datamodule/augmentation",
    name="nansypp_random_background_noise",
    node=AdditiveBackgroundNoiseConfig,
)

cs.store(
    group="datamodule/augmentation",
    name="small_scale_datamodule_random_background_noise",
    node=SmallScaleAdditiveBackgroundNoiseConfig,
)


@dataclass
class DatasetConfig:
    _target_: str = "src.data.dataset.backbone.BackboneDataset"
    segment_length_s: float = 1.0
    input_data_dirs: Optional[List[str]] = None
    input_file_list: Optional[str] = None
    input_sample_rate: int = MISSING
    output_data_dirs: Optional[List[str]] = None
    output_file_list: Optional[str] = None
    output_sample_rate: int = MISSING


@dataclass
class HifiTTSDatasetConfig(DatasetConfig):
    input_data_dirs: List[str] = field(
        default_factory=lambda: [
            "/path/to/hifitts/decoded/audio/directory",
        ]
    )
    input_sample_rate: int = 44100
    output_sample_rate: int = 44100


cs.store(group="train_dataset", name="base", node=DatasetConfig)
cs.store(
    group="train_dataset",
    name="hifitts",
    node=HifiTTSDatasetConfig,
)


@dataclass
class DataModuleConfig:
    _target_: str = "src.data.datamodule.backbone.BackboneDataModule"
    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True

    train_dataset: DatasetConfig = II("train_dataset")
    val_dataset: Optional[DatasetConfig] = II("val_dataset")
    val_split: Optional[float] = None
    val_seed: Optional[int] = None


cs.store(group="datamodule", name="base", node=DataModuleConfig)


# -------- LINGUISTIC-INFORMATION-PRESERVING PERTURBATOR -------- #
@dataclass
class PraatAugmentConfig:
    _target_: str = "src.networks.misc.praat.PraatAugment"
    sample_rate: int = II("train_dataset.input_sample_rate")
    pitch_steps: float = 0.01
    pitch_floor: float = 75
    pitch_ceil: float = 600


@dataclass
class ParametricEqualizerAugmentConfig:
    _target_: str = "src.networks.misc.peq.ParametricEqualizer"
    sample_rate: int = II("train_dataset.input_sample_rate")
    window_length: int = II("generator.mel_spectrogram_transform.window_length")


@dataclass
class InformationPerturbatorConfig:
    _target_: str = "src.networks.misc.perturbator.InformationPerturbator"
    formant_shift: float = 1.4
    pitch_shift: float = 2.0
    pitch_range: float = 1.5
    cutoff_lowpass: float = 60.0
    cutoff_highpass: float = 10000.0
    q_min: float = 2.0
    q_max: float = 5.0
    num_peaks: int = 8
    gain_range: float = 12.0

    stft_window_length: int = II("generator.mel_spectrogram_transform.window_length")
    stft_hop_length: int = II("generator.mel_spectrogram_transform.hop_length")

    praat_augment: PraatAugmentConfig = PraatAugmentConfig()
    parametric_equalizer: ParametricEqualizerAugmentConfig = (
        ParametricEqualizerAugmentConfig()
    )

    additive_noise: Optional[
        AdditiveBackgroundNoiseConfig
    ] = SmallScaleAdditiveBackgroundNoiseConfig()


@dataclass
class BypassedInformationPerturbatorConfig(InformationPerturbatorConfig):
    _target_: str = "torch.nn.Identity"


# -------- GENERATOR -------- #
@dataclass
class PretrainedWav2Vec2Config:
    name: str = MISSING
    sample_rate: int = MISSING
    """The sample rate at which the pre-trained wav2vec2 model was trained and operates.

    Make sure to provide the appropriate value here!

    Can be different from the general NANSY++ sample-rate, in which case
    inputs to the wav2vec2 encoder will be resampled.
    """


@dataclass
class PretrainedWav2Vec2XLSR53Config(PretrainedWav2Vec2Config):
    name: str = "facebook/wav2vec2-large-xlsr-53"
    sample_rate: int = 16000


cs.store(
    group="generator/linguistic_encoder/wav2vec2/pretrained_model",
    # group="pretrained_wav2vec2_model",
    name="large-xlsr-53",
    node=PretrainedWav2Vec2XLSR53Config,
)


@dataclass
class Wav2Vec2Config:
    """Operates at a sampling-frequency independent of the rest of the NANSY++ network"""

    _target_: str = "src.networks.backbone.encoders.wav2vec2.Wav2Vec2Wrapper"
    global_sample_rate: int = II("train_dataset.input_sample_rate")

    pretrained_model: PretrainedWav2Vec2Config = PretrainedWav2Vec2Config()

    layer_for_linguistic_features: int = MISSING
    """The Wav2Vec2 Transformer layer at which hidden linguistic features are extracted from.

    TODO: check the below comment from @revsic, this is not very clear
    NOTE(@revsic, legacy repository): One-base indexing required here because the
    hidden state at dimension 0 corresponds to the position-informed convolution features
    """
    trim_unused_layers: bool = True


@dataclass
class NansyppWav2Vec2Config(Wav2Vec2Config):
    layer_for_linguistic_features: int = 15  # cf. Figure 8.c in NANSY++ [choi2023]
    pretrained_model: PretrainedWav2Vec2Config = PretrainedWav2Vec2XLSR53Config()


cs.store(
    group="generator/linguistic_encoder/wav2vec2",
    name="nansypp-wav2vec2",
    node=NansyppWav2Vec2Config,
)


# all original STFT hyperparameters are UNKNOWN
@dataclass
class MelSTFTConfig:
    _target_: str = "src.networks.misc.transform.MelSpectrogram"

    hop_length: int = 256
    """Hop length, the number of frames between adjacent windows."""

    window_length: int = 1024
    """The length of the windows."""

    mel: int = 80
    """The size of the mel filterbanks."""

    fmin: float = 0
    """The minimum frequency."""

    fmax: Optional[float] = 8000
    """The maximum frequency. If None, uses half of the sample rate as default."""

    sample_rate: int = II("train_dataset.input_sample_rate")
    """The sample rate."""


@dataclass
class CQTConfig:
    _target_: str = "src.networks.misc.transform.ConstantQTransform"

    hop_length: int = 256
    """The number of samples between adjacent frame."""

    fmin: float = 32.7
    """The minimum frequency."""

    bins: int = 191
    """The number of output bins."""

    bins_per_octave: int = 24
    """The number of frequency bins per octave."""

    sample_rate: int = II("train_dataset.input_sample_rate")
    """The sampling rate."""


@dataclass
class FrameLevelSynthesizerConfig:
    _target_: str = (
        "src.networks.backbone.synthesizers.framelevel.FrameLevelSynthesizer"
    )

    in_channels: int = II("generator.linguistic_encoder.hidden_channels")
    """Size of the input channels."""

    kernels: int = 3
    """Size of the convolutional kernels."""

    dilations: List[int] = field(default_factory=lambda: [1, 3, 9, 27, 1, 3, 9, 27])
    """Dilation rates."""

    blocks: int = 2
    """Number of the 1x1 ConvGLU blocks after dilated ConvGLU."""

    leak: float = II("generator.leak")
    """Negative slope of the leaky ReLUs."""

    dropout_rate: float = II("generator.dropout_rate")
    """Dropout rates."""

    timbre_embedding_channels: int = II("generator.timbre_encoder.out_channels")
    """Size of the time-varying timbre embeddings."""


@dataclass
class LinguisticEncoderConfig:
    _target_: str = "src.networks.backbone.encoders.linguistic.LinguisticEncoder"
    wav2vec2: Wav2Vec2Config = Wav2Vec2Config()

    hidden_channels: int = 128
    """Size of the hidden channels."""

    preconv_blocks: int = 2
    """Number of pre-convolution blocks."""

    convglu_kernel_sizes: List[int] = field(
        default_factory=lambda: [3, 3, 3, 3, 3, 3, 3, 3, 1, 1]
    )  # 3 * 8 + [1] * 2
    """Size of the ConvGLU kernels."""

    leak: float = II("generator.leak")
    """Negative slope of leaky ReLUs."""

    dropout_rate: float = II("generator.dropout_rate")
    """Dropout rate."""


@dataclass
class PitchEncoderConfig:
    _target_: str = "src.networks.backbone.encoders.pitch.PitchEncoder"

    freq: int = 160
    """Number of frequency bins."""

    min_pitch: float = 50
    """The minimum predicted pitch."""

    max_pitch: float = 1000
    """The maximum predicted pitch."""

    prekernels: int = 7
    """Size of the first convolutional kernels."""

    kernels: int = 3
    """Size of the frequency-convolution kernels."""

    channels: int = 128
    """Size of the channels."""

    blocks: int = 2
    """Number of residual blocks."""

    gru_dim: int = 256  # NANSY++ value UNKNOWN
    """Size of the GRU hidden states."""

    hidden_channels: int = 256  # NANSY++ value UNKNOWN
    """Size of the hidden channels."""

    f0_bins: int = 64
    """Size of the output f0-bins."""

    f0_activation: str = "sigmoid"
    """F0 activation function."""


@dataclass
class SynthesizerConfig:
    _target_: str = "src.networks.backbone.synthesizers.synthesizer.Synthesizer"

    channels: int = 64
    kernels: int = 3
    dilation_rate: int = 2
    layers: int = 10
    cycles: int = 3
    input_sample_rate: int = II("train_dataset.input_sample_rate")
    output_sample_rate: int = II("train_dataset.output_sample_rate")
    scale: int = II("generator.cqt.hop_length")
    aux: int = II("generator.linguistic_encoder.hidden_channels")


@dataclass
class AttentiveStatisticsPoolingConfig:
    _target_: str = "src.networks.backbone.encoders.timbre.AttentiveStatisticsPooling"

    channels: int = II("generator.timbre_encoder.hidden_channels")
    bottleneck: int = II("generator.timbre_encoder.bottleneck")


@dataclass
class TimbreEncoderConfig:
    _target_: str = "src.networks.backbone.encoders.timbre.TimbreEncoder"

    in_channels: int = II("generator.mel_spectrogram_transform.mel")
    """Size of the input channels."""

    out_channels: int = 192
    """Size of the output embeddings."""

    channels: int = 512
    """Size of the major states."""

    prekernels: int = 5
    """Size of the convolutional kernels before feed to SERes2Block."""

    scale: int = 8
    """The number of the resolutions, for SERes2Block."""

    kernels: int = 3
    """Size of the convolutional kernels, for SERes2Block."""

    dilations: List[int] = field(default_factory=lambda: [2, 3, 4])
    """Dilation factors."""

    bottleneck: int = 128
    """Size of the bottleneck layers, both SERes2Block and AttentiveStatisticsPooling."""

    # NANSY++: 3072
    hidden_channels: int = 1536
    """Size of the hidden channels for attentive statistics pooling."""

    latent: int = 512
    """Size of the timbre latent query vectors."""

    timbre: int = 128
    """Size of the timbre tokens."""

    tokens: int = 50
    """The number of timbre tokens."""

    # UNKNOWN
    heads: int = 8
    """The number of attention heads, for timbre token block."""

    linguistic_encoder_hidden_channels: int = II(
        "generator.linguistic_encoder.hidden_channels"
    )
    """The size of the features returned by the linguistic encoder."""

    # UNKNOWN
    slerp: float = 0.5
    """Weight value for spherical interpolation."""

    mel_spectrogram_transform: MelSTFTConfig = II("generator.mel_spectrogram_transform")

    attentive_statistics_pooling: AttentiveStatisticsPoolingConfig = (
        AttentiveStatisticsPoolingConfig()
    )


def compute_cqt_center(cqt_bins: int, pitch_encoder_freq: int) -> int:
    return (cqt_bins - pitch_encoder_freq) // 2


OmegaConf.register_new_resolver("cqt_center", compute_cqt_center)


# -------- GENERATOR -------- #
@dataclass
class GeneratorConfig:
    _target_: str = "src.networks.backbone.generator.Generator"

    input_sample_rate: int = II("train_dataset.input_sample_rate")
    output_sample_rate: int = II("train_dataset.output_sample_rate")
    """Input and output sample-rate for the generator"""

    # UNKNOWN, default negative-slope of nn.LeakyReLU units
    leak: float = 0.01

    # UNKNOWN, default dropout rate of nn.Transformer
    dropout_rate: float = 0.1

    cqt_center: int = II(
        "cqt_center:${generator.cqt.bins},${generator.pitch_encoder.freq}"
    )

    cqt: CQTConfig = CQTConfig()
    mel_spectrogram_transform: MelSTFTConfig = MelSTFTConfig()
    frame_level_synthesizer: FrameLevelSynthesizerConfig = FrameLevelSynthesizerConfig()
    linguistic_encoder: LinguisticEncoderConfig = LinguisticEncoderConfig()
    pitch_encoder: PitchEncoderConfig = PitchEncoderConfig()
    synthesizer: SynthesizerConfig = SynthesizerConfig()
    timbre_encoder: TimbreEncoderConfig = TimbreEncoderConfig()
    information_perturbator: InformationPerturbatorConfig = MISSING


# -------- DISCRIMINATOR -------- #
@dataclass
class DiscriminatorConfig:
    _target_: str = MISSING


@dataclass
class MultiPeriodDiscriminatorConfig(DiscriminatorConfig):
    _target_: str = "src.networks.misc.mpd.MultiPeriodDiscriminator"

    periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    channels: List[int] = field(default_factory=lambda: [32, 128, 512, 1024])
    kernels: int = 5
    stride: int = 3
    postkernels: int = 3
    leak: float = 0.1


# -------- MODEL -------- #
@dataclass
class ModelConfig:
    _target_: str = "src.models.backbone.Backbone"
    input_sample_rate: float = II("train_dataset.input_sample_rate")
    output_sample_rate: float = II("train_dataset.output_sample_rate")
    accumulate_grad_batches: int = 1
    # gradient norm logging
    grad_norm_logging_interval_batches: int = 100
    # gradient clipping
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None


# -------- OPTIMIZERS -------- #
@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.Adam"

    lr: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class GeneratorOptimizerConfig(OptimizerConfig):
    pass


@dataclass
class DiscriminatorOptimizerConfig(OptimizerConfig):
    lr: float = 2e-4


# -------- GENERATOR LOSS -------- #
@dataclass
class ContrastiveLossConfig:
    _target_: str = "src.losses.backbone.generator.ContrastiveLoss"

    negative_samples_minimum_distance_to_positive: int = 5
    # UNKNOWN
    num_candidates: int = 15
    temperature: float = 0.1


@dataclass
class ReconstructionLossConfig:
    _target_: str = MISSING


@dataclass
class MelL1LossConfig(ReconstructionLossConfig):
    _target_: str = "src.losses.backbone.generator.MelL1Loss"
    mel_spectrogram_transform: MelSTFTConfig = II("generator.mel_spectrogram_transform")


@dataclass
class MultiScaleFFTLossConfig(ReconstructionLossConfig):
    _target_: str = "src.losses.backbone.generator.MultiScaleFFTLoss"

    # objective
    # 16khz sample-rate,
    # default window_sizes=[1920, 320, 80], hop_sizes=[640, 80, 40] in NSF
    window_sizes: List[int] = field(default_factory=lambda: [2048, 512, 128])
    hop_sizes: List[int] = field(default_factory=lambda: [512, 128, 32])


@dataclass
class PitchLossConfig:
    pass


@dataclass
class RelativePitchDifferenceLossConfig(PitchLossConfig):
    _target_: str = "src.losses.backbone.generator.RelativePitchDifferenceLoss"

    # pitch consistency
    cqt_center: int = II("generator.cqt_center")
    cqt_shift_min: int = -12
    cqt_shift_max: int = 12

    # huber norm
    # UNKNOWN
    sigma: float = 0.5

    pitch_freq: int = II("generator.pitch_encoder.freq")

    # delta: 1.0
    # (0.25 * sigma) as stated in Spice paper and issue: https://github.com/revsic/torch-nansypp/issues/1
    delta: float = 0.125  # 0.25 * 0.5
    pitch_encoder = II("generator.pitch_encoder")


cs.store(
    group="generator_loss/pitch_prediction_loss",
    name="relative_pitch_difference_loss",
    node=RelativePitchDifferenceLossConfig,
)


@dataclass
class PitchSupervisionLossConfig(PitchLossConfig):
    _target_: str = "src.losses.backbone.generator.PitchSupervisionLoss"
    sample_rate: int = II("train_dataset.input_sample_rate")
    cqt_hop_length: int = II("generator.cqt.hop_length")


cs.store(
    group="generator_loss/pitch_prediction_loss",
    name="pitch_supervision_loss",
    node=PitchSupervisionLossConfig,
)


@dataclass
class GeneratorLossConfig:
    _target_: str = "src.losses.backbone.generator.GeneratorLoss"

    input_sample_rate: int = II("train_dataset.input_sample_rate")
    output_sample_rate: int = II("train_dataset.output_sample_rate")

    cqt_bins_per_octave: int = II("generator.cqt.bins_per_octave")

    segment_length_s: float = II("train_dataset.segment_length_s")

    # content loss warmup
    linguistic_loss_start_weight: float = 1e-5
    linguistic_loss_end_weight: int = 10

    pitch_prediction_loss: PitchLossConfig = MISSING
    contrastive_loss: ContrastiveLossConfig = ContrastiveLossConfig()
    reconstruction_losses: List[ReconstructionLossConfig] = field(
        default_factory=lambda: [MelL1LossConfig(), MultiScaleFFTLossConfig()]
    )

    non_scalar_logging_steps: int = 5000
    vc_source_audio_paths: Dict[str, str] = field(
        default_factory=lambda: {"p225": "static/samples/vctk/p225_001.wav"}
    )
    vc_target_audio_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "p226": "static/samples/vctk/p226_002.wav",
            "p227": "static/samples/vctk/p227_003.wav",
            "p228": "static/samples/vctk/p228_004.wav",
        }
    )

    # required for logging plots
    cqt_hop_length: int = II("generator.cqt.hop_length")


# -------- LOSS DISCRIMINATOR -------- #
@dataclass
class DiscriminatorLossConfig:
    _target_: str = "src.losses.backbone.discriminator.LeastSquaresDiscriminatorLoss"


# -------- LOGGING -------- #
@dataclass
class LoggerConfig:
    _target_: str = "pytorch_lightning.loggers.tensorboard.TensorBoardLogger"
    name: str = "default"
    save_dir: str = "log/"
    default_hp_metric: bool = False


class LightiningTrainerPrecision(StringEnum):
    """
    Valid pytorch-lightining training precisions as of April 2023.

    See: https://github.com/Lightning-AI/lightning/pull/16783#issue-1587848352
    """

    mixed_16 = "16-mixed"
    mixed_bf16 = "bf16-mixed"
    full_32 = "32-true"
    double_64 = "64-true"


# -------- TRAIN -------- #
@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    default_root_dir: str = "../static"

    # maximum optimizer steps - put 2*N to aim for N training steps as there are 2 optimizers
    max_steps: int = 2 * 1000000
    val_check_interval: Optional[int] = None

    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None

    benchmark: Optional[bool] = False
    precision: LightiningTrainerPrecision = LightiningTrainerPrecision.full_32


@dataclass
class CriterionConfig:
    pass


@dataclass
class MetricConfig:
    pass


@dataclass
class CallbackConfig:
    _target_: str = MISSING
    logging_interval: str = MISSING


@dataclass
class LRMonitorCallbackConfig(CallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: str = "epoch"


@dataclass
class CallbacksConfig:
    lr_monitor: CallbackConfig = field(default_factory=LRMonitorCallbackConfig)


@dataclass
class ModelCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = "checkpoints/"
    # step is the optimizer step - divide by 2 to access training step
    filename: str = "opt-steps={step}"
    every_n_train_steps: int = 50000
    save_top_k: int = -1
    verbose: bool = False


@dataclass
class ResumeConfig:
    checkpoint_path: str = MISSING


@dataclass
class LightningProfilerConfig:
    _target_: str = MISSING
    filename: str = MISSING


@dataclass
class PassThroughLightningProfilerConfig(LightningProfilerConfig):
    _target_: str = "pytorch_lightning.profilers.PassThroughProfiler"
    filename: str = "no_profiler"


@dataclass
class SimpleLightningProfilerConfig(LightningProfilerConfig):
    _target_: str = "pytorch_lightning.profilers.SimpleProfiler"
    filename: str = "simple_profiler"


@dataclass
class AdvancedLightningProfilerConfig(LightningProfilerConfig):
    _target_: str = "pytorch_lightning.profilers.AdvancedProfiler"
    filename: str = "advanced_profiler"


@dataclass
class PytorchLightningProfilerConfig(LightningProfilerConfig):
    _target_: str = "pytorch_lightning.profilers.PyTorchProfiler"
    filename: str = "pytorch_profiler"
    with_stack = True


# Hydra-related configuration


class RunDirBase(StringEnum):
    RUNS_BACKBONE = "runs_backbone"
    RUNS_TTS = "runs_tts"
    DEBUG = "debug"


@dataclass
class Config:
    """
    NANSY++ Hydra baseline config.
    """

    model_name: str = "baseline"
    experiment: str = (
        f'{II("model_name")}-{II("dataset.name")}-{II("generator.sample_rate")}hz'
    )

    train_dataset: DatasetConfig = MISSING
    val_dataset: Optional[DatasetConfig] = None
    datamodule: DataModuleConfig = DataModuleConfig()
    perturbator: InformationPerturbatorConfig = InformationPerturbatorConfig()
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = MultiPeriodDiscriminatorConfig()
    generator_optimizer: GeneratorOptimizerConfig = GeneratorOptimizerConfig()
    discriminator_optimizer: DiscriminatorOptimizerConfig = (
        DiscriminatorOptimizerConfig()
    )
    generator_loss: GeneratorLossConfig = GeneratorLossConfig()
    discriminator_loss: DiscriminatorLossConfig = DiscriminatorLossConfig()
    model: ModelConfig = ModelConfig()
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()
    logger: LoggerConfig = LoggerConfig()
    trainer: TrainerConfig = TrainerConfig()

    profiler: LightningProfilerConfig = MISSING

    callbacks: Dict[str, CallbackConfig] = field(
        default_factory=lambda: {"lr_monitor": LRMonitorCallbackConfig()}
    )
    criteria: Dict[str, CriterionConfig] = field(default_factory=lambda: {})
    metrics: Dict[str, MetricConfig] = field(default_factory=lambda: {})

    seed: Optional[int] = None
    resume: Optional[ResumeConfig] = None

    verbose: bool = False

    # hydra-configurations
    run_dir_base: RunDirBase = RunDirBase.RUNS_BACKBONE


cs.store(name="nansypp_config", node=Config)

cs.store(group="datamodule", name="base", node=DataModuleConfig)

cs.store(name="nansypp_generator", node=GeneratorConfig)
cs.store(
    group="generator/information_perturbator",
    name="nansypp_information_perturbator",
    node=InformationPerturbatorConfig,
)
cs.store(
    group="generator/information_perturbator",
    name="bypassed_information_perturbator",
    node=BypassedInformationPerturbatorConfig,
)
cs.store(group="generator/cqt", name="nansypp_cqt", node=CQTConfig)
cs.store(
    group="generator/frame_level_synthesizer",
    name="nansypp_frame_level_synthesizer",
    node=FrameLevelSynthesizerConfig,
)
cs.store(
    group="generator/linguistic_encoder",
    name="nansypp_linguistic_encoder",
    node=LinguisticEncoderConfig,
)
cs.store(
    group="generator/loss",
    name="nansypp_generator_loss",
    node=GeneratorLossConfig,
)
cs.store(
    group="generator/mel_spectrogram_transform",
    name="nansypp_mel_spectrogram_transform",
    node=MelSTFTConfig,
)
cs.store(
    group="generator/optimizer",
    name="nansypp_generator_optimizer",
    node=GeneratorOptimizerConfig,
)
cs.store(
    group="generator/pitch_encoder",
    name="nansypp_pitch_encoder",
    node=PitchEncoderConfig,
)
cs.store(
    group="generator/timbre_encoder",
    name="nansypp_timbre_encoder",
    node=TimbreEncoderConfig,
)
cs.store(
    group="generator/wav2vec2_wrapper",
    name="nansypp_wav2vec2_wrapper",
    node=Wav2Vec2Config,
)

# discriminator
cs.store(
    name="nansypp_discriminator",
    node=DiscriminatorConfig,
)
cs.store(
    group="discriminator/loss",
    name="nansypp_discriminator_loss",
    node=DiscriminatorLossConfig,
)
cs.store(
    group="discriminator/optimizer",
    name="nansypp_discriminator_optimizer",
    node=DiscriminatorOptimizerConfig,
)

cs.store(group="logger", name="base_logger", node=LoggerConfig)

cs.store(group="profiler", name="disabled", node=PassThroughLightningProfilerConfig)
cs.store(group="profiler", name="simple", node=SimpleLightningProfilerConfig)
cs.store(group="profiler", name="advanced", node=AdvancedLightningProfilerConfig)
cs.store(group="profiler", name="pytorch", node=PytorchLightningProfilerConfig)
