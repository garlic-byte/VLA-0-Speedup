
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from transformers import PretrainedConfig

@dataclass
class DataConfig:
    """
    Configuration for dataset. It contains all parameters of dataset.
    """
    dataset_path: str = None
    """Path to the dataset root directory trajectory."""

    dataset_weights: tuple[float, ...] | None = None
    """Weight of the dataset for training."""

    modality_id: str = ""
    """Define the modality configuration for finetune."""

    seed: int = 64
    """Seed for the random number generator."""

    image_resize: tuple[int, int] = (224, 224)
    """Size of the resized image."""

    crop_fraction: float = 0.95
    """Fraction of image will be maintained."""

    color_jitter: bool = True
    """Whether to apply color jitter, default brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08,"""

    video_backend: str = "torchcodec"
    """Type of video decord: 'decord', 'torchvision_av', 'torchcodec'."""

    mask_ratio: float = 0.2
    """Ratio of mask action according the VLA-0 model."""

    is_train: bool = True
    """Whether the dataset is used for training."""

    vlm_processor_path: str = ""
    """Path to the qwen3-VL processor directory trajectory."""

    config_output_dir: str = ""
    """Path to the output directory for finetune configuration."""


@dataclass
class ModelConfig(PretrainedConfig):
    """
    Configuration for model. It contains all parameters of model.
    """
    model_path: str = ""
    """Path to the model root directory trajectory."""

    dtype: str = "bfloat16"
    """Precision for training (float32, bfloat16, fp16)."""

    tune_llm: bool = True
    """If True, fine-tune the language action model."""

    tune_visual: bool = True
    """If True, fine-tune the visual model."""

    lora_rank: int = -1
    """Rank of Lora for loaded vision-language action model. If the value of rank is set to -1, Lora will not be used for fine-tuning."""

    lora_alpha: int = 16
    """Alpha for Lora for loaded vision-language action model."""

    lora_dropout: float = 0.05
    """Dropout for Lora for loaded vision-language action model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class TrainConfig:
    """
    Configuration for training. It contains all parameters of training.
    """
    per_device_train_batch_size: int = 2
    """Batch size of train just for one gpu."""

    dataloader_num_workers: int = 4
    """The number of data loader workers to use for training."""

    gradient_accumulation_steps: int = 4
    """The number of gradient accumulation steps to use for training."""

    output_dir: str = ""
    """The output directory for fine-tuning across all gpus."""

    save_steps: int = 1000
    """The number of steps to save the model checkpoint."""

    num_gpus: int = 1
    """The number of GPUs to use for fine-tuning."""

    max_steps: int = 30000
    """Max steps for fine-tuning."""

    eval_batch_size: int = 1
    """Batch size of eval just for one gpu."""

    global_batch_size: int = 1
    """Batch size of global just for all gpus."""

    learning_rate: float = 1e-4
    """Rate of learning for fine-tuning."""

    logging_steps: int = 10
    """Step for logging parameters such as steps, loss, current learning rate."""

    save_total_limit: int = 5
    """Max number of saving model into local directory."""

    deepspeed_config: str = ""
    """DeepSpeed configuration file path."""

    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    """Report of type for fine-tuning."""



@dataclass
class FinetuneConfig:
    """
    Configuration for finetune a Vision-Language-Action (VLA) model.

    This dataclass defines all hyperparameters of the finetune model.
    It contains the following fields: model tuning options, data augmentation,
    and training hyperparameters.
    """
    model: ModelConfig = field(default_factory=ModelConfig)

    data: DataConfig = field(default_factory=DataConfig)

    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        pass
