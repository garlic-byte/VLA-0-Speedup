import os
import shutil
import transformers
from robot.experiment import run_train
from robot.config.finetune_config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
)
from robot.experiment.experiment import logging_train_config

from robot.utils import write_configs_to_jsons


def main():
    parser = transformers.HfArgumentParser(
        (ModelConfig, DataConfig, TrainConfig,)
    )
    model_config, data_config, train_config = parser.parse_args_into_dataclasses()
    # Modify data configuration
    data_config.dataset_path = "/home/wsj/Desktop/code/VLA/robot/datasets/libero_10,/home/wsj/Desktop/code/VLA/robot/datasets/libero_goal"
    data_config.modality_id = "libero_panda"
    model_config.model_path = "/home/wsj/Downloads/weights/qwen3-vl-2b"
    # Modify model configuration

    model_config.tune_llm = False
    model_config.tune_visual = False
    model_config.lora_rank = 128

    # Static configuration
    project_name = "libero"
    data_config.vlm_processor_path = model_config.model_path
    model_config.lora_alpha = model_config.lora_rank * 2
    train_config.global_batch_size = train_config.per_device_train_batch_size * train_config.num_gpus
    train_config.output_dir = f"./outputs/{project_name}/gpus_{train_config.num_gpus}_batch_size_{train_config.global_batch_size * train_config.gradient_accumulation_steps}_mask_ratio_{data_config.mask_ratio}"

    # Initialize global rank parameters
    data_config.config_output_dir = os.path.join(train_config.output_dir, "configs")

    write_configs_to_jsons(
        config_output_dir=data_config.config_output_dir,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
    )

    # Start training
    run_train(model_config, data_config, train_config)


if __name__ == "__main__":
    main()