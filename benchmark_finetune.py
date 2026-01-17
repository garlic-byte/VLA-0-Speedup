import transformers
from robot.experiment import run_train
from robot.config.finetune_config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
)


def main():
    parser = transformers.HfArgumentParser(
        (ModelConfig, DataConfig, TrainConfig,)
    )
    model_config, data_config, train_config = parser.parse_args_into_dataclasses()

    # Modify data configuration
    data_config.dataset_path = "/home/wsj/Desktop/code/VLA/robot/datasets/libero_10"
    data_config.modality_id = "libero_panda"
    data_config.vlm_processor_path = "/home/wsj/Downloads/weights/qwen3-vl-2b"
    # Modify model configuration
    model_config.lora_rank = 128

    # Static configuration
    project_name = data_config.dataset_path.split("/")[-1]
    model_config.model_path = data_config.vlm_processor_path
    model_config.lora_alpha = model_config.lora_rank * 2
    train_config.global_batch_size = train_config.per_device_train_batch_size * train_config.num_gpus
    train_config.output_dir = f"./outputs/{project_name}/gpus_{train_config.num_gpus}_batch_size_{train_config.global_batch_size * train_config.gradient_accumulation_steps}_mask_ratio_{data_config.mask_ratio}"

    # Start training
    run_train(model_config, data_config, train_config)


if __name__ == "__main__":
    main()