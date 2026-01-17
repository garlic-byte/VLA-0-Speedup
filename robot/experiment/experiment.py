import logging
import os.path
import shutil

from robot.config.finetune_config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
)
from robot.data.dataset.dataset import SingleLerobotDataset, ShardCacheDataset
from robot.model.qwen3_vl.qwen3_vl import Qwen3VLA
from robot.utils import write_config_to_json

from transformers import set_seed, TrainingArguments
import torch
from transformers.trainer import Trainer
import transformers
import torch.distributed as dist


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    # Reduce some libraries
    # logging.getLogger("transformers").setLevel(logging.WARNING)
    # logging.getLogger("datasets").setLevel(logging.WARNING)

def logging_train_config(config: TrainConfig):
    """Configure logging for training parameters."""
    prefix = "[Train loaded]"
    param_format = f"{prefix: <16} {{: <35}}: {{}}"

    logging.info(f"----------------------------Train loaded----------------------------")
    logging.info(param_format.format("Learning rate", config.learning_rate))
    logging.info(param_format.format("Per device train batch size", config.per_device_train_batch_size))
    logging.info(param_format.format("Gradient accumulation steps", config.gradient_accumulation_steps))
    logging.info(param_format.format("Global batch size (all ranks)", config.global_batch_size))
    logging.info(param_format.format("Dataloader number of workers", config.dataloader_num_workers))
    logging.info(param_format.format("Output path of directory", config.output_dir))
    logging.info(f"--------------------------------------------------------------------")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def run_train(model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig):
    """Run experiment."""

    # Initialize global rank parameters
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        # only meaningful for torchrun, for ray it is always 0
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    setup_logging()
    set_seed(data_config.seed)
    config_output_dir = os.path.join(train_config.output_dir, "configs")

    if global_rank == 0:
        # Create config output directory
        if os.path.exists(config_output_dir):
            shutil.rmtree(config_output_dir)
        os.makedirs(config_output_dir)

        # Write config to local dir
        write_config_to_json(model_config, os.path.join(config_output_dir, "model_config.json"))
        write_config_to_json(data_config, os.path.join(config_output_dir, "data_config.json"))
        write_config_to_json(train_config, os.path.join(config_output_dir, "train_config.json"))

        logging_train_config(train_config)


    # Prepare the datasets and model, and logging main parameters
    model = Qwen3VLA(model_config)
    single_dataset = SingleLerobotDataset(data_config, config_output_dir)
    collator = single_dataset.collator
    dataset = ShardCacheDataset(single_dataset)

    # Configuration for training
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        max_steps=train_config.max_steps,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=1e-5,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        fp16=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=train_config.dataloader_num_workers,
        report_to="tensorboard",
        seed=data_config.seed,
        deepspeed=train_config.deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        eval_strategy="no",
        eval_steps=500,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        ignore_data_skip=True,
        accelerator_config={"split_batches": True},
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
    )

    # Start train and save model
    trainer.train()
    trainer.save_model()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
