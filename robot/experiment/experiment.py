from robot.config.finetune_config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
)
# from robot.data.dataset.dataset import SingleLerobotDataset, ShardCacheDataset
from robot.data.dataset.mutiple_datasets import MultipleShardDataset
from robot.data.transformer import DatasetCollator
from robot.model.qwen3_vl.qwen3_vl import Qwen3VLA

from transformers import set_seed, TrainingArguments
import torch
from transformers.trainer import Trainer
import transformers

from robot.utils import setup_logging, logging_train_config


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
    setup_logging()
    set_seed(data_config.seed)
    logging_train_config(train_config)

    # Prepare the datasets and model, and logging main parameters
    model = Qwen3VLA(model_config)
    collator = DatasetCollator(data_config)
    # dataset = ShardCacheDataset(data_config)
    dataset = MultipleShardDataset(data_config)

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
