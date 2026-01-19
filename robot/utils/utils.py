import json
import logging
import os
import shutil
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field, asdict, is_dataclass
from robot.config.data.modality_config import ModalityConfig
import torch.distributed as dist
import torch

from robot.config.finetune_config import TrainConfig

GLOBAL_RANK = 0

def serialize_for_json(obj):
    """
    Recursively process all non-JSON-serializable objects,
    automatically converting them to serializable basic types/strings.

    Supported conversions:
    - Numpy types (dtype/ndarray/int/float)
    - Path objects
    - torch.device objects
    - Enum types
    - All unsupported types are converted to string format to eliminate serialization errors!
    """
    # Handle numpy dtype types (root cause of your current error)
    if isinstance(obj, np.dtype):
        return str(obj)
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle all numpy numeric types (np.int64/np.float32, etc.) → convert to native Python types
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.integer):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floats to Python float
        return float(obj)
    elif isinstance(obj, np.bool_):
        # Convert numpy bool to Python bool
        return bool(obj)
    # Handle dictionaries with recursive traversal (supports nested dictionaries)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    # Handle lists/tuples with recursive traversal
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(i) for i in obj]
    elif isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, ModalityConfig):
        return serialize_for_json(asdict(obj))
    # Return native JSON-supported types unchanged
    else:
        return str(obj)

def load_config(config_path):
    """Load config from config_path."""
    assert os.path.exists(config_path), (f"Config file {config_path} does not exist")
    with open(config_path) as f:
        config = json.load(f)
    return config

def get_all_config_path(path):
    """Get the directory of all config files"""
    config_dir_name = "configs"
    config_dir_path = os.path.join(path, config_dir_name)
    if not os.path.exists(config_dir_path):
        parent_path = os.path.dirname(path)
        config_dir_path = os.path.join(parent_path, config_dir_name)
    assert os.path.exists(config_dir_path), f"{config_dir_path} does not exist"
    return config_dir_path


def initialize_dist():
    global GLOBAL_RANK
    if dist.is_initialized():
        GLOBAL_RANK = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        # only meaningful for torchrun, for ray it is always 0
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        GLOBAL_RANK = dist.get_rank()


def write_config_to_json(config: any, path: str):
    """Save configuration to a json file."""
    # Step 1. Convert configuration to dictory
    if is_dataclass(config): config = asdict(config)
    assert isinstance(config, dict), f"{type(config)} is not a dataclass object."

    # Step 2. Universal serialization to handle all non-standard types → core fix for the error
    config = serialize_for_json(config)

    # Step 3. Write to the file of json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def write_configs_to_jsons(config_output_dir, **kwargs):
    """Save configurations to json files."""
    global GLOBAL_RANK
    if GLOBAL_RANK != 0:
        return

    # Create dictory from config_output_dir
    os.makedirs(config_output_dir, exist_ok=True)

    # Write configuration to json file
    for config_name, config in kwargs.items():
        write_config_to_json(config, os.path.join(config_output_dir, f"{config_name}.json"))


def logging_train_config(config: TrainConfig):
    """Configure logging for training parameters."""
    global GLOBAL_RANK
    if GLOBAL_RANK != 0:
        return

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


def logging_model_load(
    model_path: str,
    finetune_modules: list = None,
    total_params: int = None,
    total_trainable_params: int = None,
):
    global GLOBAL_RANK
    if GLOBAL_RANK != 0:
        return
    logging.info(f"----------------------------Model loaded----------------------------")
    logging.info(f"[Model loaded] Path of loaded model: {model_path}")
    logging.info(f"[Model loaded] Using partly parameters for training which contains: {finetune_modules} modules")
    logging.info(f"[Model loaded] Total params: {total_params:,}")
    logging.info(f"[Model loaded] Total trainable params: {total_trainable_params:,}, training radio: {total_trainable_params / total_params * 100:.2f}%")


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