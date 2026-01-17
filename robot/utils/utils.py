import json
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field, asdict, is_dataclass
from robot.config.data.modality_config import ModalityConfig


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


def write_config_to_json(config: any, path: str):
    """Save configurations to a json file."""
    # Step 1. Convert configuration to dictory
    if is_dataclass(config): config = asdict(config)
    assert isinstance(config, dict), f"{type(config)} is not a dataclass object."

    # Step 2. Universal serialization to handle all non-standard types → core fix for the error
    config = serialize_for_json(config)

    # Step 3. Write to the file of json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
