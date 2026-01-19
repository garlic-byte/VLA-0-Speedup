import os
from dataclasses import dataclass

import numpy as np
import tyro
from robot.policy import BaseRobotPolicy, PolicyServer, SimRobotPolicy
from robot.utils import get_all_config_path, load_config


@dataclass
class ServerConfig:
    """Configuration class for the server."""
    model_path: str = "/home/wsj/Desktop/code/VLA/robot/outputs/libero_10/checkpoint-22000"
    """The path of the trained model."""

    host: str = "localhost"
    """The host of the server."""

    port: int = 8000
    """The port of the server."""

    device: str = "cuda"
    """The device of the model."""

def main(config: ServerConfig):

    # Load config from trained file path
    config_dir = get_all_config_path(config.model_path)
    path_data_config = os.path.join(config_dir, "data_config.json")
    path_model_config = os.path.join(config_dir, "model_config.json")
    path_transformer_config = os.path.join(config_dir, "transformer_config.json")

    data_config = load_config(path_data_config)
    model_config = load_config(path_model_config)
    transformer_config = load_config(path_transformer_config)

    print("Starting Robot inference server...")
    print(f"  Embodiment tag: ",data_config["modality_id"])
    print(f"  Resize image: ",data_config["image_resize"])
    print(f"  Fraction crop: ",data_config["crop_fraction"])
    print(f"  Trained model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    policy = SimRobotPolicy(
        model_config=model_config,
        transformer_config=transformer_config,
        model_path=config.model_path,
        device=config.device,
    )

    # batch_size = 2
    # obs = {
    #     "video.image": np.zeros((batch_size, 1, 480, 640, 3), dtype=np.uint8),
    #     "video.wrist_image": np.zeros((batch_size, 1, 480, 640, 3), dtype=np.uint8),
    #     "annotation.human.action.task_description": ("put the white mug on the left plate and put the yellow and white mug on the right plate", ) * batch_size,
    # }
    # action = policy.get_action(obs)
    #
    # print("  Action: ", action)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
