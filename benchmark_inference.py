import argparse
import json
import os

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

from robot.data.dataset.dataset import SingleLerobotDataset
from robot.policy import BaseRobotPolicy

from robot.config.finetune_config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
)
from robot.utils import get_all_config_path, load_config


def plot_trajectory(
    modality_info: dict,
    save_plot_path: str = None,
):
    if save_plot_path is not None:
        matplotlib.use("Agg")

    gt_action_across_time = modality_info["gt_action_across_time"]
    pred_action_across_time = modality_info["pred_action_across_time"]
    modality_keys = modality_info["modality_keys"]
    trajectory_index = modality_info["trajectory_index"]
    mse = modality_info["mse"]
    action_horizon = modality_info["action_horizon"]

    step, action_dim = gt_action_across_time.shape
    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {trajectory_index}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {mse:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for j in range(0, step, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def calc_mse_for_single_trajectory(
    dataset: SingleLerobotDataset,
    trajectory_index: int,
    policy: BaseRobotPolicy,
    modality_keys: tuple,
    plot: bool = True,
    save_plot_path: str = None,
):
    gt_action_across_time = []
    pred_action_across_time = []
    decode_actions = dataset.decode_actions
    action_horizon = dataset.action_horizon
    single_trajectory = dataset[trajectory_index]
    single_trajectory_len = len(single_trajectory)

    # In each single timesteps, concatenate complete action values of each modality keys
    cur_index = 0
    inference_time = []
    while cur_index < single_trajectory_len:
        step_data = single_trajectory[cur_index]
        start_time = time.perf_counter()
        predict_text_action = policy.get_action(step_data)
        end_time = time.perf_counter()
        inference_time.append(end_time - start_time)
        print("predict action time:", end_time - start_time, 's')

        if predict_text_action == "":
            cur_index += action_horizon
            continue
        predict_action = decode_actions(predict_text_action)
        target_action = step_data['action']

        # Action has horizon according modality
        for i in range(action_horizon):
            gt_action = []
            pred_action = []
            for key in modality_keys:
                assert (target_action[key][i].shape == predict_action[key][i].shape), \
                    (f"Get gt shape is not equal to pred shape: "
                     f"{target_action[key][i].shape} != {predict_action[key][i].shape}")

                gt_action.append(target_action[key][i])
                pred_action.append(predict_action[key][i])

            gt_action_across_time.append(np.concatenate(gt_action, axis=0))
            pred_action_across_time.append(np.concatenate(pred_action, axis=0))
        cur_index += action_horizon

    print("inference avg time:", (sum(inference_time) / len(inference_time)), 's')
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)


    if plot:
        info = {
            'gt_action_across_time': gt_action_across_time,
            'pred_action_across_time': pred_action_across_time,
            "modality_keys": modality_keys,
            "trajectory_index": trajectory_index,
            'mse': mse,
            "action_horizon": action_horizon
        }
        plot_trajectory(info, save_plot_path)
    return mse


def main():
    parser = argparse.ArgumentParser(description="Benchmark ROBOT inference results")
    parser.add_argument("--model_path", type=str, default="/home/wsj/Desktop/code/VLA/robot/outputs/libero_10/checkpoint-22000")
    parser.add_argument("--dataset_path", type=str, default="/home/wsj/Desktop/code/VLA/robot/datasets/libero_10", help="Path to dataset")
    parser.add_argument("--modality_id", type=str, default="libero_panda", help="Type to modality configuration")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--modality_keys", type=tuple, default=("arm", "gripper", ), help="Which modality to use to calculate loss")
    parser.add_argument("--plot", type=str, default=True, help="Whether to plot trajectories or not")
    parser.add_argument("--save_plot_path", type=str, default="outputs/plot/libero_10_gpus_1_batch_size_8_datasets_10_action.png", help="Path to save plot")

    args = parser.parse_args()

    # Load config from trained file path
    config_dir = get_all_config_path(args.model_path)
    path_data_config = os.path.join(config_dir, "data_config.json")
    path_model_config = os.path.join(config_dir, "model_config.json")

    data_config = load_config(path_data_config)
    model_config = load_config(path_model_config)


    print("Loading robot policy...")
    policy = BaseRobotPolicy(
        config=model_config,
        model_path=args.model_path,
        device=args.device,
    )

    print("Reading dataset...")
    data_config["is_train"] = False
    data_config = DataConfig(**data_config)

    dataset = SingleLerobotDataset(data_config)

    calc_mse_for_single_trajectory(
        dataset,
        trajectory_index=10,
        policy=policy,
        modality_keys=args.modality_keys,
        plot=args.plot,
        save_plot_path=args.save_plot_path,
    )

if __name__ == "__main__":
    main()