import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, List, Any

from tensorboard.plugins.debugger_v2.debug_data_provider import source_file_run_tag_filter


def process_parquet_files(folder_path: str, output_name: str = "stats.json") -> None:
    """
    Process all parquet files in a folder, extract observation.state and action fields,
    and calculate statistics (mean, std, min, max, q01, q99).

    Args:
        folder_path: Path to folder containing parquet files
        output_name: Name to save the output JSON file
    """

    # Get all parquet files in the folder
    output_path = os.path.join(folder_path, 'meta/' + output_name)
    parquet_files = glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    # Initialize lists to store all data
    all_observation_states = []
    all_actions = []

    # Process each parquet file
    for file_idx, file_path in enumerate(parquet_files):
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)

            # Check if required columns exist
            if 'observation.state' not in df.columns or 'action' not in df.columns:
                print(f"Warning: Required columns not found in {file_path}")
                continue

            # Extract observation.state and action fields
            for _, row in df.iterrows():
                # Ensure data is in list format
                if isinstance(row['observation.state'], list):
                    all_observation_states.append(row['observation.state'])
                if isinstance(row['action'], list):
                    all_actions.append(row['action'])

            # Print progress every 10 files
            if (file_idx + 1) % 10 == 0:
                print(f"Processed {file_idx + 1}/{len(parquet_files)} files")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Check if we have valid data
    if not all_observation_states or not all_actions:
        print("No valid data found in the files")
        return

    # Convert lists to numpy arrays for statistical calculations
    observation_array = np.array(all_observation_states)
    action_array = np.array(all_actions)

    print(f"Total observation samples: {len(all_observation_states)}")
    print(f"Total action samples: {len(all_actions)}")
    print(f"Observation state shape: {observation_array.shape}")
    print(f"Action shape: {action_array.shape}")

    def calculate_statistics(data_array: np.ndarray, array_name: str) -> Dict[str, List[float]]:
        """
        Calculate statistics for a numpy array.

        Args:
            data_array: Input numpy array
            array_name: Name of the array for logging purposes

        Returns:
            Dictionary containing mean, std, min, max, q01, q99 for each dimension
        """
        if len(data_array) == 0:
            return {}

        stats = {}

        # Calculate statistics for each dimension
        stats['mean'] = np.mean(data_array, axis=0).tolist()
        stats['std'] = np.std(data_array, axis=0).tolist()
        stats['max'] = np.max(data_array, axis=0).tolist()
        stats['min'] = np.min(data_array, axis=0).tolist()

        # Calculate 1st and 99th percentiles
        stats['q01'] = np.percentile(data_array, 1, axis=0).tolist()
        stats['q99'] = np.percentile(data_array, 99, axis=0).tolist()

        # Print dimension information
        print(f"{array_name} - Number of dimensions: {data_array.shape[1]}")

        return stats

    # Calculate statistics for observation.state and action
    observation_stats = calculate_statistics(observation_array, "observation.state")
    action_stats = calculate_statistics(action_array, "action")

    # Create output dictionary
    output_dict = {
        "observation": {
            "state": observation_stats
        },
        "action": action_stats
    }

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    print(f"\nStatistics saved to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)

    for stat_name, stat_func in [("Mean", np.mean), ("Std", np.std),
                                 ("Min", np.min), ("Max", np.max)]:
        if len(observation_array) > 0:
            obs_stats = stat_func(observation_array, axis=0)
            print(f"Observation.state {stat_name}: [{obs_stats[0]:.6f}, ..., {obs_stats[-1]:.6f}]")

        if len(action_array) > 0:
            act_stats = stat_func(action_array, axis=0)
            print(f"Action {stat_name}: [{act_stats[0]:.6f}, ..., {act_stats[-1]:.6f}]")
        print()


def process_parquet_files_optimized(folder_path: str | Path, output_name: str = "statistics.json") -> None:
    """
    Optimized version for large datasets that might not fit in memory.
    Uses incremental calculation of statistics.

    Args:
        folder_path: Path to folder containing parquet files
        output_name: Name to save the output JSON file
    """

    # Get all parquet files
    folder_path = str(folder_path)
    output_path = os.path.join(folder_path, 'meta/' + output_name)
    parquet_files = glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return

    print(f"Found {len(parquet_files)} parquet files, starting calculation statistics.")

    # Initialize accumulators for incremental calculation
    obs_accumulator = {
        'sum': None,  # Sum of values
        'sum_sq': None,  # Sum of squared values
        'min': None,  # Minimum values
        'max': None,  # Maximum values
        'all_values': [],  # Store all values for percentile calculation
        'count': 0  # Number of samples
    }

    act_accumulator = {
        'sum': None,
        'sum_sq': None,
        'min': None,
        'max': None,
        'all_values': [],
        'count': 0
    }

    # Process each file
    for file_idx, file_path in enumerate(parquet_files):
        try:
            df = pd.read_parquet(file_path)

            # Check for required columns
            if 'observation.state' not in df.columns or 'action' not in df.columns:
                print(f"Warning: Required columns not found in {file_path}")
                continue

            # Extract data from current file
            observation_batch = []
            action_batch = []

            for _, row in df.iterrows():
                if isinstance(row['observation.state'], np.ndarray):
                    observation_batch.append(row['observation.state'])
                if isinstance(row['action'], np.ndarray):
                    action_batch.append(row['action'])

            if not observation_batch or not action_batch:
                continue

            # Convert to numpy arrays
            obs_array = np.array(observation_batch)
            act_array = np.array(action_batch)

            # Update observation accumulator
            if obs_accumulator['count'] == 0:
                # First batch
                obs_accumulator['sum'] = np.sum(obs_array, axis=0)
                obs_accumulator['sum_sq'] = np.sum(obs_array ** 2, axis=0)
                obs_accumulator['min'] = np.min(obs_array, axis=0)
                obs_accumulator['max'] = np.max(obs_array, axis=0)
                # Store all values for percentile calculation
                obs_accumulator['all_values'].extend(obs_array.tolist())
            else:
                # Update accumulators
                obs_accumulator['sum'] += np.sum(obs_array, axis=0)
                obs_accumulator['sum_sq'] += np.sum(obs_array ** 2, axis=0)
                obs_accumulator['min'] = np.minimum(obs_accumulator['min'],
                                                    np.min(obs_array, axis=0))
                obs_accumulator['max'] = np.maximum(obs_accumulator['max'],
                                                    np.max(obs_array, axis=0))
                obs_accumulator['all_values'].extend(obs_array.tolist())

            obs_accumulator['count'] += len(obs_array)

            # Update action accumulator
            if act_accumulator['count'] == 0:
                act_accumulator['sum'] = np.sum(act_array, axis=0)
                act_accumulator['sum_sq'] = np.sum(act_array ** 2, axis=0)
                act_accumulator['min'] = np.min(act_array, axis=0)
                act_accumulator['max'] = np.max(act_array, axis=0)
                act_accumulator['all_values'].extend(act_array.tolist())
            else:
                act_accumulator['sum'] += np.sum(act_array, axis=0)
                act_accumulator['sum_sq'] += np.sum(act_array ** 2, axis=0)
                act_accumulator['min'] = np.minimum(act_accumulator['min'],
                                                    np.min(act_array, axis=0))
                act_accumulator['max'] = np.maximum(act_accumulator['max'],
                                                    np.max(act_array, axis=0))
                act_accumulator['all_values'].extend(act_array.tolist())

            act_accumulator['count'] += len(act_array)

            # Print progress
            if (file_idx + 1) % 10 == 0:
                print(f"Processed {file_idx + 1}/{len(parquet_files)} files, "
                      f"Total samples: observation={obs_accumulator['count']}, "
                      f"action={act_accumulator['count']}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Check if we have data
    if obs_accumulator['count'] == 0 or act_accumulator['count'] == 0:
        print("No valid data found in the files")
        return

    def calculate_final_statistics(accumulator: Dict, array_name: str) -> Dict[str, List[float]]:
        """
        Calculate final statistics from accumulator.

        Args:
            accumulator: Dictionary containing accumulated statistics
            array_name: Name of the array for logging

        Returns:
            Dictionary with mean, std, min, max, q01, q99
        """
        if accumulator['count'] == 0:
            return {}

        stats = {}

        # Calculate mean
        stats['mean'] = (accumulator['sum'] / accumulator['count']).tolist()

        # Calculate standard deviation
        variance = (accumulator['sum_sq'] / accumulator['count']) - (np.array(stats['mean']) ** 2)
        stats['std'] = np.sqrt(variance).tolist()

        # Min and max
        stats['min'] = accumulator['min'].tolist()
        stats['max'] = accumulator['max'].tolist()

        # Calculate percentiles
        all_values_array = np.array(accumulator['all_values'])
        if len(all_values_array) > 0:
            stats['q01'] = np.percentile(all_values_array, 1, axis=0).tolist()
            stats['q99'] = np.percentile(all_values_array, 99, axis=0).tolist()
        else:
            # Fallback if no values
            num_dims = len(stats['mean'])
            stats['q01'] = [0] * num_dims
            stats['q99'] = [0] * num_dims

        print(f"{array_name} - Processed {accumulator['count']} samples, {len(stats['mean'])} dimensions")

        return stats

    # Calculate final statistics
    observation_stats = calculate_final_statistics(obs_accumulator, "observation.state")
    action_stats = calculate_final_statistics(act_accumulator, "action")

    # Create output dictionary
    output_dict = {
        "action": action_stats,
        "observation.state": observation_stats,
    }

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    print(f"\nStatistics saved to {output_path}")

    # Print data summary
    print("\nData Summary:")
    print(f"Observation state samples: {obs_accumulator['count']}")
    print(f"Action samples: {act_accumulator['count']}")


def valid_script(src_file_path: str, tar_file_path: str):

    with open(src_file_path, "r", encoding="utf-8") as f:
        scr_dict = json.load(f)

    with open(tar_file_path, "r", encoding="utf-8") as f:
        tar_dict = json.load(f)

    loss = 0
    for modality_type in scr_dict:
        for stats_type in scr_dict[modality_type]:
            src_stats_value = scr_dict[modality_type][stats_type]
            tar_stats_value = tar_dict[modality_type][stats_type]
            loss += np.abs(np.array(src_stats_value) - np.array(tar_stats_value)).sum()

    print(f"Loss: {loss}")



# Example usage
if __name__ == "__main__":
    # Specify the folder containing parquet files
    folder_path = "/home/wsj/Desktop/code/VLA/robot/datasets/libero_spatial"  # Modify this to your folder path

    # Choose which version to use:
    # Version 1: Simple version (for moderate datasets)
    # process_parquet_files(folder_path, "statistics.json")

    # Version 2: Optimized version (for large datasets)
    process_parquet_files_optimized(folder_path, "stats.json")
    # valid_script("/home/wsj/Desktop/code/VLA/robot/datasets/1128/meta/stats.json","/home/wsj/Desktop/code/VLA/robot/datasets/1128/meta/stats.json")
    print("Processing completed!")