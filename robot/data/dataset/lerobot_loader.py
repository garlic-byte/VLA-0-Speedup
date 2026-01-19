import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import Optional, Literal

from robot.config.data.modality_config import ModalityConfig
from robot.utils import get_frames_by_indices, process_parquet_files_optimized

LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILENAME = "stats.json"

ALLOWED_MODALITIES = ["video", "state", "action", "language"]
DEFAULT_MODALITY_MAP = {
    "action": "action",
    "state": "observation.state",
}

class LerobotLoader:
    def __init__(
        self,
        dataset_path: str | Path,
        modality_config: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict = None,
        action_name: str = "action",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.modality_config = modality_config
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.action_name = action_name

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Load metadata file
        self.tasks_map = {}
        self._statistics = defaultdict(dict)
        self._load_metadata()

        # Check modality configs after load metadata
        self._validate_modality_configs(modality_configs=modality_config)
        self.episode_lengths = self.get_episode_lengths()


    def _load_metadata(self):
        """Load metadata from jsonl file."""
        meta_path = self.dataset_path / LEROBOT_META_DIR_NAME
        with open(meta_path / LEROBOT_INFO_FILENAME, "r") as f:
            self.info_meta = json.load(f)
        with open(meta_path / LEROBOT_EPISODES_FILENAME, "r") as f:
            self.episodes_meta = [json.loads(line) for line in f]
        with open(meta_path / LEROBOT_TASKS_FILENAME, "r") as f:
            self.tasks_meta = [json.loads(line) for line in f]
            self.tasks_map = {task_info["task_index"]: task_info["task"] for task_info in self.tasks_meta}

        with open(meta_path / LEROBOT_MODALITY_FILENAME, "r") as f:
            self.modality_meta = json.load(f)

        # Calculate stats
        if not (meta_path / LEROBOT_STATS_FILENAME).exists():
            process_parquet_files_optimized(str(self.dataset_path), LEROBOT_STATS_FILENAME)
        with open(meta_path / LEROBOT_STATS_FILENAME, "r") as f:
            self.stats_meta = json.load(f)

        # Extract key configuration parameters
        self.feature_config = self.info_meta["features"]
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta["video_path"]
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta["fps"]

    @staticmethod
    def _validate_modality_configs(modality_configs: dict[str, ModalityConfig]) -> None:
        """Validate the modality configs."""
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"{modality} is not a supported modality")
            if modality == "language":
                assert modality_configs[modality].delta_indices == [0], (
                    "Language modality delta indices must be [0]"
                )
                assert len(modality_configs[modality].modality_keys) == 1, (
                    "Language modality modality keys must be exactly 1 "
                )

    def get_episode_lengths(self) -> list[int]:
        """Get length of each episode."""
        episode_lengths = []
        for episode in self.episodes_meta:
            episode_lengths.append(int(episode["length"]))
        return episode_lengths

    def get_dataset_statistics(self):
        """
        Get statistics of state and action from dataset.
        Return example: {
            'state.arm': {
                'mean': np.ndarray,
                'min': np.ndarray,
                'max': np.ndarray,
                }
        }
        """
        statistics = {}
        for modality_type in ["state", "action"]:
            if modality_type not in self.modality_config:
                continue

            # Get start and end of modality (arm, gripper) from meta info
            modality_info = self.modality_meta.get(modality_type, {})
            for modality in self.modality_config[modality_type].modality_keys:
                assert modality in modality_info, f"{modality} is not in {LEROBOT_MODALITY_FILENAME}"
                start_idx = modality_info[modality]["start"]
                end_idx = modality_info[modality]["end"]

                # modality type need converted
                original_key = DEFAULT_MODALITY_MAP[modality_type]
                statistics[f"{modality_type}.{modality}"] = {}
                for stats_type in ["mean", "min", "max"]:
                    statistics[f"{modality_type}.{modality}"][stats_type] = (
                        np.array(self.stats_meta[original_key][stats_type][start_idx:end_idx]).astype(np.float32)
                    )
        return statistics

    def get_episodes_length(self, idx: int) -> int:
        """Get specified episode length."""
        return self.episode_lengths[idx]

    def _get_joint_groups(
        self,
        df: pd.DataFrame,
        modality_type: str = "state",
    ):
        """Get all the state/ action info from parquet filename."""
        modality_info = self.modality_meta.get(modality_type, {})
        joint_data = pd.DataFrame()
        for modality in self.modality_config[modality_type].modality_keys:
            assert modality in modality_info, f"{modality} is not in {LEROBOT_MODALITY_FILENAME}"
            start_idx = modality_info[modality]["start"]
            end_idx = modality_info[modality]["end"]

            # Extract data from every joint
            original_key = DEFAULT_MODALITY_MAP[modality_type]
            joint_data[f"{modality_type}.{modality}"] = df[original_key].map(lambda x: x[start_idx:end_idx])

        return joint_data


    def _load_parquet_data(self, episode_idx: int) -> pd.DataFrame:
        """Load and process parquet data for a single episode."""
        chunk_idx = episode_idx // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_idx,
        )
        parquet_path = self.dataset_path / parquet_filename
        original_df = pd.read_parquet(parquet_path)
        processed_df = pd.DataFrame()

        # Convert task index to Language
        if "language" in self.modality_config:
            language_key = self.modality_config["language"].modality_keys[0]
            split_language_key = language_key.split(".")
            assert len(split_language_key) == 3, "Language modality key must be annotation.human.task_description"
            head_language_key = split_language_key[0]
            tail_language_key = ".".join(split_language_key[1:])
            original_key = self.modality_meta[head_language_key][tail_language_key].get("original_key")
            processed_df[f"language.{language_key}"] = original_df[original_key].apply(
                lambda x: self.tasks_map[x]
            )

        # Extract joint groups for state and action modalities from parquet
        for modality in ["state", "action"]:
            if modality not in self.modality_config:
                continue
            joint_groups_df = self._get_joint_groups(original_df, modality)
            processed_df = pd.concat([processed_df, joint_groups_df], axis=1)

        return processed_df


    def _load_video_data(self, episode_idx: int, indices: np.ndarray) -> pd.DataFrame:
        chunk_idx = episode_idx // self.chunk_size
        video_keys = self.modality_config["video"].modality_keys
        video_data = pd.DataFrame()
        for video_key in video_keys:
            original_video_key = self.modality_meta["video"][video_key].get("original_key")
            assert original_video_key in self.feature_config, (
                f"Original video key {original_video_key} not in feature_config"
            )

            # Construct video file path using pattern
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx, video_key=original_video_key, episode_index=episode_idx,
            )
            video_path = self.dataset_path / video_filename
            frames_arr = get_frames_by_indices(
                video_path,
                indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )
            video_data[f"video.{video_key}"] = list(frames_arr)
        return video_data


    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the image."""
        height = width = None
        video_keys = self.modality_config["video"].modality_keys
        for video_key in video_keys:
            original_video_key = self.modality_meta["video"][video_key].get("original_key")
            assert original_video_key in self.feature_config, (
                f"Original video key {original_video_key} not in feature_config"
            )
            if height is None or width is None:
                height, width, _ = self.feature_config[original_video_key]["shape"]
            else:
                assert [height, width, 3] == self.feature_config[original_video_key]["shape"], (
                    f"Shape {height}, {width} does not match in features."
                )

        return (height, width)

    def get_action_dimensions(self) -> dict[str, int]:
        """Get the dimensions of the action."""
        action_dimensions = {}
        # Whether to use action as trained modality, otherwise use state
        modality_info = self.modality_meta.get(f"{self.action_name}", {})
        for modality in self.modality_config[f"{self.action_name}"].modality_keys:
            assert modality in modality_info, f"{modality} is not in {LEROBOT_MODALITY_FILENAME}"
            start_idx = modality_info[modality]["start"]
            end_idx = modality_info[modality]["end"]

            # Service for transformer: get actions dimensions
            action_dimensions[modality] = end_idx - start_idx
        return action_dimensions

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """Get a specific data from the dataset, it contains language, state, action and video."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        episode_meta = self.episodes_meta[idx]
        episode_len = episode_meta["length"]
        episode_idx = episode_meta["episode_index"]

        # Load language, state, language
        parquet_data = self._load_parquet_data(episode_idx)

        # Update parquet data length
        episode_len = min(len(parquet_data), episode_len)

        # Load video
        video_data = self._load_video_data(episode_idx, np.arange(episode_len))

        # Add video frames into parquet data as PIL images
        parquet_data = pd.concat([parquet_data, video_data], axis=1)

        return parquet_data

    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.episode_lengths)



if __name__ == "__main__":
    from robot.config.data.embodiment_config import MODALITY_CONFIGS
    dataset = LerobotLoader(
        dataset_path=r"/home/wsj/Desktop/code/github/Isaac-GR00T/demo_data/1203",
        modality_config=MODALITY_CONFIGS["ymbot_d"],
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    start_time = time.time()
    data = dataset[5]["video.top"]
    print()
    end_time = time.time()
    print(end_time - start_time)