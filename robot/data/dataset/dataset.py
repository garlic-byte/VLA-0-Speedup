import logging
from typing import Dict
import pandas as pd

from robot.config.finetune_config import DataConfig
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import numpy as np

from robot.data.dataset.lerobot_loader import LerobotLoader
from robot.data.transformer import Transformer, DatasetCollator
import torch.distributed as dist
from concurrent.futures import Future, ThreadPoolExecutor
import time
from robot.config.data.embodiment_config import MODALITY_CONFIGS

class SingleLerobotDataset(Dataset):

    def __init__(self, config: DataConfig, config_output_dir: str | None = None):
        """Initialize the Lerobot dataset."""
        super().__init__()
        self._modality_id = config.modality_id
        self.action_name = config.action_name
        self.dataset_path = config.dataset_path
        self.modality_config = MODALITY_CONFIGS[config.modality_id]
        # Varify action in modality config
        assert self.action_name in self.modality_config, f"Use modality name {self.action_name} for training is not in modality config."

        # Load dataset of lerobot form
        self.dataset = LerobotLoader(
            dataset_path=config.dataset_path,
            modality_config=self.modality_config,
            video_backend=config.video_backend,
            video_backend_kwargs=config.video_backend_kwargs,
            action_name=self.action_name,
        )
        # Create balanced shards from episode timesteps
        self._seed = config.seed
        self.rng = np.random.default_rng(config.seed)
        self.episode_split_ratio = config.episode_split_ratio
        self.dataset_path = config.dataset_path
        self.modality_config = self.modality_config
        action_delta_indices = self.modality_config[self.action_name].delta_indices
        self._action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
        self.shard_size = config.shard_size
        self.shard_episodes = None
        self.shard_lengths = None
        self.shard_dataset()

        # Transform vision, language, state and action into VLM inputs for one sample
        self.transformer = Transformer(
            modality_config=self.modality_config,
            statistics=self.dataset.get_dataset_statistics(),
            mask_ratio=config.mask_ratio,
            num_bin_actions=config.num_bin_actions,
            input_shape=self.dataset.get_image_shape(),
            image_resize=config.image_resize,
            crop_fraction=config.crop_fraction,
            color_jitter=config.color_jitter,
            is_train=config.is_train,
            vlm_processor_path=config.vlm_processor_path,
            predict_action_nums=len(action_delta_indices),
            action_dimensions=self.dataset.get_action_dimensions(),
            action_name=self.action_name,
            config_output_dir=config_output_dir,
        )

        # Transform VLM inputs into model inputs for one batch samples
        self._collator = DatasetCollator(
            vlm_processor_path=config.vlm_processor_path,
            mask_ratio=config.mask_ratio,
        )

    @property
    def modality_id(self):
        """Get the modality type."""
        return self._modality_id

    @property
    def collator(self):
        """Get the collator class."""
        return self._collator

    @property
    def action_horizon(self):
        """Get horizon of actions."""
        return self._action_horizon

    @property
    def decode_actions(self):
        """Get decoded actions class."""
        return self.transformer.state_action_processor.unapply_action

    @property
    def get_dataset_statistics(self):
        """Get dataset statistics."""
        return self.dataset.get_dataset_statistics()

    @property
    def seed(self):
        """Get random seed."""
        return self._seed

    def get_effective_lengths(self, episode_index) -> int:
        """Return the effective length of the dataset."""
        original_length = self.dataset.episode_lengths[episode_index]
        return max(0, original_length - self.action_horizon + 1)

    def shard_dataset(self):
        """
        Create balanced shards by distributing episode timesteps across shards.
        This sharding process:
        1. Shuffle all episode from datasets
        2. Split each episode into multiple sub-sequences based on sampling rate
        3. Distribute sub-sequences across shards to balance shard sizes
        4. Use greedy assignment to minimize shard size variance
        :return:
        """
        shutil_episode_indices = self.rng.permutation(len(self.dataset))
        num_splits = int(1/ self.episode_split_ratio)
        assert len(shutil_episode_indices) > 0, f"{self.dataset_path} is empty!"

        # Calculate total effective numbers of datasets
        total_effective_length = np.sum(
            [self.get_effective_lengths(episode_index) for episode_index in shutil_episode_indices]
        ).astype(int)

        # Initialize shard vessel
        num_sharded_vessel = np.ceil(total_effective_length/ self.shard_size).astype(int)
        shard_episodes = [[] for _ in range(num_sharded_vessel)]
        shard_lengths = np.zeros(num_sharded_vessel, dtype=int)

        # Try best to split average all episodes into shard vessel
        for episode_index in shutil_episode_indices:
            step_indices = np.arange(0, self.get_effective_lengths(episode_index))
            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                # Use greedy balancing all shard episodes
                min_length_shard_index = np.argmin(shard_lengths)
                shard_episodes[min_length_shard_index].append((episode_index, split_step_indices))
                shard_lengths[min_length_shard_index] += len(split_step_indices)

        assert all(shard_lengths[i] > 0 for i in range(num_sharded_vessel)), (
            "All shard episodes must have length > 0."
        )
        logging.info(f"----------------------------Data loaded----------------------------")
        logging.info(f"[Data loaded] Path of dataset {self.dataset_path}")
        logging.info(f"[Data loaded] Id of modality {self.modality_id}")
        logging.info(f"[Data loaded] One episode has been split into {num_splits} episodes.")
        logging.info(f"[Data loaded] Total effective length of episodes: {total_effective_length}, {num_sharded_vessel} shard vessel were created.")

        self.shard_episodes = shard_episodes
        self.shard_lengths = shard_lengths

    def __len__(self):
        """Return the total number of shard vessels."""
        return len(self.shard_episodes)

    def get_step_data(self, episode_data: pd.DataFrame, step_index: int) -> Dict:
        """
        Get one step of data from episode data, it contains language, state, action and images commonly.
        :param episode_data: some steps from parquet
        :param step_index: need index of episode_data
        :return: Dict, contains four main kinds:
            -language: dict[str: list]
            -state: dict[str, np.array(horizon, dim)]
            -action: dict[str, np.array(horizon, dim)]
            -video: dict[str, np.array(width, height, dim)]
        """
        step_data = {}
        # Get every modality type
        for modality_type in self.modality_config:
            step_data[modality_type] = {}
            delta_indices = self.modality_config[modality_type].delta_indices
            modality_keys = self.modality_config[modality_type].modality_keys

            # Update sample timesteps according delta indices
            sample_timesteps = [delta_index + step_index for delta_index in delta_indices]
            for key in modality_keys:
                # get modality data by sample timesteps
                modality_name = f"{modality_type}.{key}"
                assert modality_name in episode_data.columns, f"{modality_name} is not in parquet file."
                modality_data = episode_data[modality_name].iloc[sample_timesteps]

                if modality_type == "language":
                    step_data[modality_type] = modality_data.tolist()[0]
                else:
                    step_data[modality_type][key] = np.vstack(
                        [
                            np.array(modality_data.iloc[i]).astype(np.float32)
                            for i in range(len(modality_data))
                        ]
                    )
        return self.transformer(step_data)


    def get_shard(self, index: int):
        """
        Load all episode timesteps across shards.
        :param index: Shard index to load
        :return: List of episode timesteps for model training
        """
        assert self.transformer is not None, (
            f"Get shard must have transformer initialized."
        )
        episode_info = self.shard_episodes[index]
        datapoints = []
        for episode_index, split_step_indices in episode_info:
            episode_data = self.dataset[episode_index]
            for split_step_index in split_step_indices:
                datapoints.append(self.get_step_data(episode_data, split_step_index))
        return datapoints

    def __getitem__(self, episode_index):
        """Get one episode from dataset."""
        one_episode_data = []
        episode_data = self.dataset[episode_index]
        # step_indices = np.arange(0, self.get_effective_lengths(episode_index))
        for step_index in range(self.get_effective_lengths(episode_index)):
            one_episode_data.append(self.get_step_data(episode_data, step_index))
        return one_episode_data


class ShardCacheDataset(IterableDataset):
    """
    Background shard caching with ThreadPoolExecutor for efficiency load dataset.
    """
    epoch = -1
    cur_shard = None
    cur_shard_index = 0
    _executor = None
    _cache_job: Future = None
    def __init__(self, dataset: SingleLerobotDataset):
        self.dataset = dataset
        self.seed = dataset.seed

        # Initialize rank and world
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0


    def _initialize_sample_schedule(self) -> list:
        """Return sample schedule from dataset."""
        rng = np.random.default_rng(self.seed + self.epoch)
        sample_indices = list(range(len(self.dataset)))
        rng.shuffle(sample_indices)
        return sample_indices

    def _filter_sample_schedule(self, sample_indices) -> list:
        """Return filtered samples schedule according to current rank and worker_id."""
        cur_worker_info = get_worker_info()
        if cur_worker_info is not None:
            cur_worker_id = cur_worker_info.id
            total_workers = cur_worker_info.num_workers
        else:
            cur_worker_id = 0
            total_workers = 1

        filtered_sample_indices = []
        for index, sample_index in enumerate(sample_indices):
            if index % (self.world_size * total_workers) == self.rank * total_workers + cur_worker_id:
                filtered_sample_indices.append(sample_index)
        return filtered_sample_indices
    
    def _reset_environment(self):
        """Reset environment after each epoch."""
        self.epoch += 1
        self.cur_shard_index = 0
        cur_sample_schedule = self._initialize_sample_schedule()
        self.filtered_sample_indices = self._filter_sample_schedule(cur_sample_schedule)

    
    def _start_get_shard(self):
        """Get shard from dataset through thread."""
        if self.cur_shard_index >= len(self.filtered_sample_indices):
            self._reset_environment()

        self._cache_job = self._executor.submit(self.dataset.get_shard,self.filtered_sample_indices[self.cur_shard_index])

    def _wait_get_shard(self):
        """Wait until shard is load."""
        assert self._executor is not None
        self.cur_shard = self._cache_job.result()
        self.cur_shard_index += 1


    def __iter__(self):
        """Iterate over dataset use thread for efficiency load dataset."""
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._reset_environment()
        rng = np.random.default_rng(self.seed + self.epoch)

        # Start caching next shard dataset
        self._start_get_shard()
        while True:
            start_time = time.time()
            self._wait_get_shard()
            end_time = time.time()
            logging.info(f"[Data loaded] Load shard {self.cur_shard_index} took {end_time - start_time} seconds.")

            self._start_get_shard()
            episode_indices = list(range(len(self.cur_shard)))
            rng.shuffle(episode_indices)
            for episode_index in episode_indices:
                yield self.cur_shard[episode_index]


