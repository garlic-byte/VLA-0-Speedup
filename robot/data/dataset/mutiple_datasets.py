from collections import defaultdict
import numpy as np
from .dataset import SingleLerobotDataset
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torch.distributed as dist
from concurrent.futures import Future, ThreadPoolExecutor
import time
import logging
from ...config.finetune_config import DataConfig
from robot.utils import write_configs_to_jsons

def merge_statistics(statistics: dict[str, np.ndarray], weights: tuple[float]) -> dict[str, np.ndarray]:
    """
    Merge statistics across multiple datasets.
    :param statistics:
        'state.arm': {
                'mean': np.ndarray, shape (length_datasets, dimension_state_arm)
                'min': np.ndarray, shape (length_datasets, dimension_state_arm)
                'max': np.ndarray, shape (length_datasets, dimension_state_arm)
        }
    :param weights: weights for each dataset. length of weights == length_datasets
    :return:
        'state.arm': {
                'mean': np.ndarray, shape (dimension_state_arm, )
                'min': np.ndarray, shape (dimension_state_arm, )
                'max': np.ndarray, shape (dimension_state_arm, )
        }
    """
    mg_stats = {}


class MultipleShardDataset(IterableDataset):
    """
    Multiple datasets as input for training will be split as evenly as possible into different vessel.
    1. Merge statistics of same modality id in all datasets.
    2. Create schedule according weights of datasets for training.
    """
    def __init__(self, config: DataConfig):
        # Initialize parameters
        self.epoch = -1
        self._executor = None
        self.cur_shard = None
        self.cur_sample_index = 0
        self._cache_job: Future | None = None
        self.num_shards_per_epoch = int(1e5)
        self.seed = config.seed
        self.config_output_dir = config.config_output_dir

        # Initialize rank and world
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Normalize weights and convert to np.ndarray
        dataset_paths = config.dataset_path.split(',')
        dataset_weights = config.dataset_weights if config.dataset_weights else [1.0] * len(dataset_paths)
        weights = np.array(dataset_weights)
        self.weights = weights / np.sum(weights)

        # Convert dataset_path to datasets
        self.datasets = [
            SingleLerobotDataset(
                    modality_id=config.modality_id,
                    dataset_path=dataset_path,
                    video_backend=config.video_backend,
                    seed=config.seed,
                    mask_ratio=config.mask_ratio,
                    image_resize=config.image_resize,
                    is_train=config.is_train,
                    vlm_processor_path=config.vlm_processor_path,
                    config_output_dir=config.config_output_dir,
            ) for dataset_path in dataset_paths
        ]

        assert len(self.datasets) > 0 and len(self.datasets) == len(self.weights), (
            f"length of datasets and weights should be same, but {len(self.datasets)} != {len(self.weights)}"
        )

        # Merge statics of datasets
        self.merge_datasets_statistics()


    def merge_datasets_statistics(self):
        """Reset statistics in all datasets just for same modality id."""
        # Count all statistics of datasets
        all_modality_stats = {}
        modality_id = self.datasets[0].modality_id
        for single_dataset in self.datasets:
            assert single_dataset.modality_id == modality_id, (
                f"all the datasets modality id should be same, but {single_dataset.modality_id} != {modality_id}"
            )
            dataset_statistics = single_dataset.get_dataset_statistics
            # Split each single modality type, for example ('action.arm', 'action.gripper')
            for modality_type, modality_stats in dataset_statistics.items():
                if modality_type not in all_modality_stats:
                    all_modality_stats[modality_type] = defaultdict(list)

                # Split each single stats type ('mean', 'max', 'min')
                for stats_type, stats in modality_stats.items():
                    all_modality_stats[modality_type][stats_type].append(stats)

        # Calculate complete stats
        combined_statistics = {}
        for modality_type, modality_stats in all_modality_stats.items():
            # Convert list[np.ndarray, ...] to np.ndarray
            np_modality_stats = {stats_type: np.stack(stats, axis=0) for stats_type, stats in modality_stats.items()}

            # Merge stats of each dataset that contains stats_type like min, max, mean
            combined_statistics[modality_type] = {}
            combined_statistics[modality_type]['min'] = np.min(np_modality_stats['min'], axis=0)
            combined_statistics[modality_type]['max'] = np.max(np_modality_stats['max'], axis=0)
            combined_statistics[modality_type]['mean'] = np.sum(np_modality_stats['mean'] * self.weights.reshape(-1, 1), axis=0)

        # Set and save stats for each dataset
        for dataset in self.datasets:
            dataset.set_dataset_statistics(combined_statistics)
        write_configs_to_jsons(self.config_output_dir, global_stats=combined_statistics)

    def _initialize_sample_schedule(self):
        """
        Return sample schedule from datasets that contain list of (index_dataset, shard_index).
        """
        rng = np.random.default_rng(self.seed + self.epoch)
        # Revise weights for each dataset because some datasets may have less length and some have more.
        avg_length_dataset_shard = np.array([dataset.get_avg_shard_lengths() for dataset in self.datasets])
        normalized_weights = self.weights / avg_length_dataset_shard
        normalized_weights = normalized_weights / np.sum(normalized_weights)

        # [length_dataset] * [percentage_sample] = self.weights
        dataset_sampling_schedule = rng.choice(
            len(self.datasets), size=self.num_shards_per_epoch, p=normalized_weights
        )

        # Crate sample schedule contains (index_dataset, index_shard)
        sample_schedule = []
        indices_shard = [[]] * len(self.datasets)
        for index_dataset in dataset_sampling_schedule:
            # supplement indices of shard to corresponding datasets
            if len(indices_shard[index_dataset]) == 0:
                length_shards = list(range(len(self.datasets[index_dataset])))
                rng.shuffle(length_shards)
                indices_shard[index_dataset] = length_shards
            # assure that every shard should be in the schedule
            sample_schedule.append((index_dataset, indices_shard[index_dataset].pop()))

        return sample_schedule

    def _filter_sample_schedule(self, sample_schedule):
        """Assure that samples are divided into equal parts across different ranks."""
        cur_worker_info = get_worker_info()
        if cur_worker_info is not None:
            cur_worker_id = cur_worker_info.id
            total_workers = cur_worker_info.num_workers
        else:
            cur_worker_id = 0
            total_workers = 1

        filtered_sample_indices = []
        for index, sample_index in enumerate(sample_schedule):
            if index % (self.world_size * total_workers) == self.rank * total_workers + cur_worker_id:
                filtered_sample_indices.append(sample_index)
        return filtered_sample_indices


    def _reset_environment(self):
        """Reset environment after each epoch."""
        self.epoch += 1
        self.cur_sample_index = 0
        cur_sample_schedule = self._initialize_sample_schedule()
        self.filtered_sample_indices = self._filter_sample_schedule(cur_sample_schedule)

    def _start_get_shard(self):
        """Get shard from dataset through thread."""
        if self.cur_sample_index >= len(self.filtered_sample_indices):
            self._reset_environment()

        # Sample from filtered_sample_indices
        index_dataset, indices_shard = self.filtered_sample_indices[self.cur_sample_index]
        self._cache_job = self._executor.submit(
            self.datasets[index_dataset].get_shard, indices_shard
        )

    def _wait_get_shard(self):
        """Wait until shard is load."""
        assert self._executor is not None
        self.cur_shard = self._cache_job.result()
        self.cur_sample_index += 1


    def __iter__(self):
        """Iterate over datasets use thread for efficiency load dataset."""
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._reset_environment()
        rng = np.random.default_rng(self.seed + self.epoch)

        # Start caching next shard dataset
        self._start_get_shard()
        while True:
            start_time = time.time()
            self._wait_get_shard()
            end_time = time.time()
            logging.info(f"[Data loaded] Load shard {self.cur_sample_index} took {end_time - start_time} seconds.")

            # Start caching next shard dataset for save time
            self._start_get_shard()
            # Yield data from random index of episode
            episode_indices = list(range(len(self.cur_shard)))
            rng.shuffle(episode_indices)
            for episode_index in episode_indices:
                yield self.cur_shard[episode_index]
