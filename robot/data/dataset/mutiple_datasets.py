from collections import defaultdict

from .dataset import IterableDataset, SingleLerobotDataset
import torch.distributed as dist
from concurrent.futures import Future, ThreadPoolExecutor


class MultipleShardDataset(IterableDataset):
    """
    Multiple datasets as input for training will be split as evenly as possible into different vessel.
    1. Merge statistics of same modality id in all datasets.
    2. Create schedule according weights of datasets for training.
    """
    def __init__(self,
        datasets: list[SingleLerobotDataset],
        weights: list[float],
        seed: int,
    ):
        self.datasets = datasets
        self.weights = weights
        self.seed = seed

        # Initialize rank and world
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Merge statics of datasets
        self.merge_datasets_statistics()

    def merge_datasets_statistics(self):
        """Reset statistics by meaning it of same modality id in all datasets."""
        # Count all statistics of datasets
        modality_id_stats = defaultdict(list)
        for single_dataset in self.datasets:
            modality_id = single_dataset.modality_id()
            dataset_statistics = single_dataset.get_dataset_statistics
            modality_id_stats[modality_id].append(dataset_statistics)

        # Calculate mean values for
