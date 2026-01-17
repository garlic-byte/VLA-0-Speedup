from dataclasses import dataclass
from typing import List


@dataclass
class ModalityConfig:
    delta_indices: List[int]
    """Delta indices to sample relative to the current index."""
    modality_keys: List[str]
    """The keys to load for each modality."""