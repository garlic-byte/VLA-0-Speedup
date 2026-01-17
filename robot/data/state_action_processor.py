from collections import defaultdict
from copy import deepcopy

import numpy as np

from robot.config.data.modality_config import ModalityConfig

def normalize_values_mean_std(values: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Normalize the values to zero mean and unit variance.
    :param values: Shape (T, D) or (B, T, D) where B is batch size, T is number of timesteps and D is number of features.
    :param params: Shape (D, ) or Shape (T, D), dictionary with "mean" and "std.
    :return: Same shape as `values`.
    """
    mean = params["mean"]
    std = params["std"]

    # Prevent denominator from being 0, keep original values for zero-std features.
    mask = std != 0
    normalized = np.zeros_like(values)
    normalized[..., mask] = (values[..., mask] - mean[..., mask]) / std[..., mask]
    normalized[..., ~mask] = values[..., ~mask]

    return normalized


def normalize_values_min_0_max_num_bin_actions(
    values: np.ndarray,
    params: dict[str, np.ndarray],
    num_bin_actions: int
) -> np.ndarray:
    """
    Normalize the minimum values to zero, and normalize the maximum values to num_bin_actions.
    :param values: Shape (T, D) or (B, T, D) where B is batch size, T is number of timesteps and D is number of features.
    :param params: Shape (D, ) or Shape (T, D), dictionary with "min" and "max.
    :param num_bin_actions: Shape(1, ), project maximum values to num_bin_actions.
    :return: Same shape as `values`.
    """
    min_values = params["min"]
    max_values = params["max"]
    normalized = np.zeros_like(values)
    mask = ~np.isclose(min_values, max_values)
    normalized[..., mask] = (values[..., mask] - min_values[..., mask]) / (max_values[..., mask] - min_values[..., mask])
    normalized *= num_bin_actions
    normalized = np.round(normalized).astype(int)
    return normalized


def denormalize_values_min_0_max_num_bin_actions(
    values: np.ndarray,
    params: dict[str, np.ndarray],
    num_bin_actions: int
) -> np.ndarray:
    """
    Denormalize values from [0, num_bin_actions] back to original [min, max] range (reverse of normalize function).
    :param values: Shape (T, D) or (B, T, D) where B is batch size, T is number of timesteps and D is number of features.
                              Values are rounded integers in [0, num_bin_actions] from normalization.
    :param params: Dictionary with keys "min" and "max":
       - "min": Shape (D, ) or (T, D) (original minimum values for each feature)
       - "max": Shape (D, ) or (T, D) (original maximum values for each feature)
    :param num_bin_actions: Scalar (original maximum projection value in normalization).
    :return: Same shape as `normalized_values`, denormalized to original [min, max] range.
    """
    min_values = np.array(params["min"])
    max_values = np.array(params["max"])

    denormalized = np.zeros_like(values, dtype=np.float32)

    # Prevent denominator from being 0, keep original values for zero-std features.
    mask = ~np.isclose(min_values, max_values)

    # Core denormalization logic (reverse of normalization steps):
    # 1. Divide by num_bin_actions to revert to [0, 1] range
    # 2. Multiply by (max - min) to restore original scale ratio
    # 3. Add min_values to restore original baseline
    denormalized[..., mask] = (
            (values[..., mask] / num_bin_actions) *
            (max_values[..., mask] - min_values[..., mask]) +
            min_values[..., mask]
    )

    # For features with min == max (mask=False), fill with min value (consistent with normalization logi
    denormalized[..., ~mask] = min_values[..., ~mask]

    return denormalized


def unnormalize_values_mean_std(values: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Unnormalize the values to zero mean and unit variance.
    :param values: Shape (T, D) or (B, T, D) where B is batch size, T is number of timesteps and D is number of features.
    :param params: Shape (D, ) or Shape (T, D), dictionary with "mean" and "std.
    :return: Same shape as `values`.
    """
    mean = params["mean"]
    std = params["std"]

    # Spacial for zero-std features
    mask = std != 0
    unnormalized = np.zeros_like(values)
    unnormalized[..., mask] = values[..., mask] * std[..., mask] + mean[..., mask]
    unnormalized[..., ~mask] = values[..., ~mask]

    return unnormalized


class StateActionProcessor:
    """
    Unified processing of states and actions.
    Handles:
    - State and action normalization (min/ max, mean/ std)
    """
    def __init__(
        self,
        modality_config: ModalityConfig | dict,
        statistics: dict | None = None,
        clip_outliers: bool = True,
        num_bin_actions: int = 1000,
        predict_action_nums: int = 10,
        action_dimensions: dict | None = None,
        action_name : str = 'action',
    ):
        # modality_config may be config from loaded config.json
        self.modality_config = modality_config if isinstance(modality_config, ModalityConfig) else ModalityConfig(**modality_config)
        self.statistics = statistics
        self.clip_outliers = clip_outliers
        self.num_bin_actions = num_bin_actions
        self.predict_action_nums = predict_action_nums
        self.action_dimensions = action_dimensions
        self.action_name = action_name


    def _convert_action_to_text(self, action: dict[str, np.ndarray]):
        """
        :param action: dictionary with keys "modality type" and "values" of action, the "values" shape (action_delta_indices, action_dimension)
        :return: string with 0-9, space and enter
        """
        target_action = []
        for index in range(self.predict_action_nums):
            cur_timestep_action = []
            for action_position in action.values():
                cur_timestep_action.append(action_position[index, :])
            cur_action = " ".join(np.concatenate(cur_timestep_action, axis=0).astype(str))
            target_action.append(cur_action)
        return " ".join(target_action)

    def _convert_text_to_action(self, text_action: str) -> dict[str, np.ndarray]:
        """
        :param text_action: string with 0-9, space and enter
        :return: The returned result is a np.ndarray of action steps,
            each containing a dictionary with each joint name as a key and joint position as a value
            Example:
                {
                    "arm": np.zeros((8, 6)),
                    "gripper": np.zeros((8, 1)),
                }
        """
        action = defaultdict(list)
        ls_action = [line.strip() for line in text_action.strip().split('\n') if line.strip()]
        for index_action in range(self.predict_action_nums):
            step_action = []
            if index_action < len(ls_action):
                # convert each line text that represent step action to list
                step_action = ls_action[index_action].split(" ")

            # Insert each joint group into action
            start_index = 0
            for joint_group, joint_dimension in self.action_dimensions.items():
                end_index = start_index + joint_dimension

                # Check action whether is valid
                if end_index <= len(step_action):
                    action[joint_group].append(np.array(step_action[start_index:end_index], dtype=np.float32))
                else:
                    # TODO revise simulate action that should be normalized action, not unnormalized action.
                    simulate_action = np.array(self.statistics[f"{self.action_name}.{joint_group}"]["mean"])
                    print(f"[warning] predict {self.action_name} is not valid, {step_action = }, {simulate_action = }")
                    action[joint_group].append(simulate_action)

                start_index = end_index

        # Convert lists of joint group to numpy
        return {joint_group: np.stack(action[joint_group], axis=0, dtype=np.float32) for joint_group in action.keys()}

    def _convert_text_to_action_v2(self, text_action: str) -> dict[str, np.ndarray]:
        """
        :param text_action: string with 0-9 and space
        :return:
        """
        action = defaultdict(list)
        text_ls = text_action.strip().split(' ')
        start_index = 0
        for index_action in range(self.predict_action_nums):
            for joint_group, joint_dimension in self.action_dimensions.items():
                end_index = start_index + joint_dimension
                action[joint_group].append(np.array(text_ls[start_index:end_index], dtype=np.float32))
                start_index = end_index
        return {key: np.stack(value, axis=0, dtype=np.float32) for key, value in action.items()}


    def apply_action(self, action: dict[str, np.ndarray]) -> str:
        """
        Apply action processing (normalize, encoding).
        :param action: Dict mapping action joint group -> raw values.
        :return: Dict mapping action joint group -> processed values.
        """
        normalized_values = {}
        values = deepcopy(action)
        for joint_group in self.modality_config.modality_keys:
            assert joint_group in values, f"Joint group {joint_group} not in parquet file."
            params = self.statistics[f"{self.action_name}.{joint_group}"]
            normalized = normalize_values_min_0_max_num_bin_actions(
                values=values[joint_group],
                params=params,
                num_bin_actions=self.num_bin_actions,
            )
            normalized_values[joint_group] = normalized

        return self._convert_action_to_text(normalized_values)

    def unapply_action(self, text_action: str) -> dict[str, np.ndarray]:
        """Convert text to action.
        :param text_action: string with 0-9, space and enter
        :return: The returned result is dictory of a np.ndarray of action steps,
        """

        # Get normalized actions
        action = self._convert_text_to_action_v2(text_action)

        # Restore origin values range according dataset statics
        unnormalized_values = {}
        values = deepcopy(action)
        for joint_group in self.modality_config.modality_keys:
            assert joint_group in values, f"Joint group {joint_group} not in parquet file."
            params = self.statistics[f"{self.action_name}.{joint_group}"]
            unnormalized = denormalize_values_min_0_max_num_bin_actions(
                values=values[joint_group],
                params=params,
                num_bin_actions=self.num_bin_actions
            )
            unnormalized_values[joint_group] = unnormalized

        return unnormalized_values
