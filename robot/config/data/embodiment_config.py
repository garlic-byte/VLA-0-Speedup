from robot.config.data.modality_config import ModalityConfig

MODALITY_CONFIGS = {
    "ymbot_d":{
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["top", "left", "right"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        )
    },
    "libero_panda":{
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["arm", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        )
    }
}