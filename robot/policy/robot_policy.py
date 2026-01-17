import os
from collections import defaultdict

import numpy as np
from transformers import AutoProcessor, LogitsProcessor
from transformers import Qwen3VLForConditionalGeneration
import torch
from safetensors.torch import load_file, safe_open
from robot.data.transformer import Transformer


class NumberSpaceOnlyProcessor(LogitsProcessor):
    """
    Logits processor that constrains generation to only numbers (0-9), spaces, and end-of-text tokens.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Get token IDs for allowed tokens
        self.allowed_tokens = set()

        # Add numbers 0-9
        for i in range(10):
            token_id = tokenizer.encode(str(i), add_special_tokens=False)[0]
            self.allowed_tokens.add(token_id)
        # Add space token
        space_token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        self.allowed_tokens.add(space_token_id)

        # Add enter token
        enter_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.allowed_tokens.add(enter_token_id)

        # Add end of text token
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        # Set logits to negative infinity for all tokens except allowed ones
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0
        return scores + mask


class BaseRobotPolicy:
    def __init__(self, config: dict, model_path: str, device: str = 'cpu'):
        """
        Base robot policy, mainly for test performance of trained model.

        Pipeline
            1. Use processor to convert template into tensors
            2. Use loaded model to generate tokens
            3. Use processor to convert tokens to text
        """
        # Initialize parameters
        self.config = config
        self.device = device
        self.model_path = model_path

        # Load model and processor for inference
        self.model = self.load_model_processor()
        self.processor = AutoProcessor.from_pretrained(self.config["model_path"])
        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

    @property
    def get_processor_path(self):
        """The processor path of the model."""
        return self.config["model_path"]

    def load_model_processor(self):
        """Merge origin Qwen3-vl model with trained VLA model."""

        # Load original model form config
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config["model_path"],
            dtype=eval(self.config["dtype"]),
            device_map='cpu',
        )

        # Load trained lora frame from package peft
        if self.config["lora_rank"] > 1:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=self.config["lora_rank"],
                lora_alpha=self.config["lora_alpha"],
                lora_dropout=self.config["lora_dropout"],
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_config)
            print("Loaded Qwen3-vl model LORA.")

        # Find all safe_tensor files
        safe_tensor_files = [f for f in os.listdir(self.model_path) if f.endswith(".safetensors")]
        if len(safe_tensor_files) > 1: safe_tensor_files.sort(key=lambda x: int(x.split("-")[1]))

        # Load trained weights from saft_tensor files
        trained_weights_state_dict = {}
        for file_name in safe_tensor_files:
            file_path = os.path.join(self.model_path, file_name)
            shard_dict = load_file(file_path)
            new_shard_dict = {}
            for old_key, value in shard_dict.items():
                # core process for full finetuning: model.model.language_model → model.language_model
                if old_key.startswith("model.model."):
                    old_key = old_key.replace("model.model.", "model.")

                # core process for lora: model.base_model → base_model
                elif old_key.startswith("model.base_model."):
                    old_key = old_key.replace("model.base_model.", "base_model.")
                new_shard_dict[old_key] = value
            trained_weights_state_dict.update(new_shard_dict)

        # Load new trained weight to original model
        model.load_state_dict(trained_weights_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        return model


    def get_action(self, inputs: dict) -> str:
        """
        Get action from fine-tuning VLA model.
        :param inputs:
            text: processed instruction and prompt by VLM processor
            image: processed image by VLM processor and transformer
        :return: action of type string calculated by VLA model
        """
        vlm_inputs = self.processor(
            text=inputs['text'], images=inputs['image'], return_tensors="pt", padding=True
        ).to(self.device)
        generated_ids = self.model.generate(
            **vlm_inputs,
            logits_processor=[self.logits_processor],
            max_new_tokens=1024,
            use_cache=True,
        )

        # Only remain the generated result by model, and convert it to text
        output_ids = generated_ids[0][len(vlm_inputs.input_ids[0]):].tolist()
        generated_action_txt = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return ''.join(generated_action_txt)

    def reset(self, options: dict):
        return {}

class RobotPolicy(BaseRobotPolicy):
    def __init__(self, model_config: dict, transformer_config: dict, model_path: str, device: str = 'cpu'):
        super().__init__(model_config, model_path, device)

        # Create fit transformer for image and language
        assert "is_train" in transformer_config, (
            "transformer_config should contain 'is_train' key."
        )
        transformer_config["is_train"] = False
        self.transformer_config = transformer_config
        self.transformer = Transformer(**transformer_config)

    def check_inputs(self, inputs: dict):
        """Check some modalities such as images, language"""
        modality_config = self.transformer_config["modality_config"]

        # Check if the video format meets the requirements
        for video_modality_keys in modality_config['video']["modality_keys"]:
            # check expected video modality whether is in inputs
            assert video_modality_keys in inputs["video"], (
                f"{video_modality_keys} not in inputs['video']"
            )

            # varify video type should be numpy
            video = inputs["video"][video_modality_keys]
            assert isinstance(video, np.ndarray), (
                f"video should be numpy array, but got {type(video)}"
            )

            # varify range of pixel from video should be 0-255
            assert video.dtype == np.uint8, (
                f"video should be numpy uint8, but got {video.dtype}"
            )

            # varify shape should be (H, W, C)
            video_shape = inputs["video"][video_modality_keys].shape
            assert len(video_shape) == 3, (
                f"inputs['video'][{video_modality_keys}].shape should be 3 but is {video_shape}"
            )

            # varify last dimension should be 3 representing R, G, B
            assert video_shape[-1] == 3, (
                f"inputs['video'][{video_modality_keys}] last dimension should be 3 but is {video_shape[-1]}"
            )

        # Check if the language format meets the requirements
        language = inputs["language"]
        assert type(language) == str, (
            f"inputs['language'] should be str but is {type(language)}"
        )


    def get_action(self, observation: dict) -> dict[str, np.ndarray]:
        """Get action for one observation which contains images and language."""
        # Step 1. varify the right of inputs
        self.check_inputs(observation)

        # Step 2. convert inputs to tensors
        processed_inputs = self.transformer.apply(observation).to(self.device)

        # Step 3. Run model inference to get actions
        with torch.no_grad():
            generated_ids = self.model.generate(
            **processed_inputs,
            logits_processor=[self.logits_processor],
            max_new_tokens=1024,
            use_cache=True,
        )
        # Split tokens of model output
        output_ids = generated_ids[0][len(processed_inputs.input_ids[0]):].tolist()
        generated_action_ls = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # convert list to text
        generated_action_txt = ''.join(generated_action_ls)

        # Step 4. decode actions from text
        actions = self.transformer.state_action_processor.unapply_action(generated_action_txt)
        return actions


class SimRobotPolicy(RobotPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, observation: dict):
        """
        Get action for perfectly adapted to simulation environment.
        :param observation:
            key = 'video.image', value.shape = (batch_size, 1, 256, 256, 3)
            key = 'video.wrist_image', value.shape = (batch_size, 1, 256, 256, 3)
            key = 'language', tuple.shape = (batch_size)
        :return:
            key = 'action.x', value.shape = (batch_size, action_dimension, 1)
            key = 'action.y', value.shape = (batch_size, action_dimension, 1)
            key = 'action.z', value.shape = (batch_size, action_dimension, 1)
            key = 'action.roll', value.shape = (batch_size, action_dimension, 1)
            key = 'action.pitch', value.shape = (batch_size, action_dimension, 1)
            key = 'action.yaw', value.shape = (batch_size, action_dimension, 1)
            key = 'action.gripper', value.shape = (batch_size, action_dimension, 1)
        """
        batch_actions = {
            'action.x': [],
            'action.y': [],
            'action.z': [],
            'action.roll': [],
            'action.pitch': [],
            'action.yaw': [],
            'action.gripper': [],
        }
        arm_modality_keys = ['action.x', 'action.y', 'action.z', 'action.roll', 'action.pitch', 'action.yaw']
        gripper_modality_keys = ['action.gripper']

        batch_size = observation['video.image'].shape[0]
        for batch_index in range(batch_size):
            obs = {
                "video": {
                    "image": observation['video.image'][batch_index, 0],
                    "wrist_image": observation['video.wrist_image'][batch_index, 0],
                },
                "language": observation['annotation.human.action.task_description'][batch_index],
            }
            action = super().get_action(obs)

            single_action = defaultdict(list)
            for arm_index, arm_modality_key in enumerate(arm_modality_keys):
                single_action[arm_modality_key].append(action['arm'][:, arm_index])

            for gripper_index, gripper_modality_key in enumerate(gripper_modality_keys):
                single_action[gripper_modality_key].append(action['gripper'][:, gripper_index])

            for modality_key in single_action:
                batch_actions[modality_key].append(np.stack(single_action[modality_key], axis=1))

        return {key: np.stack(value, axis=0) for key, value in batch_actions.items()}