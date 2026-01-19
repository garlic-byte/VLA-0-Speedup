import os.path

import numpy as np

from robot.config.data.modality_config import ModalityConfig
from robot.config.finetune_config import DataConfig
from robot.utils import write_config_to_json
from robot.data.state_action_processor import StateActionProcessor
from robot.data.image_processor import ImageProcessor
from transformers import AutoProcessor, BatchFeature
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import List, Dict, Any
import torch


class DatasetCollator:
    """
    Collator for one batch of dataset.
    """
    def __init__(self, config: DataConfig):

        self.processor = AutoProcessor.from_pretrained(config.vlm_processor_path)
        self.processor.tokenizer.padding_side = "right"
        self.mask_ratio = config.mask_ratio

    def __call__(self, features: List[Dict[str, any]]) -> BatchFeature:
        """
        Apply random mask to actions from VLM inputs.
        :param features: List of VLM inputs which contains text, images, question_length.
        :return: Integrate features into a batch. 
        """
        text_ls = [element['text'] for element in features]
        images_ls = [element['image'] for element in features]
        qus_len_ls = [element['qus_len'] for element in features]

        # Integrate text and images into a batch
        vlm_inputs = self.processor(
            text=text_ls, images=images_ls, return_tensors="pt", padding=True
        )

        labels = vlm_inputs["input_ids"].clone()
        labels[labels == 151643] = -100
        batch_size, seq_len = labels.shape
        qus_len_tensor = torch.tensor(qus_len_ls)

        # Create mask for actions
        pos_id = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        action_mask = pos_id > qus_len_tensor.unsqueeze(1)
        # element equal to -100 will not calculate the loss
        labels[~action_mask] = -100
        
        # Create random mask
        random_values = torch.rand((batch_size, seq_len))
        action_mask &= random_values < self.mask_ratio

        # Apply mask into labels and input_ids
        labels[action_mask] = -100
        vlm_inputs["input_ids"][action_mask] = 30 # element equal to 30 represent '?'

        # Reshape vlm inputs to batch shape
        vlm_inputs["input_ids"] = vlm_inputs["input_ids"].view(batch_size, 1, seq_len)
        vlm_inputs["attention_mask"] = vlm_inputs["attention_mask"].view(batch_size, 1, seq_len)
        vlm_inputs["pixel_values"] = vlm_inputs["pixel_values"].view(batch_size, -1, vlm_inputs["pixel_values"].size(1))
        vlm_inputs["image_grid_thw"] = vlm_inputs["image_grid_thw"].view(batch_size, -1, vlm_inputs["image_grid_thw"].size(1))
        vlm_inputs["labels"] = labels.view(batch_size, 1, seq_len)

        return BatchFeature(data={**vlm_inputs})


class Transformer:
    """
    Actions and states will be normalized by std and divided into 0-num_bin_actions.
    Vision will be clipped and resized.
    And then vision and language will be processed by Qwen3-VL-Processor.
    """
    def __init__(
        self,
        modality_config: dict[str, ModalityConfig],
        statistics: dict | None = None,
        clip_outliers: bool = True,
        input_shape: tuple[int, int] | None = None,
        image_resize: tuple[int, int] | None = None,
        crop_fraction: float = 0.95,
        color_jitter: bool = True,
        mask_ratio: float = 0.5,
        num_bin_actions: int = 1000,
        is_train: bool = True,
        vlm_processor_path: str = "",
        predict_action_nums: int = 10,
        action_dimensions: dict | None = None,
        action_name: str = "action",
        config_output_dir: str | None = None,
    ):
        self.modality_config = modality_config
        self.statistics = statistics
        self.clip_outliers = clip_outliers
        self.input_shape = input_shape
        self.image_resize = image_resize
        self.crop_fraction = crop_fraction
        self.color_jitter = color_jitter
        self.mask_ratio = mask_ratio
        self.num_bin_actions = num_bin_actions
        self.is_train = is_train
        self.vlm_processor_path = vlm_processor_path
        self.predict_action_nums = predict_action_nums
        self.action_dimensions = action_dimensions
        self.action_name = action_name
        self.config_output_dir = config_output_dir

        # Create processor of states and actions
        self.state_action_processor = StateActionProcessor(
            modality_config=modality_config[action_name],
            statistics=statistics,
            clip_outliers=clip_outliers,
            num_bin_actions=num_bin_actions,
            predict_action_nums=predict_action_nums,
            action_dimensions=action_dimensions,
            action_name=action_name,
        )

        # Create processor of images
        self.image_processor = ImageProcessor(
            input_shape=input_shape,
            image_resize=image_resize,
            crop_fraction=crop_fraction,
            color_jitter=color_jitter,
            is_train=is_train,
        )

        # Create processor of language and images
        self.cache_question_lens = {}
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_processor_path)
        self.system_instruction = self._get_system_instruction()
        self.mask_ratio = mask_ratio


    def _get_system_instruction(self) -> str:
        """Make the system instruction available for Qwen3-VL-Processor."""
        action_dim = sum(self.action_dimensions.values())
        system_instruction = (f"Analyze the input images and prompt to predict the joint values of the robot for the next {self.predict_action_nums} timestamps. "
                              f"Each timestamp is a one-dimensional list with a size of {action_dim}, "
                              f"with all joint values ranging from 0 to {self.num_bin_actions}. Only output the result, no other content."
                              )
        return system_instruction

    def set_statistics(self, statistics: dict[str, Any]):
        """Set transformer statistics from multiple dataset."""
        self.statistics = statistics
        self.state_action_processor.set_statistics(statistics)
        self.save_config()

    def save_config(self):
        """Save configurations to a json file."""
        transformer_config = {
            "modality_config": self.modality_config,
            "clip_outliers": self.clip_outliers,
            "input_shape": self.input_shape,
            "image_resize": self.image_resize,
            "crop_fraction": self.crop_fraction,
            "color_jitter": self.color_jitter,
            "mask_ratio": self.mask_ratio,
            "num_bin_actions": self.num_bin_actions,
            "is_train": self.is_train,
            "vlm_processor_path": self.vlm_processor_path,
            "predict_action_nums": self.predict_action_nums,
            "action_dimensions": self.action_dimensions,
            "action_name": self.action_name,
        }
        write_config_to_json(transformer_config, os.path.join(self.config_output_dir, "transformer_config.json"))


    def _apply_vlm_processing(self, images: list[Image.Image], language: str, action: str) -> dict:
        """ Convert action into text, and cat it with language, finally apply qwen3-vl template for language and images."""
        conversation = [
            {
                'role': "system",
                'content': [{"type": "text", "text": self.system_instruction}],
            },
            {
                'role': "user",
                'content': [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": language}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action}],
            }
        ]

        # Apply chat template
        vlm_text = self.vlm_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        vlm_image = process_vision_info(conversation)[0]

        # Get the length of tokens from chat exclude assistant
        if language not in self.cache_question_lens:
            question_text = self.vlm_processor.apply_chat_template(
                conversation[:-1], tokenize=False, add_generation_prompt=False
            )
            question_tokens = self.vlm_processor(
                            text=question_text,
                            images=vlm_image,
                            return_tensors="pt",
                            padding=True,
            )["input_ids"]
            question_length = question_tokens.shape[1] + 3 # add length of tokens("role": "assistant")
            self.cache_question_lens[language] = question_length
        else:
            question_length = self.cache_question_lens[language]

        return {
            'text': vlm_text,
            'image': vlm_image,
            'qus_len': question_length,
        }


    def _apply_vlm_inference(self, images: list[Image.Image], language: str, action: dict[str, np.ndarray] | None) -> dict:
        """
        :return:
            question: the content of system and user
            answer: the content of assistant
        """
        conversation = [
            {
                'role': "system",
                'content': [{"type": "text", "text": self.system_instruction}],
            },
            {
                'role': "user",
                'content': [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": language}],
            }
        ]

        # Apply chat template
        vlm_text = self.vlm_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        vlm_image = process_vision_info(conversation)[0]

        vlm_inputs = {
            'text': vlm_text,
            'image': vlm_image,
        }
        if action is not None:
            vlm_inputs['action'] = action

        return vlm_inputs

    def __call__(self, inputs: dict) -> dict:
        """
        Just for single sample, apply difference transformer into different modalities.
        """
        normalize_action = self.state_action_processor.apply_action(
            action=inputs[self.action_name],
        ) if self.is_train else inputs[self.action_name]

        assert "video" in self.modality_config, "modality config must have 'video' key"
        normalize_images = self.image_processor.apply(images=inputs["video"])

        assert "language" in self.modality_config, "modality config must have 'language'"

        if self.is_train:
            return self._apply_vlm_processing(
                images=normalize_images,
                language=inputs["language"],
                action=normalize_action
            )
        return self._apply_vlm_inference(
            images=normalize_images,
            language=inputs["language"],
            action=normalize_action)

    def apply(self, inputs: dict):
        """Apply transformer into different modalities."""
        # Ready images and language
        normalize_images = self.image_processor.apply(images=inputs["video"])
        conversation = [
            {
                'role': "system",
                'content': [{"type": "text", "text": self.system_instruction}],
            },
            {
                'role': "user",
                'content': [{"type": "image", "image": image} for image in normalize_images]
                + [{"type": "text", "text": inputs["language"]}],
            }
        ]
        # Apply processor for vision and language
        inputs = self.vlm_processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        return inputs