from transformers import Qwen3VLForConditionalGeneration, AutoModel
from robot.config.finetune_config import ModelConfig
import torch
import logging
from typing import Dict
import torch.nn.functional as F
from transformers import PreTrainedModel
import torch.cuda.nvtx as nvtx
import time


class Qwen3VLA(PreTrainedModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._setup_model()

    def _setup_model(self):
        nvtx.range_push("Load model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device.type == "cuda", "GPUs is not supported."
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path, dtype=self.config.dtype, device_map=device
        )

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        logging.info(f"----------------------------Model loaded----------------------------")
        logging.info(f"[Model loaded] Path of loaded model: {self.config.model_path}")
        # Activate parameters according configuration
        # Use Lora for training model
        if self.config.lora_rank > 1:
            from peft import LoraConfig, get_peft_model

            # Only train part of language module for lora tune as default
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_config)
            logging.info(f"[Model loaded] Using LORA model for training, modules of lora contains: {target_modules}")
        # Only training partly parameters of model
        else:
            if self.config.tune_llm:
                for param in model.model.language_model.parameters():
                    param.requires_grad = True
                logging.info(f"[Model loaded] Using partly parameters for training which contains: Language modules")
            if self.config.tune_visual:
                for param in model.model.visual.parameters():
                    param.requires_grad = True
                logging.info(f"[Model loaded] Using partly parameters for training which contains: Visual modules")

        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"[Model loaded] Total params: {total_params:,}")
        logging.info(f"[Model loaded] Total trainable params: {total_trainable_params:,}, training radio: {total_trainable_params / total_params * 100:.2f}%")

        nvtx.range_pop()
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:

        nvtx.range_push("Resize parameters")
        seq_len = input_ids.size(-1)
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        pixel_values = pixel_values.view(-1, pixel_values.size(-1))
        image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        nvtx.range_pop()

        nvtx.range_push("Qwen forward")
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits
        nvtx.range_pop()

        # Not for calculating loss
        if labels is None:
            return {'logits': logits}

        nvtx.range_push("Calculate loss")
        labels = labels.view(-1, seq_len)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1).to(logits.device),
            ignore_index=-100
        )
        nvtx.range_pop()
        return {'loss': loss}
        
