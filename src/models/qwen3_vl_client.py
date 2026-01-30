import os
from typing import List, Optional

import torch
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3VLClient:
    def __init__(self, model_path: Optional[str] = None, dtype_str: str = "bfloat16"):
        resolved_path = model_path or os.getenv("MODEL_PATH")
        if not resolved_path:
            raise ValueError("MODEL_PATH is not set. Please set MODEL_PATH to your local Qwen3-VL-4B directory.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            resolved_path, dtype=dtype, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(resolved_path)

    def _build_inputs(self, messages: List[dict]):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _decode_output(self, inputs, output_ids):
        output_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_text(self, messages: List[dict], max_new_tokens: int = 128) -> str:
        inputs = self._build_inputs(messages)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._decode_output(inputs, output_ids)

    def generate_caption(
        self,
        image_path: str,
        prompt: str = "一句话描述摄像头画面中的场景和事件.",
        max_new_tokens: int = 64,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._build_inputs(messages)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._decode_output(inputs, output_ids)
