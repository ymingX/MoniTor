import os
from typing import List, Optional

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


class Qwen3VLClient:
    def __init__(self, model_path: Optional[str] = None, dtype_str: str = "float16"):
        resolved_path = model_path or os.getenv("MODEL_PATH")
        if not resolved_path:
            raise ValueError("MODEL_PATH is not set. Please set MODEL_PATH to your local Qwen3-VL-4B directory.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dtype_str == "float16" else torch.float32

        self.processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            resolved_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def generate_text(self, messages: List[dict], max_new_tokens: int = 128) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def generate_caption(self, image, prompt: str = "Describe the image in one sentence.", max_new_tokens: int = 64) -> str:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
