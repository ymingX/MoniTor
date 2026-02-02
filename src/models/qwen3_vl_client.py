import os
from typing import List, Optional

import requests


class Qwen3VLClient:
    def __init__(self, model_path: Optional[str] = None, dtype_str: str = "bfloat16"):
        self.server = os.getenv("VLLM_SERVER", "http://127.0.0.1:8001")
        print(self.server)
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.timeout_s = int(os.getenv("VLLM_TIMEOUT", "600"))
        self.auto_start = os.getenv("VLLM_AUTO_START", "0") == "1"
        # if self.auto_start and self.model_path:
            # self._ensure_model_loaded()
        self._ensure_model_loaded()

    def _to_absolute_path(self, path: Optional[str]) -> Optional[str]:
        """将路径转换为绝对路径"""
        if path is None:
            return None
        if isinstance(path, str) and path.strip():
            return os.path.abspath(path)
        return path

    def _post(self, path: str, payload: dict, timeout: Optional[int] = None):
        r = requests.post(f"{self.server}{path}", json=payload, timeout=timeout or self.timeout_s,
                        proxies={"http": None, "https": None}  # 禁用代理  
                        )
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, timeout: Optional[int] = None):
        r = requests.get(f"{self.server}{path}", timeout=timeout or 30, proxies={"http": None, "https": None} )
        r.raise_for_status()
        return r.json()

    def _ensure_model_loaded(self):
        status = self._get("/status")
        if not status.get("loaded") or (self.model_path and status.get("model_path") != self.model_path):
            # 将 model_path 也转换为绝对路径
            abs_model_path = self._to_absolute_path(self.model_path)
            self._post("/start", {"model_path": abs_model_path})

    def _messages_to_text(self, messages: List[dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                content = " ".join(parts)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def generate_text(self, messages: List[dict], max_new_tokens: int = 128) -> str:
        text = self._messages_to_text(messages)
        resp = self._post(
            "/single",
            {
                "text": text,
                "image": None,
                "max_tokens": int(max_new_tokens),
                "temperature": 0.0,
                "return_perf": False,
            },
        )
        return resp.get("text", "")

    def generate_text_batch(self, messages_list: List[List[dict]], max_new_tokens: int = 128) -> List[str]:
        items = [{"text": self._messages_to_text(m), "image": None} for m in messages_list]
        resp = self._post(
            "/single_batch",
            {
                "items": items,
                "max_tokens": int(max_new_tokens),
                "temperature": 0.0,
                "return_perf": False,
            },
        )
        results = resp.get("results", [])
        return [r.get("output_text", "") for r in results]

    def generate_caption(
        self,
        image_path: str,
        prompt: str = "一句话描述摄像头画面中的场景和事件.",
        max_new_tokens: int = 64,
    ) -> str:
        # 转换为绝对路径
        abs_image_path = self._to_absolute_path(image_path)
        
        resp = self._post(
            "/single",
            {
                "text": prompt,
                "image": abs_image_path,
                "max_tokens": int(max_new_tokens),
                "temperature": 0.0,
                "return_perf": False,
            },
        )
        return resp.get("text", "")

    def generate_caption_batch(
        self,
        image_paths: List[str],
        prompt: str = "一句话描述摄像头画面中的场景和事件.",
        max_new_tokens: int = 64,
    ) -> List[str]:
        # 将所有图像路径转换为绝对路径
        abs_image_paths = [self._to_absolute_path(p) for p in image_paths]
        items = [{"text": prompt, "image": p} for p in abs_image_paths]
        
        resp = self._post(
            "/single_batch",
            {
                "items": items,
                "max_tokens": int(max_new_tokens),
                "temperature": 0.0,
                "return_perf": False,
            },
        )
        results = resp.get("results", [])
        return [r.get("output_text", "") for r in results]
