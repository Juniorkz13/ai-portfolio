# src/llm/model.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LLMClient:
    """
    Lightweight LLM client safe for local GPU (RTX 2060).
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 256,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens

        print(f"[LLM] Loading model on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ModelRouter:
    """
    Single-model router (safe for local GPU).
    """

    def __init__(self):
        self.model = LLMClient()

    def run(self, prompt: str, task: str = "qa") -> str:
        return self.model.generate(prompt)
