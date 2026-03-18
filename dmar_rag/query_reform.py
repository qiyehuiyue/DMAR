from typing import List

from transformers import pipeline
from .ollama_client import OllamaClient


class QueryReformer:
    def __init__(self, model_name: str = "t5-base", device: int = -1):
        self.pipe = pipeline("text2text-generation", model=model_name, device=device)

    def rewrite(self, query: str, anchors: List[str], max_new_tokens: int = 64) -> str:
        context = "\n\n".join(anchors[:3]) if anchors else ""
        prompt = (
            "Rewrite the query to improve retrieval. Keep intent consistent.\n"
            + "Query:" + query + "\n"
            + "Anchors:" + context
        )
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, num_beams=4)[0]["generated_text"].strip()
        return out


class OllamaReformer:
    def __init__(self, base_url: str, model_name: str):
        self.client = OllamaClient(base_url)
        self.model_name = model_name

    def rewrite(self, query: str, anchors: List[str], max_new_tokens: int = 64) -> str:
        context = "\n\n".join(anchors[:3]) if anchors else ""
        prompt = (
            "Rewrite the query to improve retrieval. Keep intent consistent.\n"
            + "Query:" + query + "\n"
            + "Anchors:" + context
        )
        return self.client.generate(self.model_name, prompt)