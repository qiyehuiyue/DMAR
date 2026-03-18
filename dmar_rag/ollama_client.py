import requests
import numpy as np


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, texts, model: str):
        url = self.base_url + "/api/embed"
        payload = {"model": model, "input": texts}
        last = None
        for _ in range(3):
            try:
                r = requests.post(url, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                vecs = data.get("embeddings", [])
                return np.array(vecs, dtype=np.float32)
            except Exception as e:
                last = e
        raise last

    def generate(self, model: str, prompt: str):
        url = self.base_url + "/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        last = None
        for _ in range(3):
            try:
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "").strip()
            except Exception as e:
                last = e
        raise last