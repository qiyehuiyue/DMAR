import numpy as np
from sentence_transformers import CrossEncoder


class BERTCrossReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, docs_text: list) -> np.ndarray:
        pairs = [(query, t) for t in docs_text]
        s = self.model.predict(pairs)
        return np.array(s, dtype=np.float32)