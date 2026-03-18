import numpy as np
import hnswlib


class ANNIndex:
    def __init__(self, dim: int, space: str = "cosine"):
        self.index = hnswlib.Index(space=space, dim=dim)
        self.labels = []
        self._initialized = False

    def build(self, embeddings: np.ndarray, ef_construction: int = 200, M: int = 32, ef_query: int = 200):
        num_elements = embeddings.shape[0]
        self.index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        labels = np.arange(num_elements)
        self.index.add_items(embeddings, labels)
        self.labels = labels
        self.index.set_ef(ef_query)
        self._initialized = True

    def query(self, vec: np.ndarray, top_k: int = 10):
        assert self._initialized
        labels, distances = self.index.knn_query(vec, k=top_k)
        return labels[0].tolist(), distances[0].tolist()