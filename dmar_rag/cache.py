import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


class EmbeddingDiskCache:
    def __init__(self, cache_dir: str = ".cache/dmar_rag"):
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._enabled = True
        except Exception:
            # Fall back to disabled cache if the filesystem is read-only or unavailable
            self._enabled = False

    def _key(self, texts: List[str], encoder_id: str) -> str:
        h = hashlib.sha1()
        h.update(encoder_id.encode("utf-8"))
        for t in texts:
            h.update(str(len(t)).encode("utf-8"))
            h.update(t.encode("utf-8"))
        return h.hexdigest()

    def get(self, texts: List[str], encoder_id: str) -> Optional[np.ndarray]:
        if not self._enabled:
            return None
        key = self._key(texts, encoder_id)
        path = self.cache_dir / f"emb_{key}.npz"
        if path.exists():
            data = np.load(str(path))
            return data["embeddings"]
        return None

    def set(self, texts: List[str], encoder_id: str, embeddings: np.ndarray):
        if not self._enabled:
            return
        key = self._key(texts, encoder_id)
        path = self.cache_dir / f"emb_{key}.npz"
        np.savez_compressed(str(path), embeddings=embeddings)
