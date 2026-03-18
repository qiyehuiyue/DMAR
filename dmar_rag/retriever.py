import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import requests

from .kg import KGUtil
from .kg_store import KGStore
from .ann_index import ANNIndex
from .gating_mlp import GatingMLP
from .reranker import BERTCrossReranker


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class Document:
    id: str
    title: str
    text: str
    entities: List[str]


class DMARRetriever:
    def __init__(self, encoder_model: str = "princeton-nlp/sup-simcse-bert-base-uncased", device: str = "cpu", use_spacy: bool = True, encoder_backend: str = "sentence", ollama_url: str = None, gating_weights: str = None, shapley_T: int = 50, shapley_lambda: float = 0.5, min_subgraphs: int = 1, shapley_seed: int = None, gnn_weights: str = None, kg_store: KGStore = None, disable_kg: bool = False, fixed_gate_alpha: float = None, ann_M: int = 32, ann_ef_construction: int = 200, ann_ef_query: int = 200, cross_encoder_model: str = None, cross_weight: float = 0.0, kg_depth: int = 1):
        if encoder_backend == "ollama":
            self.encoder = _OllamaEncoder(ollama_url, encoder_model)
        else:
            self.encoder = SentenceTransformer(encoder_model, device=device)
        self.kg = KGUtil(use_spacy=use_spacy, gnn_weights=gnn_weights, store=kg_store)
        self.docs: List[Document] = []
        self.bm25 = None
        self.embeddings = None
        self.ann = None
        self.gating = None
        self._gating_weights_path = gating_weights
        self._warned_no_gating = False
        self._shapley_T = shapley_T
        self._shapley_lambda = shapley_lambda
        self._min_subgraphs = min_subgraphs
        self._shapley_seed = shapley_seed
        self._ann_M = ann_M
        self._ann_ef_construction = ann_ef_construction
        self._ann_ef_query = ann_ef_query
        self._kg_depth = kg_depth
        self._disable_kg = disable_kg
        self._fixed_gate_alpha = fixed_gate_alpha
        self.cross = None
        self._cross_weight = float(cross_weight or 0.0)
        if cross_encoder_model:
            self.cross = BERTCrossReranker(cross_encoder_model, device=device)

    def build_corpus(self, docs: List[Dict[str, Any]]):
        self.docs = []
        corpus_texts = []
        for i, d in enumerate(docs):
            text = d.get("text", "")
            title = d.get("title", "")
            ents = list(self.kg.link_entities(text))
            self.docs.append(Document(id=str(d.get("id", i)), title=title, text=text, entities=ents))
            corpus_texts.append(text)
        tokenized_corpus = [tuple(_tokenize(x)) for x in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.embeddings = self.encoder.encode(corpus_texts)
        if isinstance(self.embeddings, np.ndarray):
            dim = self.embeddings.shape[1]
        else:
            dim = len(self.embeddings[0])
            self.embeddings = np.array(self.embeddings, dtype=np.float32)
        self.ann = ANNIndex(dim=dim, space="cosine")
        self.ann.build(self.embeddings, ef_construction=self._ann_ef_construction, M=self._ann_M, ef_query=self._ann_ef_query)
        self.gating = GatingMLP(d_in=dim * 2)
        if self._gating_weights_path:
            try:
                self.gating.load(self._gating_weights_path)
            except Exception:
                self._gating_weights_path = None
                print("[dmar] Warning: failed to load gating weights; fallback to heuristic gating")

    def _semantic_scores(self, qv: np.ndarray, candidates_idx: List[int]) -> np.ndarray:
        dvs = self.embeddings[candidates_idx]
        return np.dot(dvs, qv)

    def _entity_scores(self, q_entities: set, candidates_idx: List[int]) -> np.ndarray:
        if self._disable_kg:
            return np.zeros(len(candidates_idx), dtype=np.float32)
        vals = []
        for i in candidates_idx:
            d_linked = self.kg.link_entities(self.docs[i].text)
            s = self.kg.shapley_score(q_entities, d_linked, T=self._shapley_T, lambda_=self._shapley_lambda, min_subgraphs=self._min_subgraphs, seed=self._shapley_seed, depth=self._kg_depth)
            vals.append(s)
        return np.array(vals, dtype=np.float32)

    def _gate_weights(self, qv: np.ndarray, candidates_idx: List[int], q_entities: set) -> List[Tuple[float, float]]:
        res = []
        if self._fixed_gate_alpha is not None:
            a = float(self._fixed_gate_alpha)
            b = float(1.0 - a)
            for _ in candidates_idx:
                res.append((a, b))
            return res
        if self.gating is not None and self._gating_weights_path:
            qv_t = torch.tensor(qv, dtype=torch.float32)
            for i in candidates_idx:
                dv = self.embeddings[i]
                dv_t = torch.tensor(dv, dtype=torch.float32)
                a, b = self.gating.weights(qv_t, dv_t)
                res.append((a, b))
            return res
        if not self._warned_no_gating:
            print("[dmar] Warning: gating weights not provided; using heuristic gate (overlap-based)")
            self._warned_no_gating = True
        for i in candidates_idx:
            overlap = len(q_entities.intersection(set(self.docs[i].entities)))
            if overlap > 0:
                res.append((0.6, 0.4))
            else:
                res.append((0.8, 0.2))
        return res

    def initial_recall(self, query: str, top_n: int = 100) -> List[int]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        idx = np.argsort(scores)[::-1][:top_n]
        return idx.tolist()

    def build_corpus_with_embeddings(self, docs: List[Dict[str, Any]], embeddings: np.ndarray):
        self.docs = []
        corpus_texts = []
        for i, d in enumerate(docs):
            text = d.get("text", "")
            title = d.get("title", "")
            ents = list(self.kg.link_entities(text))
            self.docs.append(Document(id=str(d.get("id", i)), title=title, text=text, entities=ents))
            corpus_texts.append(text)
        tokenized_corpus = [tuple(_tokenize(x)) for x in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.embeddings = embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings, dtype=np.float32)
        dim = self.embeddings.shape[1]
        self.ann = ANNIndex(dim=dim, space="cosine")
        self.ann.build(self.embeddings, ef_construction=self._ann_ef_construction, M=self._ann_M, ef_query=self._ann_ef_query)

    def select_anchors(self, query: str, candidates_idx: List[int], m: int = 3) -> List[int]:
        qv = self.encoder.encode([query])[0]
        sem = self._semantic_scores(qv, candidates_idx)
        q_entities = self.kg.link_entities(query)
        ent = self._entity_scores(q_entities, candidates_idx)
        sem_z = self._zscore_arr(sem)
        ent_z = self._zscore_arr(ent)
        gates = self._gate_weights(qv, candidates_idx, q_entities)
        fused = []
        for idx_i, (a, b) in zip(range(len(candidates_idx)), gates):
            idx = candidates_idx[idx_i]
            score = a * sem_z[idx_i] + b * ent_z[idx_i]
            fused.append((idx, float(score)))
        fused.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in fused[:m]]

    def _zscore_arr(self, vals: np.ndarray) -> np.ndarray:
        mu = float(np.mean(vals)) if vals.size > 0 else 0.0
        sigma = float(np.std(vals)) if vals.size > 0 else 1.0
        if sigma == 0.0:
            sigma = 1.0
        return (vals - mu) / sigma

    def reformulate(self, query: str, anchor_idx: List[int], reformer) -> str:
        anchors_text = [self.docs[i].text for i in anchor_idx]
        return reformer.rewrite(query, anchors_text)

    def second_recall(self, reformulated_query: str, top_p: int = 20) -> List[int]:
        qv = self.encoder.encode([reformulated_query])
        k = min(top_p, len(self.docs)) if self.docs else 0
        if k <= 0:
            return []
        labels, _ = self.ann.query(qv, top_k=k)
        return labels

    def rerank_final(self, original_query: str, candidates_idx: List[int], k: int = 5) -> List[int]:
        qv = self.encoder.encode([original_query])[0]
        sem = self._semantic_scores(qv, candidates_idx)
        q_entities = self.kg.link_entities(original_query)
        ent = self._entity_scores(q_entities, candidates_idx)
        sem_z = self._zscore_arr(sem)
        ent_z = self._zscore_arr(ent)
        gates = self._gate_weights(qv, candidates_idx, q_entities)
        fused = []
        for idx_i, (a, b) in zip(range(len(candidates_idx)), gates):
            idx = candidates_idx[idx_i]
            score = a * sem_z[idx_i] + b * ent_z[idx_i]
            fused.append((idx, float(score)))
        if self.cross is not None and self._cross_weight > 0.0:
            texts = [self.docs[i].text for i, _ in fused]
            ce = self.cross.score(original_query, texts)
            ce_z = self._zscore_arr(ce)
            fused = [(i, (1.0 - self._cross_weight) * s + self._cross_weight * float(ce_z[j])) for j, (i, s) in enumerate(fused)]
        fused.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in fused[:k]]

    def run(self, query: str, m: int = 3, n: int = 100, p: int = 20, k: int = 5, reformer=None, disable_reform: bool = False, single_anchor: bool = False) -> List[Document]:
        c1 = self.initial_recall(query, top_n=n)
        if single_anchor:
            m = 1
        anchors = self.select_anchors(query, c1, m=m)
        q_prime = query if (reformer is None or disable_reform) else self.reformulate(query, anchors, reformer)
        c2 = self.second_recall(q_prime, top_p=p)
        final_idx = self.rerank_final(query, c2, k=k)
        return [self.docs[i] for i in final_idx]


class _OllamaEncoder:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def encode(self, texts: List[str]):
        url = self.base_url + "/api/embed"
        payload = {"model": self.model_name, "input": texts}
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        vecs = data.get("embeddings", [])
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr
