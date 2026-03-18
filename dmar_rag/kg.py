import re
import random
from typing import List, Set, Dict
from .gnn_similarity import SiameseGNN
from .kg_store import KGStore

try:
    import spacy
    _SPACY = True
except Exception:
    _SPACY = False


class KGUtil:
    def __init__(self, use_spacy: bool = True, gnn_weights: str = None, store: KGStore = None):
        self.use_spacy = use_spacy and _SPACY
        self.nlp = spacy.load("en_core_web_sm") if self.use_spacy else None
        self.gnn = SiameseGNN(weights_path=gnn_weights)
        self.store = store or KGStore()
        self._link_cache: Dict[str, Set[str]] = {}

    def extract_entities(self, text: str) -> Set[str]:
        if not text:
            return set()
        if self.nlp is not None:
            doc = self.nlp(text)
            return {ent.text.strip() for ent in doc.ents if ent.text.strip()}
        tokens = re.findall(r"[A-Z][a-zA-Z0-9_\-]+", text)
        return set(tokens)

    def link_entities(self, text: str) -> Set[str]:
        key = text.strip()
        if key in self._link_cache:
            return self._link_cache[key]
        raw = self.extract_entities(text)
        linked = set()
        for e in raw:
            lid = self.store.link(e)
            if not lid:
                lid = self.store.link_fuzzy(e)
            linked.add(lid or e)
        self._link_cache[key] = linked
        return linked

    def extract_subgraphs(self, entities: Set[str], depth: int = 1) -> List[Dict]:
        g = self.store.build_subgraph(entities, depth=depth)
        if not g.get("nodes"):
            res = []
            for e in entities:
                res.append({"nodes": [e], "edges": []})
            return res
        return [g]

    def shapley_score(self, q_entities: Set[str], d_entities: Set[str], T: int = 50, lambda_: float = 0.5, min_subgraphs: int = 1, seed: int = None, depth: int = 1) -> float:
        if not q_entities:
            return 0.0
        GQ = self.extract_subgraphs(q_entities, depth=depth)
        GD = self.extract_subgraphs(d_entities, depth=depth)
        if not GQ or len(GQ) < min_subgraphs or not GD:
            return 0.0
        total = 0.0
        rnd = random.Random(seed) if seed is not None else random
        for g in GQ:
            phi = 0.0
            for _ in range(T):
                phi += self._marginal_contribution(g, GQ, GD, lambda_, rnd)
            total += phi / float(T)
        return total / float(len(GQ))

    def _coverage(self, S: List[Dict], GD: List[Dict]) -> float:
        nodes_S = set()
        for s in S:
            if isinstance(s, dict):
                nodes_S |= set(s.get("nodes", []))
            elif isinstance(s, (set, list, tuple)):
                nodes_S |= set(s)
        if not nodes_S:
            return 0.0
        nodes_D = set()
        for d in GD:
            if isinstance(d, dict):
                nodes_D |= set(d.get("nodes", []))
            elif isinstance(d, (set, list, tuple)):
                nodes_D |= set(d)
        inter = nodes_S & nodes_D
        return float(len(inter)) / float(len(nodes_S))

    def _struct_sim(self, S: List[Dict], GD: List[Dict]) -> float:
        if not S or not GD:
            return 0.0
        total = 0.0
        for s in S:
            best = 0.0
            for d in GD:
                sim = self.gnn.similarity(s, d)
                if sim > best:
                    best = sim
            total += best
        return total / float(len(S))

    def _utility(self, S: List[Set[str]], GD: List[Set[str]], lambda_: float) -> float:
        cov = self._coverage(S, GD)
        sim = self._struct_sim(S, GD)
        return lambda_ * cov + (1.0 - lambda_) * sim

    def _marginal_contribution(self, g: Set[str], GQ: List[Set[str]], GD: List[Set[str]], lambda_: float, rnd) -> float:
        others = [x for x in GQ if x != g]
        rnd.shuffle(others)
        if not others:
            return self._utility([g], GD, lambda_)
        k = rnd.randint(0, len(others))
        S = others[:k]
        before = self._utility(S, GD, lambda_)
        after = self._utility(S + [g], GD, lambda_)
        return after - before