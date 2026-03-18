from typing import Dict, List, Tuple, Set


class KGStore:
    def __init__(self):
        self.alias: Dict[str, str] = {}
        self.alias_norm: Dict[str, str] = {}
        self.nodes: Set[str] = set()
        self.edges: List[Tuple[str, str]] = []
        self.neigh: Dict[str, List[str]] = {}

    def normalize(self, s: str) -> str:
        return s.strip()

    def add_alias(self, name: str, entity_id: str):
        n = self.normalize(name)
        self.alias[n] = entity_id
        self.alias_norm[n.lower()] = entity_id

    def add_triple(self, head: str, relation: str, tail: str):
        h = self.normalize(head)
        t = self.normalize(tail)
        self.nodes.add(h)
        self.nodes.add(t)
        self.edges.append((h, t))
        if h not in self.neigh:
            self.neigh[h] = []
        if t not in self.neigh:
            self.neigh[t] = []
        self.neigh[h].append(t)
        self.neigh[t].append(h)

    def link(self, name: str) -> str:
        n = self.normalize(name)
        if n in self.alias:
            return self.alias[n]
        ln = n.lower()
        if ln in self.alias_norm:
            return self.alias_norm[ln]
        if n in self.nodes:
            return n
        if ln in {x.lower() for x in self.nodes}:
            for x in self.nodes:
                if x.lower() == ln:
                    return x
        return ""

    def link_fuzzy(self, name: str) -> str:
        n = self.normalize(name)
        ln = n.lower()
        if ln in self.alias_norm:
            return self.alias_norm[ln]
        for a in self.alias_norm:
            if ln.startswith(a) or a.startswith(ln) or ln in a or a in ln:
                return self.alias_norm[a]
        for node in self.nodes:
            lnode = node.lower()
            if ln.startswith(lnode) or lnode.startswith(ln) or ln in lnode or lnode in ln:
                return node
        return ""

    def build_subgraph(self, entities: Set[str], depth: int = 1, max_nodes: int = 64) -> Dict:
        seeds = [self.link(e) for e in entities]
        seeds = [s for s in seeds if s]
        if not seeds:
            return {"nodes": [], "edges": []}
        visited: Set[str] = set()
        edges: List[Tuple[str, str]] = []
        frontier = list(seeds)
        visited.update(frontier)
        for _ in range(depth):
            next_frontier = []
            for u in frontier:
                if u in self.neigh:
                    for v in self.neigh[u]:
                        if len(visited) >= max_nodes:
                            break
                        edges.append((u, v))
                        if v not in visited:
                            visited.add(v)
                            next_frontier.append(v)
            frontier = next_frontier
            if len(visited) >= max_nodes:
                break
        return {"nodes": list(visited), "edges": edges}

    def load_jsonl(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                h = obj.get("head", "")
                r = obj.get("relation", "")
                t = obj.get("tail", "")
                if h and t:
                    self.add_triple(h, r, t)
                aliases = obj.get("aliases", {})
                for k, v in aliases.items():
                    self.add_alias(k, v)

    def load_wikidata_dump(self, path: str, max_entities: int = 10000):
        import json
        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "head" in obj and "tail" in obj:
                    h = obj.get("head", "")
                    t = obj.get("tail", "")
                    r = obj.get("relation", "")
                    if h and t:
                        self.add_triple(h, r, t)
                    aliases = obj.get("aliases", {})
                    for k, v in aliases.items():
                        self.add_alias(k, v)
                    cnt += 1
                    if cnt >= max_entities:
                        break
                    continue
                ent_id = obj.get("id") or obj.get("entity")
                labels = obj.get("labels", {})
                if isinstance(labels, dict):
                    for lang, entry in labels.items():
                        name = entry.get("value") if isinstance(entry, dict) else None
                        if name:
                            self.add_alias(name, ent_id or name)
                aliases = obj.get("aliases", {})
                if isinstance(aliases, dict):
                    for lang, arr in aliases.items():
                        for entry in arr if isinstance(arr, list) else []:
                            name = entry.get("value") if isinstance(entry, dict) else None
                            if name:
                                self.add_alias(name, ent_id or name)
                claims = obj.get("claims", {})
                for pid, arr in claims.items():
                    for cl in arr if isinstance(arr, list) else []:
                        mainsnak = cl.get("mainsnak", {})
                        dv = mainsnak.get("datavalue", {})
                        val = dv.get("value", {})
                        if isinstance(val, dict) and val.get("id"):
                            tail_id = val.get("id")
                            head_id = ent_id or tail_id
                            self.add_triple(str(head_id), str(pid), str(tail_id))
                cnt += 1
                if cnt >= max_entities:
                    break

    def load_wikidata_sparql(self, endpoint: str, query: str, limit: int = 1000):
        import requests
        import urllib.parse
        params = {"query": query, "format": "json"}
        url = endpoint
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        bindings = data.get("results", {}).get("bindings", [])
        count = 0
        for b in bindings:
            h = b.get("head") or b.get("item")
            t = b.get("tail") or b.get("value")
            rlbl = b.get("relation") or b.get("prop")
            head = h.get("value") if isinstance(h, dict) else None
            tail = t.get("value") if isinstance(t, dict) else None
            rel = rlbl.get("value") if isinstance(rlbl, dict) else ""
            if head and tail:
                self.add_triple(head, rel, tail)
                count += 1
                if count >= limit:
                    break