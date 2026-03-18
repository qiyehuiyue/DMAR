import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: list) -> torch.Tensor:
        neigh = []
        for i in range(len(adj)):
            idx = adj[i]
            if len(idx) == 0:
                neigh.append(torch.zeros_like(h[i]))
            else:
                neigh.append(h[idx].mean(dim=0))
        neigh = torch.stack(neigh, dim=0)
        out = self.self_lin(h) + self.neigh_lin(neigh)
        return F.relu(out)


class SiameseGNN(nn.Module):
    def __init__(self, vocab_dim: int = 50000, embed_dim: int = 64, hidden_dim: int = 64, num_layers: int = 3, weights_path: str = None):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.embed = nn.Embedding(vocab_dim, embed_dim)
        layers = []
        in_dim = embed_dim
        for _ in range(num_layers):
            layers.append(GraphSAGELayer(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(hidden_dim, hidden_dim)
        self._loaded = False
        if weights_path:
            try:
                state = torch.load(weights_path, map_location="cpu")
                self.load_state_dict(state)
                self._loaded = True
            except Exception:
                self._loaded = False

    def _hash_ids(self, nodes: list) -> list:
        ids = []
        for n in nodes:
            h = abs(hash(str(n))) % self.vocab_dim
            ids.append(h)
        return ids

    def _adjacency(self, nodes: list, edges: list) -> list:
        idx_map = {n: i for i, n in enumerate(nodes)}
        adj = [[] for _ in nodes]
        for u, v in edges:
            if u in idx_map and v in idx_map:
                ui = idx_map[u]
                vi = idx_map[v]
                adj[ui].append(vi)
                adj[vi].append(ui)
        adj = [torch.tensor(a, dtype=torch.long) if len(a) > 0 else torch.tensor([], dtype=torch.long) for a in adj]
        return adj

    def encode(self, graph: dict) -> torch.Tensor:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        if not nodes:
            return torch.zeros(self.readout.out_features)
        ids = torch.tensor(self._hash_ids(nodes), dtype=torch.long)
        h = self.embed(ids)
        adj = self._adjacency(nodes, edges)
        for layer in self.layers:
            h = layer(h, adj)
        g = h.mean(dim=0)
        g = self.readout(g)
        return F.normalize(g, dim=0)

    def similarity(self, g1: dict, g2: dict) -> float:
        v1 = self.encode(g1)
        v2 = self.encode(g2)
        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=-1)[0]
        return float(sim.item())