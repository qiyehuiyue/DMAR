import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .gnn_similarity import SiameseGNN
from .kg_store import KGStore


def sample_pairs(store: KGStore, pos_ratio: float = 0.5, total: int = 5000):
    nodes = list(store.nodes)
    edges = list(store.edges)
    neigh = store.neigh
    pairs = []
    for _ in range(total):
        if random.random() < pos_ratio and edges:
            h, t = random.choice(edges)
            g1 = {"nodes": [h] + neigh.get(h, [])[:3], "edges": [(h, x) for x in neigh.get(h, [])[:3]]}
            g2 = {"nodes": [t] + neigh.get(t, [])[:3], "edges": [(t, x) for x in neigh.get(t, [])[:3]]}
            y = 1.0
        else:
            h = random.choice(nodes) if nodes else ""
            t = random.choice(nodes) if nodes else ""
            if not h or not t or h == t:
                continue
            g1 = {"nodes": [h] + neigh.get(h, [])[:2], "edges": [(h, x) for x in neigh.get(h, [])[:2]]}
            g2 = {"nodes": [t] + neigh.get(t, [])[:2], "edges": [(t, x) for x in neigh.get(t, [])[:2]]}
            y = 0.0
        pairs.append((g1, g2, y))
    return pairs


def train_with_kg(kg_jsonl: str, weights_out: str, epochs: int = 3, lr: float = 1e-3, batch_size: int = 32, total_pairs: int = 10000):
    store = KGStore()
    store.load_jsonl(kg_jsonl)
    pairs = sample_pairs(store, pos_ratio=0.5, total=total_pairs)
    model = SiameseGNN()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            preds = []
            labels = []
            for g1, g2, y in batch:
                s = model.similarity(g1, g2)
                preds.append(s)
                labels.append(y)
            p = torch.tensor(preds, dtype=torch.float32)
            l = torch.tensor(labels, dtype=torch.float32)
            loss = loss_fn(p, l)
            opt.zero_grad()
            loss.backward()
            opt.step()
    Path(weights_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-jsonl", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--pairs", type=int, default=10000)
    args = p.parse_args()
    train_with_kg(args.kg_jsonl, args.out, epochs=args.epochs, lr=args.lr, batch_size=args.batch, total_pairs=args.pairs)


if __name__ == "__main__":
    main()