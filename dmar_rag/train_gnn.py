import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .gnn_similarity import SiameseGNN


def load_pairs(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            g1 = obj.get("g1", {"nodes": [], "edges": []})
            g2 = obj.get("g2", {"nodes": [], "edges": []})
            y = float(obj.get("label", 0.0))
            pairs.append((g1, g2, y))
    return pairs


def train(weights_out: str, data_path: str, epochs: int = 3, lr: float = 1e-3, batch_size: int = 16):
    model = SiameseGNN()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    pairs = load_pairs(data_path)
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
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    train(args.out, args.data, epochs=args.epochs, lr=args.lr, batch_size=args.batch)


if __name__ == "__main__":
    main()