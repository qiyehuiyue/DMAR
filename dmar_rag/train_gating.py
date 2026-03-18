import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .gating_mlp import GatingMLP
from .kg import KGUtil


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _token_docs(ex):
    ctxs = ex.get("contexts") or ex.get("evidence") or []
    docs = []
    for i, c in enumerate(ctxs):
        docs.append({"id": i, "title": c.get("title", ""), "text": c.get("text", "")})
    return docs


def target_alpha(query: str, doc_text: str, kg: KGUtil) -> float:
    q_entities = kg.extract_entities(query)
    d_entities = kg.extract_entities(doc_text)
    overlap = len(q_entities.intersection(d_entities))
    if overlap > 0:
        return 0.6
    return 0.8


def train(dataset_jsonl: str, out_path: str, encoder_model: str = "princeton-nlp/sup-simcse-bert-base-uncased", device: str = "cpu", epochs: int = 3, batch_size: int = 64):
    encoder = SentenceTransformer(encoder_model, device=device)
    kg = KGUtil(use_spacy=True)
    samples = []
    for ex in _read_jsonl(dataset_jsonl):
        q = ex.get("question") or ex.get("claim") or ""
        docs = _token_docs(ex)
        qv = encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        for d in docs:
            dv = encoder.encode([d["text"]], convert_to_numpy=True, normalize_embeddings=True)[0]
            a = target_alpha(q, d["text"], kg)
            b = 1.0 - a
            samples.append((qv, dv, a, b))
    if not samples:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        model = GatingMLP(d_in=768*2)
        torch.save(model.state_dict(), out_path)
        return
    d_in = len(samples[0][0]) * 2
    model = GatingMLP(d_in=d_in)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        random.shuffle(samples)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            X = []
            Y = []
            for qv, dv, a, b in batch:
                x = torch.tensor(list(qv) + list(dv), dtype=torch.float32)
                y = torch.tensor([a, b], dtype=torch.float32)
                X.append(x)
                Y.append(y)
            X = torch.stack(X)
            Y = torch.stack(Y)
            pred = model.forward(X)
            loss = loss_fn(pred, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--encoder", type=str, default="princeton-nlp/sup-simcse-bert-base-uncased")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    train(dataset_jsonl=args.data, out_path=args.out, encoder_model=args.encoder, device=args.device, epochs=args.epochs, batch_size=args.batch)


if __name__ == "__main__":
    main()