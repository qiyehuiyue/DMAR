import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from sentence_transformers import SentenceTransformer, CrossEncoder

from .gating_mlp import GatingMLP
from .kg import KGUtil


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _docs_from_ex(ex):
    ctxs = ex.get("contexts") or ex.get("evidence") or []
    docs = []
    for i, c in enumerate(ctxs):
        docs.append({"id": i, "title": c.get("title", ""), "text": c.get("text", "")})
    return docs


def _answers(ex):
    a = ex.get("answers") or []
    if isinstance(a, list):
        return [str(x).strip().lower() for x in a if isinstance(x, str)]
    return []


def build_samples(path: str, enc: SentenceTransformer, kg: KGUtil, shapley_T: int, shapley_lambda: float, min_subgraphs: int, depth: int):
    samples = []
    for ex in _read_jsonl(path):
        q = ex.get("question") or ex.get("claim") or ""
        docs = _docs_from_ex(ex)
        if not q or not docs:
            continue
        qv = enc.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_ids = kg.link_entities(q)
        dvs = []
        sems = []
        ents = []
        texts = []
        for d in docs:
            dv = enc.encode([d["text"]], convert_to_numpy=True, normalize_embeddings=True)[0]
            dvs.append(dv)
            sems.append(float(np.dot(qv, dv)))
            d_ids = kg.link_entities(d["text"])
            ents.append(kg.shapley_score(q_ids, d_ids, T=shapley_T, lambda_=shapley_lambda, min_subgraphs=min_subgraphs, depth=depth))
            texts.append(d["text"])
        sems = np.array(sems, dtype=np.float32)
        ents = np.array(ents, dtype=np.float32)
        samples.append((q, qv, dvs, sems, ents, texts))
    return samples


def train_joint(data_path: str, out_path: str, encoder_model: str, cross_model: str, device: str = "cpu", epochs: int = 2, batch_size: int = 32, shapley_T: int = 10, shapley_lambda: float = 0.5, min_subgraphs: int = 1, depth: int = 1, loss_alpha: float = 0.7):
    enc = SentenceTransformer(encoder_model, device=device)
    ce = CrossEncoder(cross_model, device=device)
    kg = KGUtil(use_spacy=True)
    data = build_samples(data_path, enc, kg, shapley_T, shapley_lambda, min_subgraphs, depth)
    if not data:
        model = GatingMLP(d_in=1536)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        return
    d_in = len(data[0][1]) * 2
    model = GatingMLP(d_in=d_in)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        random.shuffle(data)
        for ex in data:
            q, qv, dvs, sems, ents, texts = ex
            qv_t = torch.tensor(qv, dtype=torch.float32)
            ce_scores = ce.predict([(q, t) for t in texts])
            ce_scores = np.array(ce_scores, dtype=np.float32)
            # z-score
            def z(x):
                mu = float(np.mean(x)); sd = float(np.std(x)); sd = sd if sd != 0.0 else 1.0
                return (x - mu) / sd
            sem_z = z(sems)
            ent_z = z(ents)
            ce_z = z(ce_scores)
            fused = []
            for dv, sz, ez in zip(dvs, sem_z, ent_z):
                dv_t = torch.tensor(dv, dtype=torch.float32)
                a, b = model.weights(qv_t, dv_t)
                fused.append(a * sz + b * ez)
            fused_t = torch.tensor(fused, dtype=torch.float32)
            ce_t = torch.tensor(ce_z, dtype=torch.float32)
            # distillation loss + pairwise ranking to match CE ordering
            mse = F.mse_loss(fused_t, ce_t)
            rank_loss = 0.0
            for i in range(len(fused) - 1):
                for j in range(i + 1, len(fused)):
                    diff_student = fused_t[i] - fused_t[j]
                    diff_teacher = ce_t[i] - ce_t[j]
                    # logistic ranking aligning signs and margins
                    rank_loss = rank_loss + F.softplus(-(diff_student * torch.sign(diff_teacher)))
            rank_loss = rank_loss / max(len(fused) * (len(fused) - 1) / 2, 1)
            loss = loss_alpha * mse + (1.0 - loss_alpha) * rank_loss
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
    p.add_argument("--cross", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--shapley-T", type=int, default=10)
    p.add_argument("--shapley-lambda", type=float, default=0.5)
    p.add_argument("--min-subgraphs", type=int, default=1)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--loss-alpha", type=float, default=0.7)
    args = p.parse_args()
    train_joint(data_path=args.data, out_path=args.out, encoder_model=args.encoder, cross_model=args.cross, device=args.device, epochs=args.epochs, batch_size=args.batch, shapley_T=args.shapley_T, shapley_lambda=args.shapley_lambda, min_subgraphs=args.min_subgraphs, depth=args.depth, loss_alpha=args.loss_alpha)


if __name__ == "__main__":
    main()