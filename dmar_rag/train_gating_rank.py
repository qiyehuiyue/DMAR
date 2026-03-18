import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

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


def _answers_from_ex(ex):
    a = ex.get("answers") or []
    if isinstance(a, list):
        return [str(x).strip().lower() for x in a if isinstance(x, str)]
    return []


def _is_pos(doc_title: str, doc_text: str, ex: dict) -> bool:
    title_l = (doc_title or "").lower()
    doc_l = (doc_text or "").lower()
    answers = _answers_from_ex(ex)
    for ans in answers:
        if ans and (ans in doc_l or ans == title_l):
            return True
    label = str(ex.get("label", "")).strip().lower()
    if label and label != "notenoughinfo":
        ev = ex.get("evidence") or []
        ev_titles = set()
        for e in ev:
            if isinstance(e, dict):
                t = str(e.get("title", "")).lower()
                if t:
                    ev_titles.add(t)
        if title_l in ev_titles:
            return True
        claim = str(ex.get("claim", "")).lower()
        tokens = [t for t in claim.split() if t]
        for t in tokens:
            if t and t in doc_l:
                return True
    return False

def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma == 0.0:
        sigma = 1.0
    return (x - mu) / sigma


def train_rank(dataset_jsonl: str, out_path: str, encoder_model: str = "princeton-nlp/sup-simcse-bert-base-uncased", device: str = "cpu", epochs: int = 3, batch_size: int = 64, margin: float = 0.1, shapley_T: int = 10, shapley_lambda: float = 0.5, min_subgraphs: int = 1, shapley_seed: int = None, loss_type: str = "softplus", hard_neg: bool = True):
    enc = SentenceTransformer(encoder_model, device=device)
    kg = KGUtil(use_spacy=True)
    pairs = []
    for ex in _read_jsonl(dataset_jsonl):
        q = ex.get("question") or ex.get("claim") or ""
        if not q:
            continue
        docs = _docs_from_ex(ex)
        if not docs:
            continue
        qv = enc.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        dvs = []
        for d in docs:
            dv = enc.encode([d["text"]], convert_to_numpy=True, normalize_embeddings=True)[0]
            dvs.append(dv)
        sems = np.array([float(np.dot(qv, dv)) for dv in dvs], dtype=np.float32)
        q_entities = kg.extract_entities(q)
        ents_vals = []
        for d in docs:
            d_entities = kg.extract_entities(d["text"])
            s = kg.shapley_score(q_entities, d_entities, T=shapley_T, lambda_=shapley_lambda, min_subgraphs=min_subgraphs, seed=shapley_seed)
            ents_vals.append(s)
        ents = np.array(ents_vals, dtype=np.float32)
        sems_z = _zscore(sems)
        ents_z = _zscore(ents)
        pos_idx = [i for i, d in enumerate(docs) if _is_pos(d.get("title", ""), d.get("text", ""), ex)]
        neg_idx = [i for i in range(len(docs)) if i not in pos_idx]
        if not pos_idx or not neg_idx:
            continue
        if hard_neg:
            neg_idx.sort(key=lambda i: sems[i], reverse=True)
        random.shuffle(pos_idx)
        m = min(len(pos_idx), len(neg_idx))
        for i in range(m):
            pi = pos_idx[i]
            ni = neg_idx[i]
            pairs.append((qv, dvs[pi], dvs[ni], sems_z[pi], sems_z[ni], ents_z[pi], ents_z[ni]))
    d_in = pairs[0][0].shape[0] * 2 if pairs else 1536
    model = GatingMLP(d_in=d_in)
    if not pairs:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        return
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_name = loss_type
    for _ in range(epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            pos_scores = []
            neg_scores = []
            for qv, dvp, dvn, sp, sn, ep, en in batch:
                qv_t = torch.tensor(qv, dtype=torch.float32)
                dvp_t = torch.tensor(dvp, dtype=torch.float32)
                dvn_t = torch.tensor(dvn, dtype=torch.float32)
                a_p, b_p = model.weights(qv_t, dvp_t)
                a_n, b_n = model.weights(qv_t, dvn_t)
                pos_scores.append(a_p * sp + b_p * ep)
                neg_scores.append(a_n * sn + b_n * en)
            pos_scores = torch.tensor(pos_scores, dtype=torch.float32)
            neg_scores = torch.tensor(neg_scores, dtype=torch.float32)
            diff = pos_scores - neg_scores
            if loss_name == "hinge":
                loss = F.relu(margin - diff).mean()
            elif loss_name == "logistic":
                loss = (-F.logsigmoid(diff)).mean()
            else:
                loss = F.softplus(margin - diff).mean()
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
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--shapley-T", type=int, default=10)
    p.add_argument("--shapley-lambda", type=float, default=0.5)
    p.add_argument("--min-subgraphs", type=int, default=1)
    p.add_argument("--shapley-seed", type=int, default=None)
    p.add_argument("--loss", type=str, default="softplus")
    p.add_argument("--hard-neg", action="store_true")
    args = p.parse_args()
    train_rank(dataset_jsonl=args.data, out_path=args.out, encoder_model=args.encoder, device=args.device, epochs=args.epochs, batch_size=args.batch, margin=args.margin, shapley_T=args.shapley_T, shapley_lambda=args.shapley_lambda, min_subgraphs=args.min_subgraphs, shapley_seed=args.shapley_seed, loss_type=args.loss, hard_neg=args.hard_neg)


if __name__ == "__main__":
    main()