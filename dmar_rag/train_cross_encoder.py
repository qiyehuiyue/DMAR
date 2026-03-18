import argparse
import json
import random
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _docs_from_ex(ex):
    ctxs = ex.get("contexts") or ex.get("evidence") or []
    docs = []
    for c in ctxs:
        docs.append({"title": c.get("title", ""), "text": c.get("text", "")})
    return docs


def _answers_from_ex(ex):
    a = ex.get("answers") or []
    if isinstance(a, list):
        return [str(x).strip().lower() for x in a if isinstance(x, str)]
    return []


def _is_pos(title: str, text: str, ex: dict) -> bool:
    tl = (title or "").lower()
    dl = (text or "").lower()
    answers = _answers_from_ex(ex)
    for ans in answers:
        if ans and (ans in dl or ans == tl):
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
        if tl in ev_titles:
            return True
        claim = str(ex.get("claim", "")).lower()
        tokens = [t for t in claim.split() if t]
        for t in tokens:
            if t and t in dl:
                return True
    return False


def build_examples(path: str, neg_ratio: float = 1.0, max_samples: int = None):
    examples = []
    for ex in _read_jsonl(path):
        q = ex.get("question") or ex.get("claim") or ""
        if not q:
            continue
        docs = _docs_from_ex(ex)
        if not docs:
            continue
        pos = []
        neg = []
        for d in docs:
            label = 1.0 if _is_pos(d.get("title", ""), d.get("text", ""), ex) else 0.0
            eg = InputExample(texts=[q, d.get("text", "")], label=label)
            if label == 1.0:
                pos.append(eg)
            else:
                neg.append(eg)
        if not pos:
            continue
        random.shuffle(neg)
        keep_neg = int(len(pos) * neg_ratio)
        examples.extend(pos)
        examples.extend(neg[:keep_neg])
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def train(data_path: str, out_path: str, base_model: str, epochs: int = 2, batch_size: int = 16, neg_ratio: float = 1.0, max_samples: int = None):
    train_data = build_examples(data_path, neg_ratio=neg_ratio, max_samples=max_samples)
    model = CrossEncoder(base_model, num_labels=1)
    model.fit(train_data=train_data, epochs=epochs, batch_size=batch_size)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--base-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    train(data_path=args.data, out_path=args.out, base_model=args.base_model, epochs=args.epochs, batch_size=args.batch, neg_ratio=args.neg_ratio, max_samples=args.max_samples)


if __name__ == "__main__":
    main()