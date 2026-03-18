import re
from typing import List, Dict

from transformers import pipeline


def _tok(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]


def compute_em_f1(pred: str, golds: List[str]) -> Dict[str, float]:
    if not golds:
        return {"em": 0.0, "f1": 0.0}
    pred_toks = _tok(pred)
    em = 0.0
    best_f1 = 0.0
    for g in golds:
        gt = _tok(g)
        if " ".join(pred_toks) == " ".join(gt):
            em = 1.0
        common = len(set(pred_toks) & set(gt))
        if common == 0:
            f1 = 0.0
        else:
            precision = common / max(len(pred_toks), 1)
            recall = common / max(len(gt), 1)
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return {"em": em, "f1": best_f1}


class T5AnswerGenerator:
    def __init__(self, model_name: str = "t5-base", device: int = -1):
        self.pipe = pipeline("text2text-generation", model=model_name, device=device)

    def generate(self, question: str, docs: List[Dict[str, str]], max_new_tokens: int = 32) -> str:
        ctx = "\n".join([d.get("text", "") for d in docs])
        prompt = "Answer the question based on the context.\nQuestion: " + question + "\nContext: " + ctx
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, num_beams=4)[0]["generated_text"].strip()
        return out


class T5LabelGenerator:
    def __init__(self, model_name: str = "t5-base", device: int = -1):
        self.pipe = pipeline("text2text-generation", model=model_name, device=device)

    def classify(self, claim: str, docs: List[Dict[str, str]], max_new_tokens: int = 8) -> str:
        ctx = "\n".join([d.get("text", "") for d in docs])
        prompt = (
            "Classify the claim using the context as one of: Supported, Refuted, NotEnoughInfo.\n"
            + "Claim: " + claim + "\nContext: " + ctx
        )
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, num_beams=4)[0]["generated_text"].strip()
        return out