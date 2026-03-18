from typing import List, Dict, Any

from datasets import load_dataset

from .retriever import DMARRetriever
from .kg_store import KGStore
from .generator import T5AnswerGenerator, T5LabelGenerator, compute_em_f1
from .query_reform import QueryReformer, OllamaReformer


def _build_docs_from_hotpot_context(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for i, c in enumerate(contexts):
        title = c.get("title", "") if isinstance(c, dict) else c[0]
        sentences = c.get("sentences", []) if isinstance(c, dict) else c[1]
        text = "\n".join(sentences) if isinstance(sentences, list) else str(sentences)
        docs.append({"id": i, "title": title, "text": text})
    return docs


def evaluate_hotpotqa_distractor(limit: int = 50, k: int = 5, device: str = "cpu", use_ollama: bool = False, ollama_url: str = None, ollama_embed_model: str = "nomic-embed-text", ollama_gen_model: str = "qwen2:7b", gating_weights: str = None, gnn_weights: str = None, kg_jsonl: str = None, disable_reform: bool = False, disable_kg: bool = False, single_anchor: bool = False, fixed_gate_alpha: float = None, ann_M: int = 32, ann_ef_construction: int = 200, ann_ef_query: int = 200, cross_encoder_model: str = None, cross_weight: float = 0.0, kg_depth: int = 1) -> Dict[str, float]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    cnt = 0
    hits = 0
    reformer = QueryReformer(device=-1 if device == "cpu" else 0) if not use_ollama else OllamaReformer(ollama_url, ollama_gen_model)
    store = None
    if kg_jsonl:
        store = KGStore()
        try:
            store.load_jsonl(kg_jsonl)
        except Exception:
            store = None
    for ex in ds:
        if cnt >= limit:
            break
        q = ex["question"]
        ctx = ex["context"]
        supp = ex["supporting_facts"]
        gt_titles = {t for t, _ in supp}
        docs = _build_docs_from_hotpot_context(ctx)
        if not docs:
            continue
        retriever = DMARRetriever(device=device, gating_weights=gating_weights, gnn_weights=gnn_weights, kg_store=store, disable_kg=disable_kg, fixed_gate_alpha=fixed_gate_alpha, ann_M=ann_M, ann_ef_construction=ann_ef_construction, ann_ef_query=ann_ef_query, cross_encoder_model=cross_encoder_model, cross_weight=cross_weight, kg_depth=kg_depth) if not use_ollama else DMARRetriever(device=device, encoder_backend="ollama", ollama_url=ollama_url, encoder_model=ollama_embed_model, gating_weights=gating_weights, gnn_weights=gnn_weights, kg_store=store, disable_kg=disable_kg, fixed_gate_alpha=fixed_gate_alpha, ann_M=ann_M, ann_ef_construction=ann_ef_construction, ann_ef_query=ann_ef_query, cross_encoder_model=cross_encoder_model, cross_weight=cross_weight, kg_depth=kg_depth)
        retriever.build_corpus(docs)
        final_docs = retriever.run(q, reformer=reformer, k=k, disable_reform=disable_reform, single_anchor=single_anchor)
        pred_titles = {d.title for d in final_docs}
        if gt_titles & pred_titles:
            hits += 1
        cnt += 1
    recall = hits / float(cnt) if cnt else 0.0
    return {"recall@{}".format(k): recall}


def _read_jsonl(path: str):
    import json
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def evaluate_nq_sample(path: str, k: int = 5, device: str = "cpu", gnn_weights: str = None, gating_weights: str = None, kg_depth: int = 1) -> Dict[str, float]:
    gen = T5AnswerGenerator(device=-1 if device == "cpu" else 0)
    em_sum = 0.0
    f1_sum = 0.0
    n = 0
    for ex in _read_jsonl(path):
        q = ex["question"]
        answers = ex.get("answers", [])
        ctxs = ex.get("contexts", [])
        docs = [{"id": i, "title": c.get("title", ""), "text": c.get("text", "")} for i, c in enumerate(ctxs)]
        retriever = DMARRetriever(device=device, gnn_weights=gnn_weights, gating_weights=gating_weights, kg_depth=kg_depth)
        texts = [d["text"] for d in docs]
        from .cache import EmbeddingDiskCache
        import numpy as np
        cache = EmbeddingDiskCache()
        emb = cache.get(texts, encoder_id="sentence")
        if emb is None:
            emb = retriever.encoder.encode(texts)
            emb = emb if isinstance(emb, np.ndarray) else np.array(emb)
            cache.set(texts, encoder_id="sentence", embeddings=emb)
        retriever.build_corpus_with_embeddings(docs, emb)
        final_docs = retriever.run(q, k=k)
        pred = gen.generate(q, [{"title": d.title, "text": d.text} for d in final_docs])
        m = compute_em_f1(pred, answers)
        em_sum += m["em"]
        f1_sum += m["f1"]
        n += 1
    return {"em": em_sum / max(n, 1), "f1": f1_sum / max(n, 1)}


def evaluate_triviaqa_sample(path: str, k: int = 5, device: str = "cpu", gnn_weights: str = None, gating_weights: str = None, kg_depth: int = 1) -> Dict[str, float]:
    return evaluate_nq_sample(path, k=k, device=device, gnn_weights=gnn_weights, gating_weights=gating_weights, kg_depth=kg_depth)


def evaluate_webqsp_sample(path: str, k: int = 5, device: str = "cpu", gnn_weights: str = None, gating_weights: str = None, kg_depth: int = 1) -> Dict[str, float]:
    return evaluate_nq_sample(path, k=k, device=device, gnn_weights=gnn_weights, gating_weights=gating_weights, kg_depth=kg_depth)


def evaluate_fever_sample(path: str, k: int = 5, device: str = "cpu", gnn_weights: str = None, gating_weights: str = None, kg_depth: int = 1) -> Dict[str, float]:
    lab = T5LabelGenerator(device=-1 if device == "cpu" else 0)
    acc = 0.0
    n = 0
    for ex in _read_jsonl(path):
        claim = ex["claim"]
        label = ex["label"]
        ev = ex.get("evidence", [])
        docs = [{"id": i, "title": e.get("title", ""), "text": e.get("text", "")} for i, e in enumerate(ev)]
        retriever = DMARRetriever(device=device, gnn_weights=gnn_weights, gating_weights=gating_weights, kg_depth=kg_depth)
        texts = [d["text"] for d in docs]
        from .cache import EmbeddingDiskCache
        import numpy as np
        cache = EmbeddingDiskCache()
        emb = cache.get(texts, encoder_id="sentence")
        if emb is None:
            emb = retriever.encoder.encode(texts)
            emb = emb if isinstance(emb, np.ndarray) else np.array(emb)
            cache.set(texts, encoder_id="sentence", embeddings=emb)
        retriever.build_corpus_with_embeddings(docs, emb)
        final_docs = retriever.run(claim, k=k)
        pred_label = lab.classify(claim, [{"title": d.title, "text": d.text} for d in final_docs])
        norm = pred_label.strip().lower()
        gold = label.strip().lower()
        if gold in norm:
            acc += 1.0
        n += 1
    return {"accuracy": acc / max(n, 1)}
def evaluate_batch_jsonl(path: str, task: str, k: int = 5, device: str = "cpu", gnn_weights: str = None, gating_weights: str = None, workers: int = 4, report_out: str = None, cross_encoder_model: str = None, cross_weight: float = 0.0, kg_depth: int = 1) -> Dict[str, float]:
    import concurrent.futures
    import numpy as np
    from .cache import EmbeddingDiskCache
    gen = T5AnswerGenerator(device=-1 if device == "cpu" else 0)
    lab = T5LabelGenerator(device=-1 if device == "cpu" else 0)
    examples = list(_read_jsonl(path))
    cache = EmbeddingDiskCache()
    def process_ex(ex):
        if task in ("nq","triviaqa","webqsp"):
            q = ex["question"]
            answers = ex.get("answers", [])
            ctxs = ex.get("contexts", [])
            docs = [{"id": i, "title": c.get("title", ""), "text": c.get("text", "")} for i, c in enumerate(ctxs)]
            r = DMARRetriever(device=device, gnn_weights=gnn_weights, gating_weights=gating_weights, cross_encoder_model=cross_encoder_model, cross_weight=cross_weight, kg_depth=kg_depth)
            texts = [d["text"] for d in docs]
            emb = cache.get(texts, encoder_id="sentence")
            if emb is None:
                emb = r.encoder.encode(texts)
                emb = emb if isinstance(emb, np.ndarray) else np.array(emb)
                cache.set(texts, encoder_id="sentence", embeddings=emb)
            r.build_corpus_with_embeddings(docs, emb)
            final_docs = r.run(q, k=k)
            pred = gen.generate(q, [{"title": d.title, "text": d.text} for d in final_docs])
            m = compute_em_f1(pred, answers)
            return (m["em"], m["f1"]) , None
        elif task == "fever":
            claim = ex["claim"]
            label = ex["label"]
            ev = ex.get("evidence", [])
            docs = [{"id": i, "title": e.get("title", ""), "text": e.get("text", "")} for i, e in enumerate(ev)]
            r = DMARRetriever(device=device, gnn_weights=gnn_weights, gating_weights=gating_weights, cross_encoder_model=cross_encoder_model, cross_weight=cross_weight, kg_depth=kg_depth)
            texts = [d["text"] for d in docs]
            emb = cache.get(texts, encoder_id="sentence")
            if emb is None:
                emb = r.encoder.encode(texts)
                emb = emb if isinstance(emb, np.ndarray) else np.array(emb)
                cache.set(texts, encoder_id="sentence", embeddings=emb)
            r.build_corpus_with_embeddings(docs, emb)
            final_docs = r.run(claim, k=k)
            pred_label = lab.classify(claim, [{"title": d.title, "text": d.text} for d in final_docs])
            norm = pred_label.strip().lower()
            gold = label.strip().lower()
            acc = 1.0 if gold in norm else 0.0
            return None, acc
        return (0.0,0.0), 0.0
    em_sum = 0.0
    f1_sum = 0.0
    acc_sum = 0.0
    n = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex_pool:
        for res in ex_pool.map(process_ex, examples):
            qa_metrics, acc = res
            if qa_metrics is not None:
                em_sum += qa_metrics[0]
                f1_sum += qa_metrics[1]
            if acc is not None:
                acc_sum += acc
            n += 1
    report = {}
    if task == "fever":
        report["accuracy"] = acc_sum / max(n, 1)
    else:
        report["em"] = em_sum / max(n, 1)
        report["f1"] = f1_sum / max(n, 1)
    if report_out:
        import json
        with open(report_out, "w", encoding="utf-8") as w:
            json.dump(report, w, ensure_ascii=False)
    return report
