import argparse
import json
import os
import sys
from pathlib import Path

try:
    from . import evaluate as evaluate_mod
    from . import convert_datasets as convert_mod
    from . import train_gating as train_gating_mod
    from . import train_gnn_neg as train_gnn_neg_mod
    from . import train_gating_rank as train_gating_rank_mod
    from . import train_cross_encoder as train_cross_encoder_mod
    from . import train_joint_gating_ce as train_joint_gating_ce_mod
    from . import make_example_weights as make_example_weights_mod
except ImportError:
    # Allow running as a script without package context
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import evaluate as evaluate_mod
    import convert_datasets as convert_mod
    import train_gating as train_gating_mod
    import train_gnn_neg as train_gnn_neg_mod
    import train_gating_rank as train_gating_rank_mod
    import train_cross_encoder as train_cross_encoder_mod
    import train_joint_gating_ce as train_joint_gating_ce_mod
    import make_example_weights as make_example_weights_mod


BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"


def _example_path(name: str) -> str:
    return str(EXAMPLES_DIR / name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["evaluate-hotpotqa","evaluate-nq-sample","evaluate-triviaqa-sample","evaluate-webqsp-sample","evaluate-fever-sample","evaluate-batch-jsonl","convert-nq","convert-triviaqa","convert-fever","convert-webqsp","train-gating","train-gating-rank","train-gnn-neg","train-cross-encoder","train-joint-gating-ce","make-example-weights"], help="Task to run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-ollama", action="store_true")
    p.add_argument("--ollama-url", type=str, default="http://58.199.164.87:11434")
    p.add_argument("--ollama-embed-model", type=str, default="nomic-embed-text:latest")
    p.add_argument("--ollama-gen-model", type=str, default="qwen3:latest")
    p.add_argument("--gating-weights", type=str, default=None)
    p.add_argument("--gnn-weights", type=str, default=None)
    p.add_argument("--kg-jsonl", type=str, default=None)
    p.add_argument("--disable-reform", action="store_true")
    p.add_argument("--disable-kg", action="store_true")
    p.add_argument("--single-anchor", action="store_true")
    p.add_argument("--fixed-gate-alpha", type=float, default=None)
    p.add_argument("--ann-M", type=int, default=32)
    p.add_argument("--ann-ef-construction", type=int, default=200)
    p.add_argument("--ann-ef-query", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report-out", type=str, default=None)
    p.add_argument("--batch-task", choices=["nq","triviaqa","webqsp","fever"], default="nq")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--file", type=str, default=None)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--cross-weight", type=float, default=0.5)
    p.add_argument("--ce-base-model", type=str, default=None)
    p.add_argument("--ce-epochs", type=int, default=None)
    p.add_argument("--ce-batch", type=int, default=None)
    p.add_argument("--ce-neg-ratio", type=float, default=1.0)
    p.add_argument("--ce-max-samples", type=int, default=None)
    p.add_argument("--kg-depth", type=int, default=1)
    args = p.parse_args()
    if args.task == "evaluate-hotpotqa":
        eval_limit = args.limit if args.limit is not None else 50
        res = evaluate_mod.evaluate_hotpotqa_distractor(limit=eval_limit, k=args.k, device=args.device, use_ollama=args.use_ollama, ollama_url=args.ollama_url, ollama_embed_model=args.ollama_embed_model, ollama_gen_model=args.ollama_gen_model, gating_weights=args.gating_weights, gnn_weights=args.gnn_weights, kg_jsonl=args.kg_jsonl, disable_reform=args.disable_reform, disable_kg=args.disable_kg, single_anchor=args.single_anchor, fixed_gate_alpha=args.fixed_gate_alpha, ann_M=args.ann_M, ann_ef_construction=args.ann_ef_construction, ann_ef_query=args.ann_ef_query, cross_encoder_model=args.cross_encoder_model, cross_weight=args.cross_weight, kg_depth=args.kg_depth)
        print(json.dumps(res, ensure_ascii=False))
    elif args.task == "evaluate-nq-sample":
        sample = args.kg_jsonl or _example_path("nq_sample.jsonl")
        res = evaluate_mod.evaluate_nq_sample(path=sample, k=args.k, device=args.device, gnn_weights=args.gnn_weights, gating_weights=args.gating_weights, kg_depth=args.kg_depth)
        print(json.dumps(res, ensure_ascii=False))
    elif args.task == "evaluate-triviaqa-sample":
        sample = args.kg_jsonl or _example_path("triviaqa_sample.jsonl")
        res = evaluate_mod.evaluate_triviaqa_sample(path=sample, k=args.k, device=args.device, gnn_weights=args.gnn_weights, gating_weights=args.gating_weights, kg_depth=args.kg_depth)
        print(json.dumps(res, ensure_ascii=False))
    elif args.task == "evaluate-webqsp-sample":
        sample = args.kg_jsonl or _example_path("webqsp_sample.jsonl")
        res = evaluate_mod.evaluate_webqsp_sample(path=sample, k=args.k, device=args.device, gnn_weights=args.gnn_weights, gating_weights=args.gating_weights, kg_depth=args.kg_depth)
        print(json.dumps(res, ensure_ascii=False))
    elif args.task == "evaluate-fever-sample":
        sample = args.kg_jsonl or _example_path("fever_sample.jsonl")
        res = evaluate_mod.evaluate_fever_sample(path=sample, k=args.k, device=args.device, gnn_weights=args.gnn_weights, gating_weights=args.gating_weights, kg_depth=args.kg_depth)
        print(json.dumps(res, ensure_ascii=False))
    elif args.task == "evaluate-batch-jsonl":
        res = evaluate_mod.evaluate_batch_jsonl(path=args.kg_jsonl, task=args.batch_task, k=args.k, device=args.device, gnn_weights=args.gnn_weights, gating_weights=args.gating_weights, workers=args.workers, report_out=args.report_out, cross_encoder_model=args.cross_encoder_model, cross_weight=args.cross_weight, kg_depth=args.kg_depth)
        print(json.dumps(res, ensure_ascii=False))
    elif args.task == "convert-nq":
        out = args.out or args.kg_jsonl or "nq.jsonl"
        convert_mod.convert_nq(out_path=out, split=args.split, limit=args.limit, max_chars=args.max_chars)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "convert-triviaqa":
        out = args.out or args.kg_jsonl or "triviaqa_rc.jsonl"
        convert_mod.convert_triviaqa_rc(out_path=out, split=args.split, limit=args.limit, max_chars=args.max_chars)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "convert-fever":
        out = args.out or args.kg_jsonl or "fever.jsonl"
        convert_mod.convert_fever(out_path=out, split=args.split, limit=args.limit, max_chars=args.max_chars)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "convert-webqsp":
        in_file = args.file or args.kg_jsonl or "WebQSP.test.json"
        out = args.out or "webqsp.jsonl"
        convert_mod.convert_webqsp(in_file=in_file, out_path=out, limit=args.limit, max_chars=args.max_chars)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "train-gating":
        out = args.out or "gating_mlp.pt"
        data = args.kg_jsonl or _example_path("nq_sample.jsonl")
        train_gating_mod.train(dataset_jsonl=data, out_path=out, encoder_model=args.ollama_embed_model if args.use_ollama else "princeton-nlp/sup-simcse-bert-base-uncased", device=args.device, epochs=args.limit or 3, batch_size=args.k or 64)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "train-gnn-neg":
        out = args.out or "siamese_gnn.pt"
        kgp = args.kg_jsonl or _example_path("kg_example.jsonl")
        train_gnn_neg_mod.train_with_kg(kg_jsonl=kgp, weights_out=out, epochs=args.limit or 3, lr=1e-3, batch_size=args.k or 32, total_pairs=args.workers * 1000)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "train-gating-rank":
        out = args.out or "gating_mlp_rank.pt"
        data = args.kg_jsonl or _example_path("nq_sample.jsonl")
        train_gating_rank_mod.train_rank(dataset_jsonl=data, out_path=out, encoder_model=args.ollama_embed_model if args.use_ollama else "princeton-nlp/sup-simcse-bert-base-uncased", device=args.device, epochs=args.limit or 3, batch_size=args.k or 64, margin=0.1, shapley_T=10, shapley_lambda=0.5, min_subgraphs=1, shapley_seed=None, loss_type="softplus", hard_neg=True)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "train-cross-encoder":
        out = args.out or "cross_encoder_tuned"
        data = args.kg_jsonl or _example_path("nq_sample.jsonl")
        base_model = args.ce_base_model or args.cross_encoder_model
        epochs = args.ce_epochs if args.ce_epochs is not None else (args.limit or 2)
        batch = args.ce_batch if args.ce_batch is not None else (args.k or 16)
        train_cross_encoder_mod.train(data_path=data, out_path=out, base_model=base_model, epochs=epochs, batch_size=batch, neg_ratio=args.ce_neg_ratio, max_samples=args.ce_max_samples)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "train-joint-gating-ce":
        out = args.out or "gating_mlp_joint.pt"
        data = args.kg_jsonl or _example_path("nq_sample.jsonl")
        train_joint_gating_ce_mod.train_joint(data_path=data, out_path=out, encoder_model=args.ollama_embed_model if args.use_ollama else "princeton-nlp/sup-simcse-bert-base-uncased", cross_model=args.cross_encoder_model, device=args.device, epochs=args.ce_epochs or 2, batch_size=args.ce_batch or 32, shapley_T=10, shapley_lambda=0.5, min_subgraphs=1, depth=args.kg_depth, loss_alpha=0.7)
        print(json.dumps({"out": out}, ensure_ascii=False))
    elif args.task == "make-example-weights":
        if args.out:
            make_example_weights_mod.save_gating(args.out, d_in=args.k * 2 if args.k else 1536)
        if args.report_out:
            make_example_weights_mod.save_gnn(args.report_out)
        print(json.dumps({"gating_out": args.out, "gnn_out": args.report_out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
