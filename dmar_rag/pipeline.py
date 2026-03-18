"""
End-to-end training / evaluation pipeline runner.

This automates the typical flow:
1) (Optional) dataset conversion to JSONL
2) Train gating MLP
3) Train GNN on KG pairs
4) Fine-tune cross-encoder
5) Train joint gating + cross-encoder
6) Batch evaluation

Use a JSON config to control paths/params or rely on defaults that point to the bundled examples.
Example:
python -m dmar_rag.pipeline --config my_pipeline.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from . import convert_datasets as convert_mod
    from . import train_gating as train_gating_mod
    from . import train_gnn_neg as train_gnn_neg_mod
    from . import train_cross_encoder as train_cross_encoder_mod
    from . import train_joint_gating_ce as train_joint_gating_ce_mod
    from . import evaluate as evaluate_mod
except ImportError:
    # Allow running as a script without package context
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import convert_datasets as convert_mod
    import train_gating as train_gating_mod
    import train_gnn_neg as train_gnn_neg_mod
    import train_cross_encoder as train_cross_encoder_mod
    import train_joint_gating_ce as train_joint_gating_ce_mod
    import evaluate as evaluate_mod


BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"


def _default_config() -> Dict[str, Any]:
    return {
        "data": {
            "nq_jsonl": str(EXAMPLES_DIR / "nq_sample.jsonl"),
            "kg_jsonl": str(EXAMPLES_DIR / "kg_example.jsonl"),
            "triviaqa_jsonl": str(EXAMPLES_DIR / "triviaqa_sample.jsonl"),
            "fever_jsonl": str(EXAMPLES_DIR / "fever_sample.jsonl"),
            "webqsp_jsonl": str(EXAMPLES_DIR / "webqsp_sample.jsonl"),
            "webqsp_source": None,
        },
        "runtime": {
            "device": "cpu",
            "use_ollama": False,
            "ollama_url": "http://localhost:11434",
            "ollama_embed_model": "nomic-embed-text:latest",
        },
        "train": {
            "gating": {"enabled": True, "out": "gating_mlp.pt", "epochs": 3, "batch": 64},
            "gnn_neg": {"enabled": True, "out": "siamese_gnn.pt", "epochs": 3, "lr": 1e-3, "batch": 32, "pairs": 4000},
            "cross_encoder": {
                "enabled": True,
                "out": "cross_encoder_tuned",
                "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "epochs": 2,
                "batch": 16,
                "neg_ratio": 1.0,
                "max_samples": None,
            },
            "joint": {"enabled": True, "out": "gating_mlp_joint.pt", "epochs": 2, "batch": 32, "kg_depth": 1},
        },
        "eval": {
            "enabled": True,
            "task": "nq",
            "k": 5,
            "report_out": "report.json",
            "cross_weight": 0.5,
            "kg_depth": 1,
        },
    }


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return _default_config()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _log(msg: str):
    print(f"[pipeline] {msg}")


def _run_convert(cfg: Dict[str, Any]):
    # Only convert if target files do not exist to avoid re-downloading unnecessarily.
    data_cfg = cfg.get("data", {})
    if data_cfg.get("nq_jsonl") and not Path(data_cfg["nq_jsonl"]).exists():
        _log("Converting NQ to JSONL...")
        convert_mod.convert_nq(out_path=data_cfg["nq_jsonl"], split="validation", limit=None, max_chars=2000)
    if data_cfg.get("triviaqa_jsonl") and not Path(data_cfg["triviaqa_jsonl"]).exists():
        _log("Converting TriviaQA to JSONL...")
        convert_mod.convert_triviaqa_rc(out_path=data_cfg["triviaqa_jsonl"], split="validation", limit=None, max_chars=2000)
    if data_cfg.get("fever_jsonl") and not Path(data_cfg["fever_jsonl"]).exists():
        _log("Converting FEVER to JSONL...")
        convert_mod.convert_fever(out_path=data_cfg["fever_jsonl"], split="validation", limit=None, max_chars=2000)
    # WebQSP requires a local JSON; only convert when a source is provided.
    if data_cfg.get("webqsp_source") and data_cfg.get("webqsp_jsonl") and not Path(data_cfg["webqsp_jsonl"]).exists():
        _log("Converting WebQSP to JSONL...")
        convert_mod.convert_webqsp(in_file=data_cfg["webqsp_source"], out_path=data_cfg["webqsp_jsonl"], limit=None, max_chars=2000)


def _run_train_gating(cfg: Dict[str, Any]):
    tcfg = cfg["train"]["gating"]
    if not tcfg.get("enabled", False):
        return
    _log(f"Training gating MLP -> {tcfg['out']}")
    train_gating_mod.train(
        dataset_jsonl=cfg["data"]["nq_jsonl"],
        out_path=tcfg["out"],
        encoder_model=cfg["runtime"]["ollama_embed_model"] if cfg["runtime"].get("use_ollama") else "princeton-nlp/sup-simcse-bert-base-uncased",
        device=cfg["runtime"]["device"],
        epochs=tcfg.get("epochs", 3),
        batch_size=tcfg.get("batch", 64),
    )


def _run_train_gnn(cfg: Dict[str, Any]):
    tcfg = cfg["train"]["gnn_neg"]
    if not tcfg.get("enabled", False):
        return
    _log(f"Training Siamese GNN -> {tcfg['out']}")
    train_gnn_neg_mod.train_with_kg(
        kg_jsonl=cfg["data"]["kg_jsonl"],
        weights_out=tcfg["out"],
        epochs=tcfg.get("epochs", 3),
        lr=tcfg.get("lr", 1e-3),
        batch_size=tcfg.get("batch", 32),
        total_pairs=tcfg.get("pairs", 4000),
    )


def _run_train_cross(cfg: Dict[str, Any]):
    tcfg = cfg["train"]["cross_encoder"]
    if not tcfg.get("enabled", False):
        return
    _log(f"Fine-tuning cross-encoder -> {tcfg['out']}")
    train_cross_encoder_mod.train(
        data_path=cfg["data"]["nq_jsonl"],
        out_path=tcfg["out"],
        base_model=tcfg.get("base_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        epochs=tcfg.get("epochs", 2),
        batch_size=tcfg.get("batch", 16),
        neg_ratio=tcfg.get("neg_ratio", 1.0),
        max_samples=tcfg.get("max_samples"),
    )


def _run_train_joint(cfg: Dict[str, Any]):
    tcfg = cfg["train"]["joint"]
    if not tcfg.get("enabled", False):
        return
    _log(f"Training joint gating + CE -> {tcfg['out']}")
    train_joint_gating_ce_mod.train_joint(
        data_path=cfg["data"]["nq_jsonl"],
        out_path=tcfg["out"],
        encoder_model=cfg["runtime"]["ollama_embed_model"] if cfg["runtime"].get("use_ollama") else "princeton-nlp/sup-simcse-bert-base-uncased",
        cross_model=cfg["train"]["cross_encoder"].get("out") or cfg["train"]["cross_encoder"].get("base_model"),
        device=cfg["runtime"]["device"],
        epochs=tcfg.get("epochs", 2),
        batch_size=tcfg.get("batch", 32),
        shapley_T=10,
        shapley_lambda=0.5,
        min_subgraphs=1,
        depth=tcfg.get("kg_depth", 1),
        loss_alpha=0.7,
    )


def _run_eval(cfg: Dict[str, Any]):
    ecfg = cfg["eval"]
    if not ecfg.get("enabled", False):
        return
    _log("Running batch evaluation...")
    res = evaluate_mod.evaluate_batch_jsonl(
        path=cfg["data"]["nq_jsonl"],
        task=ecfg.get("task", "nq"),
        k=ecfg.get("k", 5),
        device=cfg["runtime"]["device"],
        gnn_weights=cfg["train"]["gnn_neg"].get("out") if cfg["train"]["gnn_neg"].get("enabled") else None,
        gating_weights=cfg["train"]["gating"].get("out") if cfg["train"]["gating"].get("enabled") else None,
        workers=4,
        report_out=ecfg.get("report_out"),
        cross_encoder_model=cfg["train"]["cross_encoder"].get("out") if cfg["train"]["cross_encoder"].get("enabled") else None,
        cross_weight=ecfg.get("cross_weight", 0.5),
        kg_depth=ecfg.get("kg_depth", 1),
    )
    _log(f"Eval metrics: {json.dumps(res, ensure_ascii=False)}")


def main():
    ap = argparse.ArgumentParser(description="DMAR pipeline automation")
    ap.add_argument("--config", type=str, default=None, help="JSON config file; if omitted, built-in defaults are used")
    ap.add_argument(
        "--steps",
        type=str,
        default="convert,train_gating,train_gnn,train_cross,train_joint,eval",
        help="Comma-separated steps to run",
    )
    args = ap.parse_args()
    cfg = _load_config(args.config)
    steps: List[str] = [s.strip() for s in args.steps.split(",") if s.strip()]

    if "convert" in steps:
        _run_convert(cfg)
    if "train_gating" in steps:
        _run_train_gating(cfg)
    if "train_gnn" in steps:
        _run_train_gnn(cfg)
    if "train_cross" in steps:
        _run_train_cross(cfg)
    if "train_joint" in steps:
        _run_train_joint(cfg)
    if "eval" in steps:
        _run_eval(cfg)


if __name__ == "__main__":
    main()
