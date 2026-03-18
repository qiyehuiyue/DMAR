# FEVER Import & Evaluation

## Environment
- `pip install datasets transformers sentence-transformers rank-bm25 hnswlib`

## Download & Convert
- Use `datasets`:
  - `from datasets import load_dataset`
  - `ds = load_dataset("fever", split="validation")`
- Convert to JSONL:
  - Each line: `{"claim":..., "label":..., "evidence":[{"title":..., "text":...}]}`
  - If dataset provides only page names or sentence IDs, fetch corresponding wiki text beforehand

## Training (Optional)
- Siamese GNN: `train_gnn_neg.py`
- Cross-encoder fine-tuning: `python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data <dataset.jsonl> --out d:\models\cross_encoder_tuned`
- Joint gating + CE: `python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl <dataset.jsonl> --out d:\models\gating_mlp_joint.pt --kg-depth 2`

## Evaluation
- Sample: `python Part-Divided/version2/dmar_rag/cli.py evaluate-fever-sample --k 5 --device cpu`
- Batch: `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task fever --kg-jsonl d:\data\fever_val.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\fever_val.json --kg-depth 2`

## Notes
- Combine `--kg-jsonl`, `--kg-depth` and `--gnn-weights` to strengthen structural signal; optionally enable cross-encoder reranking via `--cross-encoder-model --cross-weight`