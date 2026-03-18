# Natural Questions (NQ) Import & Evaluation

## Environment
- `pip install datasets transformers sentence-transformers rank-bm25 hnswlib`
- Optional: `python -m spacy download en_core_web_sm`

## Download & Convert
- Use `datasets`:
  - `from datasets import load_dataset`
  - `ds = load_dataset("natural_questions", "default", split="validation")`
- Convert to JSONL:
  - Each line: `{"question":..., "answers":[...], "contexts":[{"title":..., "text":...}]}`
  - Example: write `question`, answers from `answers.text`, and `document_text` truncated

## Training (Optional)
- Siamese GNN: `train_gnn_neg.py` generates structural similarity weights
- Gating supervised: `python Part-Divided/version2/dmar_rag/cli.py train-gating --data Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\gating_mlp.pt`
- Gating pairwise ranking: `python Part-Divided/version2/dmar_rag/cli.py train-gating-rank --data Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\gating_mlp_rank.pt`
- Cross-encoder fine-tuning: `python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\cross_encoder_tuned`
- Joint gating + CE: `python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\gating_mlp_joint.pt --kg-depth 2`

## Evaluation
- Sample: `python Part-Divided/version2/dmar_rag/cli.py evaluate-nq-sample --k 5 --device cpu`
- Batch:
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task nq --kg-jsonl d:\data\nq_val.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\nq_val.json --kg-depth 2`
  - Optional: `--gnn-weights --gating-weights --cross-encoder-model --cross-weight`
