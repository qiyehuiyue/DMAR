# TriviaQA Import & Evaluation

## Environment
- `pip install datasets transformers sentence-transformers rank-bm25 hnswlib`

## Download & Convert
- Use `datasets` rc split:
  - `from datasets import load_dataset`
  - `ds = load_dataset("trivia_qa", "rc", split="validation")`
- Convert to JSONL:
  - Each line: `{"question":..., "answers":[...], "contexts":[{"title":..., "text":...}]}`
  - Answers: `[answer.value] + answer.aliases`; context: `entity_pages[0].wiki_context`

## Training (Optional)
- Siamese GNN: `train_gnn_neg.py`
- Cross-encoder fine-tuning: `python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data <dataset.jsonl> --out d:\models\cross_encoder_tuned`
- Joint gating + CE: `python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl <dataset.jsonl> --out d:\models\gating_mlp_joint.pt --kg-depth 2`

## Evaluation
- Sample: `python Part-Divided/version2/dmar_rag/cli.py evaluate-triviaqa-sample --k 5 --device cpu`
- Batch: `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task triviaqa --kg-jsonl d:\data\triviaqa_val.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\triviaqa_val.json --kg-depth 2`
