# WebQuestionsSP (WebQSP) Import & Evaluation

## Environment
- `pip install transformers datasets sentence-transformers rank-bm25 hnswlib`

## Download & Convert
- Official JSON files (e.g., `WebQSP.train.json`, `WebQSP.test.json`) include `Questions` with `QuestionText` and `Answers`
- Convert to JSONL lines: `{"question":..., "answers":[...], "contexts":[{"title":..., "text":...}]}`
  - Contexts may be generated from KG entities or local wiki text

## Training (Optional)
- Siamese GNN with local KG: `train_gnn_neg.py`
- Cross-encoder fine-tuning: `python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data <dataset.jsonl> --out d:\models\cross_encoder_tuned`
- Joint gating + CE: `python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl <dataset.jsonl> --out d:\models\gating_mlp_joint.pt --kg-depth 2`

## Evaluation
- Sample: `python Part-Divided/version2/dmar_rag/cli.py evaluate-webqsp-sample --k 5 --device cpu`
- Batch: `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task webqsp --kg-jsonl d:\data\webqsp_test.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\webqsp_test.json --kg-depth 2`
