# Natural Questions (NQ) 导入与评测

## 环境准备
- 安装依赖：
  - `pip install datasets transformers sentence-transformers rank-bm25 hnswlib`
  - 可选：`pip install spacy` 并下载模型 `python -m spacy download en_core_web_sm`

## 下载与转换
- 使用 `datasets` 下载：
  - Python：
    - `from datasets import load_dataset`
    - `ds = load_dataset("natural_questions", "default", split="validation")`
- 转换为 JSONL（本项目评测所需格式）：
  - 每行：`{"question":..., "answers":[...], "contexts":[{"title":..., "text":...}, ...]}`
  - 示例转换脚本片段（参考）：
    - `for ex in ds: q = ex["question"]; answers = ex.get("answers", {}).get("text", []); ctx = [{"title":"Wikipedia", "text": ex.get("document_text", "")[:1000]}]; write_jsonl({"question": q, "answers": answers, "contexts": ctx})`
  - 输出到如 `d:\data\nq_val.jsonl`

## 训练（可选）
- 结构相似度孪生 GNN：
  - 使用本地 KG（示例见 `examples/kg_example.jsonl`）生成图对：
    - `python Part-Divided/version2/dmar_rag/train_gnn_neg.py --kg-jsonl Part-Divided/version2/dmar_rag/examples/kg_example.jsonl --out d:\models\siamese_gnn.pt --epochs 5 --batch 32 --pairs 20000`
- 门控监督训练：
  - `python Part-Divided/version2/dmar_rag/cli.py train-gating --data Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\gating_mlp.pt`
- 门控排序训练（pairwise ranking）：
  - `python Part-Divided/version2/dmar_rag/cli.py train-gating-rank --data Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\gating_mlp_rank.pt`
- 交叉编码器微调：
  - `python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\cross_encoder_tuned --ce-base-model cross-encoder/ms-marco-MiniLM-L-6-v2 --ce-epochs 2 --ce-batch 16 --ce-neg-ratio 1.0 --ce-max-samples 5000`
- 门控联合优化（蒸馏+排序对齐）：
  - `python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl Part-Divided/version2/dmar_rag/examples/nq_sample.jsonl --out d:\models\gating_mlp_joint.pt --kg-depth 2`

## 评测
- 样例评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-nq-sample --k 5 --device cpu`
- 完整 JSONL 批量评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task nq --kg-jsonl d:\data\nq_val.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\nq_val.json --kg-depth 2`
  - 可选参数：`--gnn-weights d:\models\siamese_gnn.pt --gating-weights d:\models\gating.pt --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 --cross-weight 0.5`
- 指标：EM/F1（见 `generator.compute_em_f1`），结果写入 `--report-out`
