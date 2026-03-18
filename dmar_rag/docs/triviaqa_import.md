# TriviaQA 导入与评测

## 环境准备
- `pip install datasets transformers sentence-transformers rank-bm25 hnswlib`

## 下载与转换
- 使用 `datasets` 下载 rc 版本：
  - `from datasets import load_dataset`
  - `ds = load_dataset("trivia_qa", "rc", split="validation")`
- 转换为 JSONL：
  - 每行：`{"question":..., "answers":[...], "contexts":[{"title":..., "text":...}, ...]}`
  - 示例转换：
    - `answers = [ex["answer"]["value"]] + ex.get("answer", {}).get("aliases", [])`
    - `contexts = [{"title":"passage", "text": ex.get("entity_pages", [{}])[0].get("wiki_context", "")[:1000]}]`
    - 写出 JSONL 到 `d:\data\triviaqa_val.jsonl`

## 训练（可选）
- 孪生 GNN：同 NQ 部分，使用 `train_gnn_neg.py` 生成权重。
- 交叉编码器微调：`python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data <dataset.jsonl> --out d:\models\cross_encoder_tuned`
- 门控联合优化：`python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl <dataset.jsonl> --out d:\models\gating_mlp_joint.pt --kg-depth 2`

## 评测
- 样例评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-triviaqa-sample --k 5 --device cpu`
- 完整 JSONL 批量评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task triviaqa --kg-jsonl d:\data\triviaqa_val.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\triviaqa_val.json --kg-depth 2`
- 指标：EM/F1 输出到报告文件