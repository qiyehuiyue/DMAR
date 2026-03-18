# FEVER 导入与评测

## 环境准备
- `pip install datasets transformers sentence-transformers rank-bm25 hnswlib`

## 下载与转换
- 使用 `datasets` 下载：
  - `from datasets import load_dataset`
  - `ds = load_dataset("fever", split="validation")`
- 转换为 JSONL：
  - 每行：`{"claim":..., "label":..., "evidence":[{"title":..., "text":...}, ...]}`
  - 证据来源：
    - 若数据集中提供证据片段与页面标题，直接映射到 `evidence` 数组
    - 若仅提供页面名与句子 ID，需先从维基抓取对应文本再填充
  - 输出到如 `d:\data\fever_val.jsonl`

## 训练（可选）
- 孪生 GNN：
  - 使用本地 KG，通过 `train_gnn_neg.py` 生成结构相似度权重：
    - `python Part-Divided/version2/dmar_rag/train_gnn_neg.py --kg-jsonl Part-Divided/version2/dmar_rag/examples/kg_example.jsonl --out d:\models\siamese_gnn.pt --epochs 5 --batch 32 --pairs 20000`

## 评测
- 样例评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-fever-sample --k 5 --device cpu`
- 完整 JSONL 批量评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task fever --kg-jsonl d:\data\fever_val.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\fever_val.json --kg-depth 2`
- 指标：Accuracy（使用 `T5LabelGenerator`），结果写入 `--report-out`

## 可选增强
- 可与 `--kg-jsonl` 加载的本地 KG、`--kg-depth` 的关系子图深度与 `--gnn-weights` 的结构相似度共同使用，提高检索与证据质量。
- 配合交叉编码器精排：`--cross-encoder-model` 与 `--cross-weight`。