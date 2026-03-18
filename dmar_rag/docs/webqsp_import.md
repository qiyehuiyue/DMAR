# WebQuestionsSP (WebQSP) 导入与评测

## 环境准备
- `pip install transformers datasets sentence-transformers rank-bm25 hnswlib`

## 下载与转换
- 官方源通常提供 JSON（`WebQSP.train.json`, `WebQSP.test.json`）：
  - 字段包含 `Questions` 数组，元素含 `QuestionText` 与 `Answers`
- 转换为 JSONL：
  - 每行：`{"question":..., "answers":[...], "contexts":[{"title":..., "text":...}, ...]}`
  - 上下文来源：
    - 可使用 `KGStore` 提供的别名链接到实体，并构造 `contexts` 的文本描述（例如从维基或本地说明文本集）
    - 或将 KG 子图的节点名作为 `title`，边关联的简要说明作为 `text`
  - 输出到如 `d:\data\webqsp_test.jsonl`

## 训练（可选）
- 使用本地 KG + `train_gnn_neg.py` 训练孪生 GNN 权重，以增强结构信号在 KGQA 的效果。
- 交叉编码器微调：`python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data <dataset.jsonl> --out d:\models\cross_encoder_tuned`
- 门控联合优化：`python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl <dataset.jsonl> --out d:\models\gating_mlp_joint.pt --kg-depth 2`

## 评测
- 样例评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-webqsp-sample --k 5 --device cpu`
- 完整 JSONL 批量评测：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task webqsp --kg-jsonl d:\data\webqsp_test.jsonl --k 5 --device cpu --workers 4 --report-out d:\reports\webqsp_test.json --kg-depth 2`
- 指标：EM/F1 输出到报告文件