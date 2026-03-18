# Wikidata 导入说明

## 方式一：使用本地 Dump
- 下载 Wikidata JSON dump（例如 `latest-all.json.bz2`）。
- 解析 dump，抽取需要的实体与关系，生成 JSONL：每行包含 `head/relation/tail`，以及可选的 `aliases` 字典。
- 生成的 JSONL 可通过 `KGStore.load_jsonl(path)` 导入，供 `KGUtil` 构建子图。
 - 也可以使用转换工具：
   - `python Part-Divided/version2/dmar_rag/wikidata_tools.py convert-dump --dump d:\data\wikidata_dump.jsonl --out d:\data\wikidata_edges.jsonl --max 50000`

## 方式二：使用 SPARQL Endpoint
- 在可用的 SPARQL 服务（例如 `https://query.wikidata.org/sparql`）运行查询，批量获取三元组与别名。
- 将查询结果转存为本地 JSONL（含 `head/relation/tail/aliases`），通过 `KGStore.load_jsonl(path)` 导入。
 - 也可以使用抓取工具：
   - 准备 SPARQL 查询文件 `query.sparql`，并执行：
   - `python Part-Divided/version2/dmar_rag/wikidata_tools.py fetch-sparql --endpoint https://query.wikidata.org/sparql --query-file d:\data\query.sparql --out d:\data\wikidata_edges.jsonl --limit 20000`

## 与当前代码对接
- 本地 KG 文件：`--kg-jsonl <path>` 在评测时加载并生效。
- 孪生 GNN 权重：`--gnn-weights <path>` 在结构相似度计算中使用。
- 如需自定义加载流程，可扩展 `KGStore.load_wikidata_dump(path)` 或 `KGStore.load_wikidata_sparql(endpoint)`，将解析后的三元组与别名注入到 `KGStore`。

## 实体链接与鲁棒性
- 代码在 `KGStore` 中增加了别名大小写归一化与模糊匹配（前后缀与包含式匹配），调用顺序为精确匹配→大小写匹配→模糊匹配→节点名匹配。
- `KGUtil.link_entities(text)` 会缓存链接结果并自动回退模糊匹配，提升链接成功率。

## 关系子图深度
- Shapley 评估已支持 `depth` 参数，基于 `KGStore.build_subgraph(entities, depth)` 构造子图；评测时可通过 `--kg-depth` 指定：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task nq --kg-jsonl d:\data\nq_val.jsonl --kg-depth 2 --k 5 --device cpu --workers 4 --report-out d:\reports\nq_val.json`

## 交叉精排与联合训练
- 交叉精排：评测命令支持 `--cross-encoder-model` 与 `--cross-weight` 开启精排增强。
- 交叉精排微调：
  - `python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data <dataset.jsonl> --out <ce_out_dir> --ce-base-model cross-encoder/ms-marco-MiniLM-L-6-v2 --ce-epochs 3 --ce-batch 32 --ce-neg-ratio 1.5 --ce-max-samples 5000`
- 门控联合优化（蒸馏+排序对齐）：
  - `python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl <dataset.jsonl> --out <gating_out.pt> --device cpu --ce-epochs 2 --ce-batch 32 --kg-depth 2`

## 训练孪生 GNN 示例
- 训练数据样例：`Part-Divided/version2/dmar_rag/examples/train_pairs.jsonl`
- 训练命令：
  - `python Part-Divided/version2/dmar_rag/train_gnn.py --data Part-Divided/version2/dmar_rag/examples/train_pairs.jsonl --out d:\models\siamese_gnn.pt --epochs 5 --lr 1e-3 --batch 16`
- 评测时加载权重：
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-hotpotqa --limit 50 --k 5 --device cpu --kg-jsonl Part-Divided/version2/dmar_rag/examples/kg_example.jsonl --gnn-weights d:\models\siamese_gnn.pt`