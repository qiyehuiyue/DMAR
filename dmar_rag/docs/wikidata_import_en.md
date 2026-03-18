# Wikidata Import Guide

## Overview
- This guide explains how to integrate Wikidata into the DMAR pipeline via local JSONL edges and aliases.
- Two routes are supported: converting dumps and fetching via SPARQL, then loading into `KGStore`.

## Route 1: Local Dump
- Download Wikidata JSON dump (e.g., `latest-all.json.bz2`).
- Parse and convert to JSONL lines with fields: `head/relation/tail` and optional `aliases`.
- Use `KGStore.load_jsonl(path)` to load; or use the converter:
  - `python Part-Divided/version2/dmar_rag/wikidata_tools.py convert-dump --dump d:\data\wikidata_dump.jsonl --out d:\data\wikidata_edges.jsonl --max 50000`

## Route 2: SPARQL Endpoint
- Run SPARQL on an endpoint (e.g., `https://query.wikidata.org/sparql`) to retrieve triples and aliases.
- Save results to local JSONL and load via `KGStore.load_jsonl(path)`; or use the fetch tool:
  - `python Part-Divided/version2/dmar_rag/wikidata_tools.py fetch-sparql --endpoint https://query.wikidata.org/sparql --query-file d:\data\query.sparql --out d:\data\wikidata_edges.jsonl --limit 20000`

## Integration with Code
- Pass `--kg-jsonl <path>` in evaluation to enable KG.
- Structural similarity can use Siamese GNN via `--gnn-weights <path>`.

## Entity Linking Robustness
- `KGStore` provides case-normalized alias dictionary and fuzzy matching (prefix/suffix/containment).
- `KGUtil.link_entities(text)` caches linking results and falls back to fuzzy matching.

## Subgraph Depth
- Shapley evaluation supports `depth` via `KGStore.build_subgraph(entities, depth)`.
- Set depth in evaluation with `--kg-depth <int>`:
  - `python Part-Divided/version2/dmar_rag/cli.py evaluate-batch-jsonl --batch-task nq --kg-jsonl d:\data\nq_val.jsonl --kg-depth 2 --k 5 --device cpu --workers 4 --report-out d:\reports\nq_val.json`

## Cross-Encoder Reranking & Joint Training
- Reranking: enable with `--cross-encoder-model` and `--cross-weight`.
- Cross-encoder fine-tuning:
  - `python Part-Divided/version2/dmar_rag/cli.py train-cross-encoder --data <dataset.jsonl> --out <ce_out_dir> --ce-base-model cross-encoder/ms-marco-MiniLM-L-6-v2 --ce-epochs 3 --ce-batch 32 --ce-neg-ratio 1.5 --ce-max-samples 5000`
- Joint gating optimization (distillation + ranking alignment):
  - `python Part-Divided/version2/dmar_rag/cli.py train-joint-gating-ce --kg-jsonl <dataset.jsonl> --out <gating_out.pt> --device cpu --ce-epochs 2 --ce-batch 32 --kg-depth 2`
