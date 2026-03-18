import argparse
import json
import requests
from pathlib import Path


def convert_dump(dump_path: str, out_path: str, max_entities: int = 10000):
    from .kg_store import KGStore
    store = KGStore()
    store.load_wikidata_dump(dump_path, max_entities=max_entities)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for h, t in store.edges:
            obj = {"head": h, "relation": "", "tail": t}
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")


def fetch_sparql(endpoint: str, query_file: str, out_path: str, limit: int = 1000):
    with open(query_file, "r", encoding="utf-8") as f:
        query = f.read()
    from .kg_store import KGStore
    store = KGStore()
    store.load_wikidata_sparql(endpoint, query, limit=limit)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for h, t in store.edges:
            obj = {"head": h, "relation": "", "tail": t}
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    c1 = sub.add_parser("convert-dump")
    c1.add_argument("--dump", type=str, required=True)
    c1.add_argument("--out", type=str, required=True)
    c1.add_argument("--max", type=int, default=10000)
    c2 = sub.add_parser("fetch-sparql")
    c2.add_argument("--endpoint", type=str, required=True)
    c2.add_argument("--query-file", type=str, required=True)
    c2.add_argument("--out", type=str, required=True)
    c2.add_argument("--limit", type=int, default=1000)
    args = p.parse_args()
    if args.cmd == "convert-dump":
        convert_dump(args.dump, args.out, max_entities=args.max)
    elif args.cmd == "fetch-sparql":
        fetch_sparql(args.endpoint, args.query_file, args.out, limit=args.limit)


if __name__ == "__main__":
    main()