import argparse
import json
from typing import Optional

def _safe_str(x: object) -> str:
    try:
        s = str(x) if x is not None else ""
    except Exception:
        s = ""
    return s.replace("\r", " ").replace("\n", " ").strip()

def _truncate(s: str, n: int) -> str:
    return s[:n] if isinstance(s, str) else ""

def convert_nq(out_path: str, split: str = "validation", limit: Optional[int] = None, max_chars: int = 2000):
    from datasets import load_dataset
    ds = load_dataset("natural_questions", "default", split=split)
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in ds:
            try:
                q = _safe_str(ex.get("question") or ex.get("question_text"))
                ans = ex.get("answers")
                answers = []
                if isinstance(ans, dict):
                    answers = ans.get("text") or ans.get("value") or []
                elif isinstance(ans, list):
                    answers = [a for a in ans if isinstance(a, str)]
                ctx_text = _safe_str(ex.get("document_text") or ex.get("context"))
                rec = {"question": q, "answers": answers or [], "contexts": [{"title": "NQ", "text": _truncate(ctx_text, max_chars)}]}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
                if limit and cnt >= limit:
                    break
            except Exception:
                continue

def convert_triviaqa_rc(out_path: str, split: str = "validation", limit: Optional[int] = None, max_chars: int = 2000):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc", split=split)
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in ds:
            try:
                q = _safe_str(ex.get("question"))
                ans_obj = ex.get("answer") or {}
                main = _safe_str(ans_obj.get("value"))
                aliases = ans_obj.get("aliases") or []
                answers = ([main] if main else []) + ([a for a in aliases if isinstance(a, str)])
                pages = ex.get("entity_pages") or []
                ctx_text = ""
                if isinstance(pages, list) and len(pages) > 0:
                    ctx_text = _safe_str(pages[0].get("wiki_context"))
                rec = {"question": q, "answers": answers, "contexts": [{"title": "TriviaQA", "text": _truncate(ctx_text, max_chars)}]}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
                if limit and cnt >= limit:
                    break
            except Exception:
                continue

def convert_fever(out_path: str, split: str = "validation", limit: Optional[int] = None, max_chars: int = 2000):
    from datasets import load_dataset
    ds = load_dataset("fever", split=split)
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in ds:
            try:
                claim = _safe_str(ex.get("claim"))
                label = _safe_str(ex.get("label") or "NotEnoughInfo")
                ev = ex.get("evidence") or []
                contexts = []
                if isinstance(ev, list):
                    for e in ev:
                        title = "Evidence"
                        text = ""
                        if isinstance(e, dict):
                            title = _safe_str(e.get("title") or title)
                            text = _safe_str(e.get("text") or e.get("evidence"))
                        contexts.append({"title": title, "text": _truncate(text, max_chars)})
                rec = {"claim": claim, "label": label, "evidence": contexts}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
                if limit and cnt >= limit:
                    break
            except Exception:
                continue

def convert_webqsp(in_file: str, out_path: str, limit: Optional[int] = None, max_chars: int = 2000):
    try:
        with open(in_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = {}
    qs = obj.get("Questions") or obj.get("questions") or []
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for q in qs:
            try:
                qt = _safe_str(q.get("QuestionText") or q.get("question"))
                ans = q.get("Answers") or []
                answers = []
                if isinstance(ans, list):
                    for a in ans:
                        val = a.get("AnswerArgument") if isinstance(a, dict) else None
                        if isinstance(val, str) and val:
                            answers.append(val)
                rec = {"question": qt, "answers": answers, "contexts": [{"title": "WebQSP", "text": _truncate("", max_chars)}]}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
                if limit and cnt >= limit:
                    break
            except Exception:
                continue

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    c1 = sub.add_parser("convert-nq")
    c1.add_argument("--out", type=str, required=True)
    c1.add_argument("--split", type=str, default="validation")
    c1.add_argument("--limit", type=int, default=None)
    c1.add_argument("--max-chars", type=int, default=2000)
    c2 = sub.add_parser("convert-triviaqa")
    c2.add_argument("--out", type=str, required=True)
    c2.add_argument("--split", type=str, default="validation")
    c2.add_argument("--limit", type=int, default=None)
    c2.add_argument("--max-chars", type=int, default=2000)
    c3 = sub.add_parser("convert-fever")
    c3.add_argument("--out", type=str, required=True)
    c3.add_argument("--split", type=str, default="validation")
    c3.add_argument("--limit", type=int, default=None)
    c3.add_argument("--max-chars", type=int, default=2000)
    c4 = sub.add_parser("convert-webqsp")
    c4.add_argument("--file", type=str, required=True)
    c4.add_argument("--out", type=str, required=True)
    c4.add_argument("--limit", type=int, default=None)
    c4.add_argument("--max-chars", type=int, default=2000)
    args = p.parse_args()
    if args.cmd == "convert-nq":
        convert_nq(args.out, split=args.split, limit=args.limit, max_chars=args.max_chars)
    elif args.cmd == "convert-triviaqa":
        convert_triviaqa_rc(args.out, split=args.split, limit=args.limit, max_chars=args.max_chars)
    elif args.cmd == "convert-fever":
        convert_fever(args.out, split=args.split, limit=args.limit, max_chars=args.max_chars)
    elif args.cmd == "convert-webqsp":
        convert_webqsp(args.file, args.out, limit=args.limit, max_chars=args.max_chars)

if __name__ == "__main__":
    main()