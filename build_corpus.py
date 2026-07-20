"""Assemble a mixed corpus (Boonie voice + source code) into one portable file.

Output is a jsonl of {"doc": <full tagged document>} lines that both
train_tokenizer.py and train.py consume verbatim (data.py passes "doc" through
untouched). This lets us build the mix on sc4 — where the data lives — and ship
a single small file to the Gateway, instead of copying raw source trees around.

    python build_corpus.py --code ~/ggmlagent ~/furnace ~/bfbitcoin \
        ~/luautonomous ~/aibike ~/picoir --out corpus_codemix.jsonl
"""
import argparse
import json
import os

from data import CODE, clean, iter_documents

SKIP_DIRS = (".venv", "__pycache__", "node_modules", ".git")


def gather_code(roots, exts, max_kb):
    exts = tuple("." + e.strip().lstrip(".") for e in exts)
    docs = []
    for root in roots:
        root = os.path.expanduser(root)
        base = os.path.dirname(root.rstrip("/"))
        for dirpath, _, files in os.walk(root):
            if any(s in dirpath for s in SKIP_DIRS):
                continue
            for fn in files:
                if not fn.endswith(exts):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(path) > max_kb * 1024:
                        continue
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError:
                    continue
                if content.strip():
                    rel = os.path.relpath(path, base)
                    docs.append(f"{CODE}# {rel}\n{clean(content)}")
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+",
                    default=["~/ggmlagent", "~/frontier-boonie/logs"])
    ap.add_argument("--scope", default="voice")
    ap.add_argument("--code", nargs="*", default=[], help="source dirs to fold in")
    ap.add_argument("--code-ext", default="py,sh")
    ap.add_argument("--code-max-kb", type=int, default=64)
    ap.add_argument("--out", default="corpus_codemix.jsonl")
    args = ap.parse_args()

    voice = list(iter_documents(args.jsonl, args.scope))
    code = gather_code(args.code, args.code_ext.split(","), args.code_max_kb)

    vch = sum(len(d) for d in voice)
    cch = sum(len(d) for d in code)
    tot = vch + cch or 1
    with open(args.out, "w", encoding="utf-8") as f:
        for d in voice + code:
            f.write(json.dumps({"doc": d}) + "\n")

    print(f"voice: {len(voice):4d} docs  ~{vch/4/1e6:.2f}M tok  ({100*vch/tot:.0f}% of chars)")
    print(f"code : {len(code):4d} docs  ~{cch/3.5/1e6:.2f}M tok  ({100*cch/tot:.0f}% of chars)")
    print(f"wrote {args.out}  ({len(voice)+len(code)} docs)")


if __name__ == "__main__":
    main()
