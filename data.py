"""Shared corpus loader for tinyllm (Boonie session logs).

One place that knows how to turn the raw jsonl zoo into clean text documents,
used by both train_tokenizer.py and train.py so the two never drift.

What it handles:
  * multiple schemas: {"messages":[...]} sessions, and flat telegram records
    ({"direction","from","text",...})
  * the role mislabel: in the messages logs, role "user" is actually the
    ENVIRONMENT/harness output, not a human. We relabel it. The only real
    human input is the telegram channel.
  * UUID noise: hashes like 6c7661fc-... are unlearnable for a tiny model, so
    they collapse to a single <id> token (~7% of the corpus).
  * duplication: boonie_corpus.jsonl appears to aggregate sessions that also
    exist standalone, so we dedup at the DOCUMENT level (by content), which
    also subsumes exact-duplicate files.

Scopes (what the model is graded on as next-token prediction):
  voice     -> system identity + Boonie's turns + telegram   (default; ~0.85M tok)
  full      -> everything incl. environment output           (~3.0M tok)
  assistant -> only Boonie's turns                            (~0.62M tok)
"""
import glob
import hashlib
import json
import os
import re

# Atomic role markers — registered as special tokens so the tokenizer never
# splits them and the model can switch "voices" with a single token.
EOS = "<|endoftext|>"
ID = "<id>"
SYS = "<|system|>"
ENV = "<|env|>"
FOXO = "<|foxo|>"
BOONIE = "<|boonie|>"
CODE = "<|code|>"
SPECIAL_TOKENS = [EOS, ID, SYS, ENV, FOXO, BOONIE, CODE]

SCOPES = ("voice", "full", "assistant")

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# role (as stored) -> (tag, semantic-role-we-actually-mean)
_ROLE = {
    "system": (SYS, "system"),
    "user": (ENV, "env"),         # the mislabel: "user" == environment output
    "assistant": (BOONIE, "boonie"),
}


def clean(text):
    return _UUID.sub(ID, text or "")


def resolve(paths):
    """Expand a mix of files / dirs / globs into a sorted list of jsonl files."""
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)
        else:
            out += glob.glob(p)
    return sorted(set(out))


def _render(rec, scope):
    """Turn one jsonl record into a document string, or None to skip it."""
    if "doc" in rec:
        return rec["doc"]            # pre-assembled document (see build_corpus.py)
    if "messages" in rec:
        parts = []
        for m in rec["messages"]:
            content = m.get("content")
            if not isinstance(content, str):
                continue
            tag, role = _ROLE.get(m.get("role"), (ENV, "env"))
            if scope == "assistant" and role != "boonie":
                continue
            if scope == "voice" and role == "env":
                continue   # drop environment dumps; keep system identity + Boonie
            parts.append(tag + clean(content).strip())
        return "\n".join(parts) if parts else None

    # flat telegram record: direction "in" == from human (Foxo), "out" == Boonie
    if "text" in rec:
        if scope == "assistant":
            return None
        tag = FOXO if rec.get("direction") == "in" else BOONIE
        text = clean(rec.get("text")).strip()
        return tag + text if text else None

    return None


def iter_documents(paths, scope="voice"):
    """Yield deduplicated document strings (no EOS appended — caller decides)."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    seen = set()
    n_files = n_docs = n_dup = 0
    for path in resolve(paths):
        n_files += 1
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # dedup on the raw record content, independent of scope
                key = hashlib.md5(line.encode("utf-8")).hexdigest()
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                doc = _render(rec, scope)
                if doc:
                    n_docs += 1
                    yield doc
    print(f"[data] scope={scope}: {n_docs} docs from {n_files} files "
          f"({n_dup} duplicate records skipped)")


if __name__ == "__main__":
    import sys

    roots = sys.argv[1:] or ["~/ggmlagent", "~/frontier-boonie/logs"]
    for sc in SCOPES:
        chars = sum(len(d) for d in iter_documents(roots, sc))
        print(f"  -> {sc:9s} ~{chars / 4 / 1e6:.2f}M tokens\n")
