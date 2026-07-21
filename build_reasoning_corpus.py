"""Assemble the ~4.3B-token general reasoning+agentic base corpus from public
HF datasets (measured, not estimated - see conversation for the sampling that
picked these sources and ratios):

  - nvidia/Nemotron-Agentic-v1  (both subsets, used in full - the agentic/
    tool-use trajectories that most directly match the "make it agentic" goal)
  - open-thoughts/OpenThoughts3-1.2M  (math/code/science CoT, R1-distilled;
    26B tokens available, sliced down - this is NOT Boonie-flavored data,
    it's a general reasoning-competence base; Boonie's voice is a separate,
    much smaller, later fine-tune stage - see boonie-finetune-goals memory)
  - HuggingFaceFW/fineweb-edu sample-10BT  (general web text, sliced, so the
    model isn't purely STEM/agentic-brained)

New tags distinct from Boonie's (<|boonie|>/<|foxo|>/<|env|>) so this general
base and Boonie's specific voice never collide when we later fine-tune on top:
  <|system|>  (reused name, same concept: a system/policy prompt)
  <|user|>    generic human turn
  <|assistant|>  generic AI-response turn (may already contain literal
                 <think>...</think> text from R1 distillation - left as-is)
  <|tool|>    a tool/function's result being fed back to the assistant

Streams everything (no full-dataset download) and stops each source once its
OWN measured token budget (via the tokenizer you pass in) is hit - budgets are
targets, not exact, since real per-row length varies.

    python build_reasoning_corpus.py --tokenizer tokenizer-codemix \
        --out corpus_reasoning.jsonl \
        --openthoughts-tokens 1.81e9 --fineweb-tokens 0.78e9
"""
import argparse
import itertools
import json
import re

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

SYS, USER, ASSISTANT, TOOL = "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"

# same UUID pattern as data.py's clean(), plus bare 32-hex-char ids (no dashes) -
# e.g. Nemotron's tool_call_id "chatcmpl-tool-877c1a2d5cb149f685397719015694ed".
# Collapsing both sides of a call/result pair to the same <id> token preserves
# the *pairing structure* while removing the unlearnable entropy (same idea as
# Boonie's UUID scrub in data.py).
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HEX32 = re.compile(r"\b[0-9a-f]{32}\b")


def to_text(v):
    """Nemotron's fields (e.g. tool-call 'arguments') are inconsistently typed -
    sometimes a JSON string, sometimes an already-parsed dict/list. Coerce to
    text before cleaning so clean()'s regex .sub() never sees a non-string."""
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v)


def clean(text):
    return _HEX32.sub("<id>", _UUID.sub("<id>", to_text(text)))


def as_list(v):
    return json.loads(v) if isinstance(v, str) else v


def render_openthoughts(row):
    parts = []
    for turn in as_list(row["conversations"]):
        tag = USER if turn["from"] == "human" else ASSISTANT
        parts.append(tag + clean(turn["value"]).strip())
    return "\n".join(parts)


def render_nemotron(row):
    parts = []
    for m in as_list(row["messages"]):
        role = m.get("role")
        if role == "tool":
            parts.append(TOOL + clean(m.get("content", "")).strip())
            continue
        tag = {"system": SYS, "user": USER, "assistant": ASSISTANT}.get(role)
        if tag is None:
            continue
        chunks = []
        reasoning = m.get("reasoning_content")
        if reasoning:
            chunks.append(f"<think>{clean(reasoning).strip()}</think>")
        for call in (m.get("tool_calls") or []):
            fn = call.get("function", {})
            chunks.append(f"CALL {fn.get('name')}({clean(fn.get('arguments', ''))})")
        content = m.get("content")
        if content:
            chunks.append(clean(content).strip())
        if chunks:
            parts.append(tag + " ".join(chunks))
    return "\n".join(parts)


def render_fineweb(row):
    return clean(row["text"]).strip()


def iter_raw_jsonl(repo_id, filename):
    """Download+iterate a dataset's raw .jsonl file as plain JSON lines, bypassing
    datasets' pyarrow streaming - Nemotron's per-tool JSON-schema fields vary in
    nested shape across rows, which breaks pyarrow's cross-shard type unification.
    Plain json.loads() doesn't care about type consistency across rows, same as
    how data.py handles Boonie's own jsonl files."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_budget(label, tok, out, render_fn, token_budget=None, raw_jsonl=None,
                   skip_rows=0, **load_kwargs):
    """Stream a dataset, render+write docs, stop at token_budget (None = no cap).
    raw_jsonl=(repo_id, filename) bypasses datasets/pyarrow entirely (see iter_raw_jsonl).
    skip_rows resumes past rows an earlier pull already consumed, so expanding a
    source later doesn't re-fetch or duplicate the same rows."""
    ds = iter_raw_jsonl(*raw_jsonl) if raw_jsonl else load_dataset(streaming=True, **load_kwargs)
    if skip_rows:
        ds = ds.skip(skip_rows)
        print(f"  [{label}] skipping first {skip_rows} rows (already used)")
    total_tok = n_docs = 0
    for row in ds:
        try:
            doc = render_fn(row)
        except Exception:
            continue
        if not doc:
            continue
        n = len(tok(doc)["input_ids"])
        out.write(json.dumps({"doc": doc}) + "\n")
        total_tok += n
        n_docs += 1
        if token_budget and total_tok >= token_budget:
            break
        if n_docs % 5000 == 0:
            print(f"  [{label}] {n_docs} docs, {total_tok/1e6:.1f}M tokens so far...")
    print(f"[{label}] DONE: {n_docs} docs, {total_tok/1e6:.1f}M tokens")
    return total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer-codemix",
                     help="used to MEASURE token budgets during assembly; the final "
                          "training tokenizer should be retrained on a sample of the "
                          "output, which may shift real counts slightly")
    ap.add_argument("--out", default="corpus_reasoning.jsonl")
    ap.add_argument("--only", choices=["nemotron", "openthoughts", "fineweb"], default=None,
                     help="run just one source (for running the 3 sources as separate "
                          "concurrent processes - they're independent, CPU-bound work, "
                          "and don't share any state)")
    ap.add_argument("--nemotron-tokens", type=float, default=None,
                     help="cap for testing; real run wants None (use all of it)")
    ap.add_argument("--openthoughts-tokens", type=float, default=1.81e9)
    ap.add_argument("--fineweb-tokens", type=float, default=0.78e9)
    ap.add_argument("--openthoughts-skip", type=int, default=0,
                     help="rows an earlier pull already consumed - resume past them")
    ap.add_argument("--fineweb-skip", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    grand_total = 0

    with open(args.out, "w", encoding="utf-8") as out:
        if args.only in (None, "nemotron"):
            print("=== Nemotron-Agentic-v1 (both subsets, full - no cap) ===")
            for subset in ("interactive_agent", "tool_calling"):
                grand_total += stream_budget(
                    f"nemotron/{subset}", tok, out, render_nemotron,
                    token_budget=args.nemotron_tokens,
                    raw_jsonl=("nvidia/Nemotron-Agentic-v1", f"data/{subset}.jsonl"),
                )

        if args.only in (None, "openthoughts"):
            print(f"\n=== OpenThoughts3-1.2M (capped at {args.openthoughts_tokens/1e9:.2f}B tokens) ===")
            grand_total += stream_budget(
                "openthoughts3", tok, out, render_openthoughts,
                token_budget=args.openthoughts_tokens, skip_rows=args.openthoughts_skip,
                path="open-thoughts/OpenThoughts3-1.2M", split="train",
            )

        if args.only in (None, "fineweb"):
            print(f"\n=== FineWeb-Edu sample-10BT (capped at {args.fineweb_tokens/1e9:.2f}B tokens) ===")
            grand_total += stream_budget(
                "fineweb-edu", tok, out, render_fineweb,
                token_budget=args.fineweb_tokens, skip_rows=args.fineweb_skip,
                path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
            )

    print(f"\nGRAND TOTAL: ~{grand_total/1e9:.2f}B tokens (measured with {args.tokenizer}) -> {args.out}")


if __name__ == "__main__":
    main()
