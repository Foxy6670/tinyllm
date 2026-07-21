"""Fix third-person self-narration in the reasoning/agentic corpus.

Adapted from ~/ggmlagent/extract_training.py's _fix_third_person, but for a
DIFFERENT problem than Boonie's own logs. Boonie's fix targets "the user" as
a self-reference, because Boonie is an autonomous agent with no real separate
user in scope for most of a session - so "the user" in its own <think> is
almost always a confusion. THIS corpus (Nemotron/OpenThoughts3) is the
opposite: it's customer-service/Q&A data with a genuinely distinct human in
nearly every conversation, so "the user" there is correct and left alone.

The actual analogous problem here (confirmed empirically - see conversation):
Nemotron's synthetic generation writes reasoning_content in a narrator voice
describing "the assistant"/"the AI"/"the model"/"the agent" as an external
entity (830K+ hits), instead of the assistant reasoning as "I". Left in, this
would train the model to narrate its own actions from a spectator's
perspective - the opposite of the thought-to-action linking this corpus
exists to build.

Usage:
    python fix_dissociation.py corpus_nemotron.jsonl corpus_nemotron.fixed.jsonl
    python fix_dissociation.py corpus_openthoughts.jsonl corpus_openthoughts.fixed.jsonl
"""
import argparse
import json
import re

# "the agent" deliberately excluded: in this customer-service data it's
# genuinely overloaded to mean a HUMAN rep too ("A human agent will handle
# verification... the agent will document the cancellation"), including inside
# the model's own <think> when it reasons about escalating/handing off. Too
# ambiguous to blindly rewrite - confirmed by inspecting real output, not just
# hit-counting (see conversation). "the assistant"/"the AI"/"the model" don't
# have that human-meaning collision.
_SUBJECTS = ["the assistant", "the AI", "the model"]

# Verb-form pairs first, so the catch-all ("the assistant" -> "I") doesn't
# consume matches that need a different conjugation - same ordering principle
# as Boonie's _THIRD_PERSON_SUBS.
_VERB_PAIRS = [
    ("is", "am"), ("was", "was"), ("has", "have"), ("had", "had"),
    ("does", "do"), ("doesn't", "don't"),
    ("needs", "need"), ("wants", "want"), ("should", "should"),
    ("will", "will"), ("can", "can"), ("might", "might"),
    ("could", "could"), ("would", "would"),
    ("tries", "try"), ("tried", "tried"), ("thinks", "think"),
    ("decides", "decide"), ("decided", "decided"),
]

# "the assistant (me) doesn't have..." - the parenthetical becomes redundant
# once the subject -> "I", and it also hides the verb from the pairs above
# (they only look for a verb immediately after the subject). Rare (~0.2% of
# docs) but cheap to strip before the main substitution runs.
_PAREN_ME = re.compile(r"\b(the (?:assistant|AI|model))\s*\((?:me|myself)\)", re.IGNORECASE)

# object-position guard: "conveyed to the assistant" -> "conveyed to me", not
# the nominative "I" the bare catch-all would otherwise force in. Checked via
# a lookbehind on the catch-all itself rather than trying to enumerate every
# preposition as its own rule.
_PREPOSITIONS = ("to", "for", "with", "from", "at", "by", "of", "about")

# "as the AI, I should..." / "as the assistant, I have..." is an identity-
# affirming apposition, already correctly first-person after the comma - same
# carve-out as extract_training.py's _AI_BIND_RE. Blindly substituting inside
# it produces "as I, I should..." (confirmed - see conversation). A fixed-width
# negative lookbehind on "as " skips these without needing a separate pass.
_NOT_APPOSITIVE = r"(?<!as )(?<!As )"

_SUBS = []
for subj in _SUBJECTS:
    for third, first in _VERB_PAIRS:
        _SUBS.append((re.compile(rf"\b{subj} {third}\b", re.IGNORECASE), f"I {first}"))
    _SUBS.append((re.compile(rf"\b{subj}'s\b", re.IGNORECASE), "my"))
    prep_alt = "|".join(_PREPOSITIONS)
    _SUBS.append((re.compile(rf"\b(?:{prep_alt}) {subj}\b", re.IGNORECASE),
                  lambda m, s=subj: m.group(0)[: -len(s)] + "me"))
    _SUBS.append((re.compile(rf"{_NOT_APPOSITIVE}\b{subj}\b", re.IGNORECASE), "I"))  # catch-all, last


def fix_dissociation(text):
    text = _PAREN_ME.sub(lambda m: m.group(1), text)  # strip the redundant "(me)" first
    for pattern, replacement in _SUBS:
        text = pattern.sub(replacement, text)
    # re-capitalise 'i'/'my' at sentence boundaries, same as extract_training.py.
    # "after <think>" counts as a boundary too (fix_doc passes the tag itself in).
    _BOUNDARY = r"(?:^|(?<=[.!?])\s+|(?<=<think>))"
    text = re.sub(_BOUNDARY + r"(i)\b", lambda m: m.group(0)[:-1] + "I", text)
    text = re.sub(_BOUNDARY + r"(my)\b", lambda m: m.group(0)[:-2] + "My", text)
    return text


_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def fix_doc(doc):
    """Only touch text inside <think>...</think> - the model's own private
    deliberation, where third-person self-narration is the actual problem.
    Everything else (tool results, customer-facing replies, system prompts)
    may legitimately reference other people/entities and is left untouched."""
    return _THINK.sub(lambda m: fix_dissociation(m.group(0)), doc)


_CHECK = re.compile(r"\bthe (?:assistant|ai|model)\b|\bthe (?:assistant|ai|model)'s\b",
                     re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    args = ap.parse_args()

    n_docs = n_fixed = hits_before = hits_after = 0
    with open(args.infile, encoding="utf-8") as fin, open(args.outfile, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            doc = rec["doc"]
            # scope the "before" count to <think> spans only, matching what
            # fix_doc actually touches - text outside <think> is intentionally
            # left alone, so hits there shouldn't count as "still remaining"
            think_spans = _THINK.findall(doc)
            before = sum(len(_CHECK.findall(span)) for span in think_spans)
            if before:
                doc = fix_doc(doc)
                after = sum(len(_CHECK.findall(span)) for span in _THINK.findall(doc))
                hits_before += before
                hits_after += after
                n_fixed += 1
                rec["doc"] = doc
            n_docs += 1
            fout.write(json.dumps(rec) + "\n")
            if n_docs % 20000 == 0:
                print(f"  {n_docs} docs processed, {n_fixed} touched so far...")

    print(f"\n{args.infile} -> {args.outfile}")
    print(f"docs: {n_docs}  touched: {n_fixed}")
    print(f"third-person self-ref hits: {hits_before} -> {hits_after} remaining")


if __name__ == "__main__":
    main()
