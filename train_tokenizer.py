"""Train a byte-level BPE tokenizer on the Boonie corpus.

Byte-level BPE (GPT-2 / Qwen style) never hits an unknown token and keeps
whitespace/braces explicit, which matters for the command + code fragments in
Boonie's turns. Role markers (<|boonie|> etc.) and <id> are registered as
special tokens so they stay atomic. Document assembly / cleaning lives in
data.py.

Usage:
    python train_tokenizer.py                          # default roots, voice scope
    python train_tokenizer.py --scope full --vocab-size 8000
"""
import argparse

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from data import EOS, REASONING_SPECIAL_TOKENS, SCOPES, SPECIAL_TOKENS, iter_documents
from model_config import VOCAB_SIZE

SCHEMES = {"boonie": SPECIAL_TOKENS, "reasoning": REASONING_SPECIAL_TOKENS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["~/ggmlagent", "~/frontier-boonie/logs"],
                    help="jsonl files / dirs / globs")
    ap.add_argument("--scope", default="voice", choices=SCOPES,
                     help="ignored for pre-assembled {\"doc\": ...} jsonl (e.g. "
                          "build_reasoning_corpus.py output) - _render() passes "
                          "those through verbatim regardless of scope")
    ap.add_argument("--scheme", default="boonie", choices=list(SCHEMES),
                     help="special-token set: 'boonie' (legacy voice-scheme "
                          "tags) or 'reasoning' (build_reasoning_corpus.py's "
                          "<|user|>/<|assistant|>/<|tool|>/<think> tag set)")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--out", default="tokenizer")
    args = ap.parse_args()

    special_tokens = SCHEMES[args.scheme]
    print(f"Training {args.vocab_size}-token BPE (scope={args.scope}, scheme={args.scheme})...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,                  # EOS -> id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_documents(args.data, args.scope), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=EOS,
        eos_token=EOS,
        pad_token=EOS,
        unk_token=None,
        additional_special_tokens=special_tokens[1:],   # the role markers + <id>
    )
    fast.save_pretrained(args.out)
    print(f"Saved {fast.vocab_size}-token tokenizer to {args.out}/")


if __name__ == "__main__":
    main()
