"""Fine-tune an EXISTING tinyllm checkpoint (not a cold init) on a smaller,
targeted corpus. Built for the v0 (89.6M) recovery attempt: v0 pretrained on
a corpus with zero non-agentic <|user|> turns (see corpus investigation this
session) and never learned to just answer casually - this continues training
the same checkpoint on SODA alone (pure casual dialogue, no tool/task
framing) to see whether that specific gap can be patched post-hoc.

    python finetune.py --checkpoint ckpt-final-probe --data corpus_soda.jsonl \
        --block-size 1024 --epochs 1.0 --out finetune_out
"""
import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
from itertools import chain

import datasets
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from data import iter_documents

datasets.disable_progress_bars()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="existing model dir to continue training")
    ap.add_argument("--data", nargs="+", required=True, help="jsonl files / dirs / globs")
    ap.add_argument("--out", default="finetune_out")
    ap.add_argument("--block-size", type=int, default=1024,
                     help="shorter than the model's max_position_embeddings is fine - "
                          "RoPE doesn't require matching the pretrain block size, and a "
                          "short-dialogue-shaped fine-tune benefits from not force-packing "
                          "unrelated conversations into one 8192-token block")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                     help=">0 overrides --epochs; for quick throughput probes")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5,
                     help="fine-tune LR, well below the 3e-4 pretrain peak - this is "
                          "continuing an already-trained model, not cold-initing one")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--keep-recent", type=int, default=3)
    ap.add_argument("--num-proc", type=int, default=0,
                     help="dataset .map() worker count; 0 = auto-scale to dataset "
                          "size instead of always maxing out cores (see comment "
                          "at call site - oversubscribing this on a small sample "
                          "is what caused the Gateway swap-thrashing incident)")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    eos = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)  # NOT a cold init
    print(f"Loaded {model.num_parameters() / 1e6:.1f}M-param checkpoint from {args.checkpoint}")

    def _doc_gen():
        for d in iter_documents(args.data, "voice"):
            yield {"text": d}

    shutil.rmtree(datasets.config.HF_DATASETS_CACHE, ignore_errors=True)
    ds = Dataset.from_generator(_doc_gen)
    if len(ds) == 0:
        raise SystemExit(f"No documents found in {args.data}")
    ds = ds.shuffle(seed=42)

    def tok_fn(batch):
        return tokenizer([t + eos for t in batch["text"]])

    # Each worker forks a full python+torch+transformers process (~300-500MB
    # baseline RSS just from imports) - on an 8-thread/6.7GB box, cpu_count()
    # workers for a tiny sample is pure overhead, and stacking multiple such
    # pools across sequential test runs is what pushed Gateway into swap
    # thrashing badly enough to become SSH-unresponsive. Scale workers to
    # actual dataset size instead of always maxing out cores.
    n_proc = args.num_proc if args.num_proc > 0 else min(os.cpu_count() or 1, max(1, len(ds) // 500))
    print(f"[finetune] using num_proc={n_proc} for {len(ds)} docs")
    ds = ds.map(tok_fn, batched=True, batch_size=64, remove_columns=ds.column_names, num_proc=n_proc)

    block = args.block_size

    def group(batch):
        ids = list(chain(*batch["input_ids"]))
        total = (len(ids) // block) * block
        chunks = [ids[i : i + block] for i in range(0, total, block)]
        return {"input_ids": chunks, "attention_mask": [[1] * block for _ in chunks]}

    ds = ds.map(group, batched=True, batch_size=256, num_proc=n_proc)
    if len(ds) == 0:
        raise SystemExit(f"Not enough text to fill a single {block}-token block.")
    print(f"{len(ds)} blocks x {block} tokens  (~{len(ds) * block / 1e6:.2f}M tokens)")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.1,
        adam_beta2=0.95,
        optim="adamw_torch",          # no bitsandbytes here - CPU only, no CUDA
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=args.keep_recent,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved fine-tuned model + tokenizer to {args.out}/")


if __name__ == "__main__":
    main()
