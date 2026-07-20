"""Pretrain tinyllm from a cold init with the HF Trainer.

    python train.py                               # voice scope, nano, default roots
    python train.py --scope full --epochs 3
    python train.py --data ~/ggmlagent --size micro

Document assembly (relabeling, UUID-stripping, dedup, scope) lives in data.py.
"""
import argparse
import os
from itertools import chain

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from data import SCOPES, iter_documents
from model_config import BLOCK_SIZE, PRESETS, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["~/ggmlagent", "~/frontier-boonie/logs"],
                    help="jsonl files / dirs / globs")
    ap.add_argument("--scope", default="voice", choices=SCOPES)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--size", default="nano", choices=list(PRESETS))
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help=">0 overrides --epochs; handy for a quick sanity run")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)   # use all logical cores (SMT helps here)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token

    docs = list(iter_documents(args.data, args.scope))
    if not docs:
        raise SystemExit(f"No documents found for scope={args.scope} in {args.data}")
    ds = Dataset.from_dict({"text": docs})

    # 1) tokenize, appending EOS as a document separator
    def tok_fn(batch):
        return tokenizer([t + eos for t in batch["text"]])

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    # 2) concatenate everything and chop into fixed-length blocks
    block = args.block_size

    def group(batch):
        ids = list(chain(*batch["input_ids"]))
        total = (len(ids) // block) * block
        chunks = [ids[i : i + block] for i in range(0, total, block)]
        return {"input_ids": chunks, "attention_mask": [[1] * block for _ in chunks]}

    ds = ds.map(group, batched=True)
    if len(ds) == 0:
        raise SystemExit(
            f"Not enough text to fill a single {block}-token block. "
            "Add more data or lower --block-size."
        )
    print(f"{len(ds)} blocks x {block} tokens  (~{len(ds) * block / 1e6:.2f}M tokens)")

    model = build_model(vocab_size=tokenizer.vocab_size, size=args.size)
    print(f"Model: {model.num_parameters() / 1e6:.1f}M params")

    use_bf16 = torch.cuda.is_available()
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
        adam_beta2=0.95,            # the standard LLM-pretraining beta2
        bf16=use_bf16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved model + tokenizer to {args.out}/  (now run export_gguf.py)")


if __name__ == "__main__":
    main()
