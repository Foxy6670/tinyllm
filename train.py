"""Pretrain tinyllm from a cold init with the HF Trainer.

    python train.py                               # voice scope, nano, default roots
    python train.py --scope full --epochs 3
    python train.py --data ~/ggmlagent --size micro

Document assembly (relabeling, UUID-stripping, dedup, scope) lives in data.py.
"""
import argparse
import os

# Must be set before any tokenizer use: combining num_proc (below) with a fast
# (Rust) tokenizer's own internal thread pool risks a fork/thread deadlock.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import queue
import re
import shutil
import threading
from itertools import chain

import datasets
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from data import SCOPES, iter_documents
from model_config import BLOCK_SIZE, PRESETS, build_model

# A live tqdm bar per map() call is fine on a terminal, but at num_proc>1 that's
# one concurrently-redrawing bar per worker, streamed into a Colab output cell —
# enough rendering churn to make the browser tab itself lock up. The stage-level
# prints already in this script (doc/file counts, block counts) are enough signal.
datasets.disable_progress_bars()


class AsyncCheckpointSync(TrainerCallback):
    """Mirrors each local checkpoint to a (typically Drive-mounted, slow) directory
    on a background thread, so a slow write only ever competes for CPU/bandwidth —
    it never pauses training the way saving straight to that directory would.

    Trainer's own save_total_limit is disabled locally when this is active (see
    main()); a local checkpoint is only deleted after its copy is confirmed on
    sync_dir, so a lagging upload can never lose a checkpoint to local rotation.
    Retention (`keep`) is enforced on the sync_dir side instead.
    """

    def __init__(self, sync_dir: str, keep: int):
        self.sync_dir = sync_dir
        self.keep = keep
        os.makedirs(sync_dir, exist_ok=True)
        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            local_dir = self._q.get()
            if local_dir is None:
                self._q.task_done()
                return
            try:
                self._sync_one(local_dir)
            except Exception as e:
                print(f"[async-sync] failed to sync {local_dir}: {e}")
            finally:
                self._q.task_done()

    def _sync_one(self, local_dir: str):
        name = os.path.basename(local_dir)
        dest = os.path.join(self.sync_dir, name)
        tmp = dest + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(local_dir, tmp)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(tmp, dest)                       # atomic-ish swap once fully copied
        shutil.rmtree(local_dir, ignore_errors=True)  # safe: it's on sync_dir now

        ckpts = sorted(
            (int(m.group(1)), n)
            for n in os.listdir(self.sync_dir)
            if (m := re.fullmatch(r"checkpoint-(\d+)", n))
        )
        for _, n in ckpts[:-self.keep]:
            shutil.rmtree(os.path.join(self.sync_dir, n), ignore_errors=True)

    def on_save(self, args, state, control, **kwargs):
        local_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._q.put(local_dir)

    def wait(self):
        """Block until every queued sync has completed."""
        self._q.join()
        self._q.put(None)
        self._worker.join()


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
    ap.add_argument("--save-total-limit", type=int, default=1)
    ap.add_argument("--sync-out", default=None,
                     help="mirror checkpoints here on a background thread (e.g. a "
                          "Drive-mounted dir); --out becomes a fast local scratch dir "
                          "and --save-total-limit is enforced on --sync-out instead")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)   # use all logical cores (SMT helps here)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token

    # Stream straight into the Arrow table instead of materializing the whole
    # corpus as a Python list first — at 19GB+ that list plus its Arrow copy
    # existing simultaneously is what was blowing out RAM.
    def _doc_gen():
        for d in iter_documents(args.data, args.scope):
            yield {"text": d}

    ds = Dataset.from_generator(_doc_gen)
    if len(ds) == 0:
        raise SystemExit(f"No documents found for scope={args.scope} in {args.data}")

    # 1) tokenize, appending EOS as a document separator
    def tok_fn(batch):
        return tokenizer([t + eos for t in batch["text"]])

    n_proc = os.cpu_count() or 1
    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names, num_proc=n_proc)

    # 2) concatenate everything and chop into fixed-length blocks
    block = args.block_size

    def group(batch):
        ids = list(chain(*batch["input_ids"]))
        total = (len(ids) // block) * block
        chunks = [ids[i : i + block] for i in range(0, total, block)]
        return {"input_ids": chunks, "attention_mask": [[1] * block for _ in chunks]}

    ds = ds.map(group, batched=True, num_proc=n_proc)
    if len(ds) == 0:
        raise SystemExit(
            f"Not enough text to fill a single {block}-token block. "
            "Add more data or lower --block-size."
        )
    print(f"{len(ds)} blocks x {block} tokens  (~{len(ds) * block / 1e6:.2f}M tokens)")

    model = build_model(vocab_size=tokenizer.vocab_size, size=args.size)
    print(f"Model: {model.num_parameters() / 1e6:.1f}M params")

    # If resuming and a checkpoint already made it to --sync-out (e.g. after a
    # disconnect wiped the local scratch disk), pull the latest one back down
    # before training starts so we can hand it to Trainer as resume_from_checkpoint.
    resume_from = None
    if args.sync_out:
        os.makedirs(args.sync_out, exist_ok=True)
        existing = sorted(
            (int(m.group(1)), n)
            for n in os.listdir(args.sync_out)
            if (m := re.fullmatch(r"checkpoint-(\d+)", n))
        )
        if existing:
            _, name = existing[-1]
            resume_from = os.path.join(args.out, name)
            print(f"[async-sync] resuming from {name} found on {args.sync_out}")
            shutil.copytree(os.path.join(args.sync_out, name), resume_from, dirs_exist_ok=True)

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
        # When syncing async, local rotation is disabled (AsyncCheckpointSync deletes
        # a local checkpoint only once it's confirmed on sync_out, and enforces
        # save_total_limit there instead) so a slow upload can never race a local delete.
        save_total_limit=None if args.sync_out else args.save_total_limit,
        report_to="none",
    )

    sync_cb = None
    callbacks = []
    if args.sync_out:
        sync_cb = AsyncCheckpointSync(args.sync_out, keep=args.save_total_limit)
        callbacks.append(sync_cb)

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator,
                       callbacks=callbacks)
    trainer.train(resume_from_checkpoint=resume_from)

    if sync_cb is not None:
        sync_cb.wait()   # drain any checkpoints still in flight before the final save

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    if args.sync_out:
        for item in os.listdir(args.out):
            s, d = os.path.join(args.out, item), os.path.join(args.sync_out, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"Final model synced to {args.sync_out}/")

    print(f"Saved model + tokenizer to {args.out}/  (now run export_gguf.py)")


if __name__ == "__main__":
    main()
