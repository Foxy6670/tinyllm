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
    """Mirrors each local checkpoint to a (typically Drive-mounted, slow, or just
    roomier) directory on a background thread, so a slow write only ever competes
    for CPU/bandwidth - it never pauses training the way saving straight to that
    directory would.

    Trainer's own save_total_limit is disabled locally when this is active (see
    main()); a local checkpoint is only deleted after its copy is confirmed on
    sync_dir, so a lagging upload can never lose a checkpoint to local rotation.

    Retention on the sync_dir side: the most recent `keep_recent` checkpoints
    stay (quick-resume window), PLUS every `keep_every`-th save (by save index,
    i.e. step // save_steps) is kept permanently as a historical artifact -
    never pruned, regardless of age.
    """

    def __init__(self, sync_dir: str, keep_recent: int, keep_every: int, save_steps: int):
        self.sync_dir = sync_dir
        self.keep_recent = keep_recent
        self.keep_every = keep_every
        self.save_steps = save_steps
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
        keep_names = {n for _, n in ckpts[-self.keep_recent:]}       # quick-resume window
        if self.keep_every:
            keep_names |= {n for step, n in ckpts                    # historical artifacts
                           if (step // self.save_steps) % self.keep_every == 0}
        for _, n in ckpts:
            if n not in keep_names:
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
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--keep-recent", type=int, default=1,
                     help="quick-resume window: most recent N checkpoints always kept")
    ap.add_argument("--keep-every", type=int, default=0,
                     help="also keep every Nth save (by save index) permanently as a "
                          "historical artifact, on top of --keep-recent. 0 = disabled. "
                          "Only enforced on --sync-out (local --out is transient scratch "
                          "when --sync-out is active - see AsyncCheckpointSync)")
    ap.add_argument("--sync-out", default=None,
                     help="mirror checkpoints here on a background thread (e.g. a "
                          "Drive-mounted dir, or just a roomier disk); --out becomes a "
                          "fast local scratch dir and retention is enforced on --sync-out")
    ap.add_argument("--resume", action="store_true",
                     help="resume from the latest checkpoint - in --sync-out if set, "
                          "else --out. Without this flag, training always starts fresh "
                          "even if old checkpoints are sitting in --out/--sync-out.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                     help="trade ~20-30% more compute time for much lower activation "
                          "VRAM, by recomputing activations in backward instead of "
                          "storing them. Use when VRAM, not compute, is the wall.")
    ap.add_argument("--8bit-adam", action="store_true", dest="adam8bit",
                     help="quantize AdamW's momentum buffers to 8-bit (bitsandbytes) "
                          "to cut optimizer-state VRAM well below the fp32 default. "
                          "Master weights/grads stay full precision.")
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

    # datasets caches EACH pipeline stage as a separate on-disk Arrow file. At
    # multi-GB corpus scale, all three stages (raw text / tokenized / grouped)
    # coexisting can approach 3x the corpus size. cleanup_stale_cache() below
    # removes a finished stage's file(s) once the next stage no longer needs
    # them, keeping peak disk usage to roughly one stage at a time.
    def cache_files(dataset):
        return {f["filename"] for f in dataset.cache_files}

    def cleanup_stale_cache(old_files, new_files):
        # Only remove files the CURRENT dataset doesn't reference - e.g. shuffle()
        # often just wraps an indices mapping around the same underlying file
        # rather than copying it, so "previous stage" isn't always safe to delete
        # outright; the set difference is what's actually been superseded.
        for f in old_files - new_files:
            if os.path.exists(f):
                os.remove(f)

    # datasets caches Dataset.from_generator()'s output by fingerprint and can try
    # to reuse a cache entry from a previous, unrelated launch (e.g. --resume after
    # a kill/restart). But cleanup_stale_cache() below deletes each stage's cache
    # as soon as the next stage supersedes it, so a cross-launch cache is never
    # safely reusable - it's either already gone or a stale reference to files
    # that cleanup already deleted, which crashes with FileNotFoundError instead
    # of regenerating. Starting every launch with a clean cache dir avoids that.
    shutil.rmtree(datasets.config.HF_DATASETS_CACHE, ignore_errors=True)

    ds = Dataset.from_generator(_doc_gen)
    if len(ds) == 0:
        raise SystemExit(f"No documents found for scope={args.scope} in {args.data}")

    # Shuffle before sharding across num_proc workers. map(num_proc=N) splits the
    # dataset into N CONTIGUOUS index ranges, not random samples - a corpus built
    # by concatenating sources in sequence (e.g. all ~20K-token OpenThoughts3-style
    # docs together, then short FineWeb-Edu paragraphs) can leave one worker's
    # batches full of huge documents while another's are all tiny, blowing that
    # worker's RAM far past the others. Confirmed: this took the TUF down (100%
    # RAM + swap thrashing) on the first attempt at this corpus - don't remove.
    ds = ds.shuffle(seed=42)
    stage_files = cache_files(ds)

    # 1) tokenize, appending EOS as a document separator
    def tok_fn(batch):
        return tokenizer([t + eos for t in batch["text"]])

    n_proc = os.cpu_count() or 1
    # Small batch_size caps peak memory per worker regardless of doc length mix -
    # defense in depth alongside the shuffle above, not a substitute for it.
    ds = ds.map(tok_fn, batched=True, batch_size=64, remove_columns=ds.column_names, num_proc=n_proc)
    cleanup_stale_cache(stage_files, cache_files(ds))
    stage_files = cache_files(ds)

    # 2) concatenate everything and chop into fixed-length blocks
    block = args.block_size

    def group(batch):
        ids = list(chain(*batch["input_ids"]))
        total = (len(ids) // block) * block
        chunks = [ids[i : i + block] for i in range(0, total, block)]
        return {"input_ids": chunks, "attention_mask": [[1] * block for _ in chunks]}

    ds = ds.map(group, batched=True, batch_size=256, num_proc=n_proc)
    if len(ds) == 0:
        raise SystemExit(
            f"Not enough text to fill a single {block}-token block. "
            "Add more data or lower --block-size."
        )
    cleanup_stale_cache(stage_files, cache_files(ds))
    print(f"{len(ds)} blocks x {block} tokens  (~{len(ds) * block / 1e6:.2f}M tokens)")

    # tokenizer.vocab_size only reflects the base BPE model's own size and silently
    # excludes any tokens appended via add_special_tokens() beyond that range - use
    # len(tokenizer) so the model's embedding table always matches the real id space.
    model = build_model(vocab_size=len(tokenizer), size=args.size, block_size=args.block_size)
    print(f"Model: {model.num_parameters() / 1e6:.1f}M params")

    # Native bf16 weights instead of the classic fp32-master-weights + autocast
    # recipe. That recipe exists for fp16's narrow dynamic range (5 exponent
    # bits - small updates could underflow); bf16 shares fp32's 8 exponent
    # bits, so the original justification doesn't apply. Halves weight+grad
    # VRAM (measured ~500MB saved on the 111M large preset) with the loss
    # itself still computed in fp32 (ForCausalLMLoss upcasts logits
    # unconditionally, confirmed against the installed transformers version).
    use_bf16 = torch.cuda.is_available()
    if use_bf16:
        model = model.to(torch.bfloat16)

    # --resume gates this explicitly: without it, training always starts fresh,
    # even if --out/--sync-out already has checkpoints sitting in it (e.g. from
    # a previous unrelated run reusing the same paths).
    resume_from = None
    if args.resume and args.sync_out:
        # local scratch disk may have been wiped between sessions - pull the
        # latest checkpoint back down from sync_out before handing it to Trainer.
        os.makedirs(args.sync_out, exist_ok=True)
        existing = sorted(
            (int(m.group(1)), n)
            for n in os.listdir(args.sync_out)
            if (m := re.fullmatch(r"checkpoint-(\d+)", n))
        )
        if existing:
            _, name = existing[-1]
            resume_from = os.path.join(args.out, name)
            print(f"[resume] found {name} on --sync-out, copying to local scratch...")
            shutil.copytree(os.path.join(args.sync_out, name), resume_from, dirs_exist_ok=True)
        else:
            print("[resume] --resume given but no checkpoint found on --sync-out; starting fresh")
    elif args.resume:
        resume_from = True   # Trainer auto-detects the latest checkpoint in --out

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
        # bf16 autocast is redundant/off here - the model's own weights are
        # already native bf16 (see above) when CUDA is available.
        gradient_checkpointing=args.grad_checkpoint,
        optim="adamw_bnb_8bit" if args.adam8bit else "adamw_torch",
        logging_steps=10,
        save_steps=args.save_steps,
        # When syncing async, local rotation is disabled (AsyncCheckpointSync deletes
        # a local checkpoint only once it's confirmed on sync_out, and enforces
        # retention there instead) so a slow upload can never race a local delete.
        save_total_limit=None if args.sync_out else args.keep_recent,
        report_to="none",
    )

    sync_cb = None
    callbacks = []
    if args.sync_out:
        sync_cb = AsyncCheckpointSync(args.sync_out, keep_recent=args.keep_recent,
                                       keep_every=args.keep_every, save_steps=args.save_steps)
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
