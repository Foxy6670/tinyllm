"""Measure REAL training throughput on this machine before committing to a run.

Estimates are worthless on an unusual CPU (no AVX2/FMA) — so just run a few
real forward+backward steps and report tokens/sec and peak RAM:

    python bench.py --size nano
    python bench.py --size nano --block-size 256 --batch-size 4

Use the printed tok/s to sanity-check how long a real epoch will take.
"""
import argparse
import resource
import time

import torch

from model_config import BLOCK_SIZE, PRESETS, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="nano", choices=list(PRESETS))
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="0 = leave torch default")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch using {torch.get_num_threads()} thread(s)")

    model = build_model(vocab_size=args.vocab_size, size=args.size)
    model.train()
    params_m = model.num_parameters() / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    B, T = args.batch_size, args.block_size
    ids = torch.randint(0, args.vocab_size, (B, T))
    batch = {"input_ids": ids, "labels": ids}

    def step():
        opt.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        opt.step()

    for _ in range(3):          # warmup (lets BLAS settle)
        step()

    t0 = time.time()
    for _ in range(args.steps):
        step()
    dt = time.time() - t0

    toks = args.steps * B * T
    tps = toks / dt
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on Linux

    print(f"\n{args.size}: {params_m:.1f}M params | batch {B} x {T} tok")
    print(f"  {tps:6.0f} tok/s   ({dt / args.steps * 1000:.0f} ms/step)")
    print(f"  peak RAM: {peak_mb:.0f} MB")
    for n_tok in (1e6, 5e6, 20e6):
        print(f"  1 epoch over {n_tok/1e6:>4.0f}M tokens -> {n_tok / tps / 3600:5.1f} h")


if __name__ == "__main__":
    main()
