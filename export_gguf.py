"""Convert a trained checkpoint to GGUF (v3) for llama.cpp / koboldcpp.

This is a thin wrapper around llama.cpp's own convert_hf_to_gguf.py, which
is the known-good path and writes GGUF v3 by default. You need a llama.cpp
clone:

    git clone https://github.com/ggml-org/llama.cpp

Then:

    python export_gguf.py --llama-cpp ../llama.cpp --outtype q8_0
"""
import argparse
import os
import sys
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints", help="HF checkpoint directory")
    ap.add_argument("--llama-cpp", required=True, help="path to a llama.cpp clone")
    ap.add_argument("--outfile", default="tinyllm.gguf")
    ap.add_argument(
        "--outtype",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0"],
        help="f16 is a safe default; q8_0 is smaller with negligible quality loss",
    )
    args = ap.parse_args()

    script = os.path.join(args.llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.exists(script):
        sys.exit(
            f"convert_hf_to_gguf.py not found at {script}\n"
            "Clone it first:  git clone https://github.com/ggml-org/llama.cpp"
        )

    cmd = [
        sys.executable, script, args.model,
        "--outfile", args.outfile,
        "--outtype", args.outtype,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nWrote {args.outfile} — load it with koboldcpp or `llama-cli -m {args.outfile}`")


if __name__ == "__main__":
    main()
