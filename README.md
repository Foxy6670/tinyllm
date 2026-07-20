# tinyllm

A ~44M-param **Qwen3** micro language model, pretrained from a cold init and
exported to GGUF v3 for `llama.cpp` / `koboldcpp`.

See [`idea.md`](idea.md) for the why. The short version: this is *pretraining*
from random weights, not fine-tuning.

## Architecture

| | |
|---|---|
| arch | Qwen3 (= llama + QK-norm + tied embeddings) |
| params | ~44M |
| hidden / layers / heads | 512 / 12 / 8 (head_dim 64) |
| KV heads | 2 (GQA 4:1) |
| MLP | SwiGLU, intermediate 1536 |
| context | 1024 |
| vocab | 16k, your own byte-level BPE |

The big trap at this scale is the embedding table: Qwen3's native 151k vocab
would burn the entire param budget on a lookup table, so we train our own
small tokenizer and tie input/output embeddings.

## Pipeline

```bash
pip install -r requirements.txt

# 0) sanity-check the model definition
python model_config.py

# 1) train a tokenizer on your jsonl
python train_tokenizer.py --data 'data/*.jsonl' --text-field text

# 2) smoke-test the training loop (built-in tiny corpus, no data needed)
python train.py

# 2') the real run
python train.py --data 'data/*.jsonl' --epochs 3

# 3) export to GGUF
git clone https://github.com/ggml-org/llama.cpp
python export_gguf.py --llama-cpp ./llama.cpp --outtype q8_0
```

### Data format

One JSON object per line, with a text field (default key `text`):

```json
{"text": "some document ..."}
{"text": "def add(a, b):\n    return a + b\n"}
```

Use `--text-field content` (etc.) if your key differs.

## Scaling knobs (in `model_config.py`)

- **More capacity (~53M):** `hidden_size=640, num_attention_heads=10, num_hidden_layers=10, intermediate_size=1728`
- **Faster experiment cycles (~32M):** `num_hidden_layers=8`
