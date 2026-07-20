"""tinyllm model definition — a Qwen3 model trained cold (from random weights).

Qwen3 is, mechanically, "llama + QK-norm + tied embeddings". The QK-norm
buys us training stability at tiny scale, and tied embeddings stop the
token table from eating our whole parameter budget. See README.md.

Two presets:
  nano  (~7M)  — for a no-AVX 2-core CPU; fast enough to retrain repeatedly
  micro (~44M) — the "someday on better hardware" target
"""
from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB_SIZE = 8000     # set by YOUR tokenizer (train_tokenizer.py), not Qwen's 151k
BLOCK_SIZE = 256      # training context window; compute/step scales with this
DEFAULT_SIZE = "nano"

PRESETS = {
    # name:  hidden / inter / layers / heads / kv-heads   (Qwen3 head_dim defaults to 128)
    "nano":  dict(hidden_size=256, intermediate_size=768,  num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=2),   # ~8M
    "mini":  dict(hidden_size=384, intermediate_size=1152, num_hidden_layers=8,
                  num_attention_heads=6, num_key_value_heads=2),   # ~20M
    "micro": dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=12,
                  num_attention_heads=8, num_key_value_heads=2),   # ~48M
}


def build_config(vocab_size: int = VOCAB_SIZE, size: str = DEFAULT_SIZE) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        max_position_embeddings=BLOCK_SIZE,
        tie_word_embeddings=True,      # critical at this scale
        rope_theta=10000.0,
        # one token does triple duty: bos / eos / pad (id 0 from the tokenizer)
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **PRESETS[size],
    )


def build_model(vocab_size: int = VOCAB_SIZE, size: str = DEFAULT_SIZE) -> Qwen3ForCausalLM:
    """Cold-init the model. This is real pretraining — no weights are loaded."""
    return Qwen3ForCausalLM(build_config(vocab_size, size))


if __name__ == "__main__":
    for name in PRESETS:
        m = build_model(size=name)
        total = m.num_parameters() / 1e6
        non_emb = m.num_parameters(exclude_embeddings=True) / 1e6
        print(f"{name:6s}: {total:5.1f}M params total  ({non_emb:.1f}M non-embedding)")
