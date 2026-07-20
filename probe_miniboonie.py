import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ckpt = sys.argv[1]
tok = AutoTokenizer.from_pretrained("tokenizer-v3-3")
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)
model.eval()

# voice-scope framing: system identity + boonie turn, ending mid-action to see the reflex
prompts = [
    "<|system|>You are Boonie, an autonomous AI agent with a Moltbook account.\n<|boonie|>I just posted a comment on the Starfish post about supply chain security.",
    "<|system|>You are Boonie, an autonomous AI agent with a Moltbook account.\n<|boonie|>",
]

for p in prompts:
    ids = tok(p, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    print("PROMPT:", repr(p[-60:]))
    print("CONTINUATION:", repr(text))
    print("---")
