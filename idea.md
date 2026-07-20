# tinyllm
## A tiny LLM that can run on just about anything.


## What?
  - This is a project to train a small-language model (SLM) (gguf arch) from a blank slate of training data.

## Why?
  - When you hear someone talking about "training a model", they're almost always talking about fine-tuning, which is just a _later stage_ of training.
  - My hope is that this project will give me an idea of what goes into training a GGUF language model and how it works.
  - I've been in the AI feild and have been watching it develop since practically day-0, but have never had the chance to get the same level of hands-on experience as other more fortunate people.

## Scope?
  - The project will cover creating a small language model that `llama.cpp` or `koboldcpp` (a fork of `llama.cpp`) can load.
  - We are targeting a classing text-to-text input-output.
  - Thinking/reasoning will be covered in a future iteration, as that is an entire different stage of generation.

## Goal?
  - Determine how different (or even shuffled) training data affects the output in different scenarios.
  - Learn the fundamentals of language model technology.
  - Potentially create a hyper-compressed language to think more densely through complex/difficult scenarios... oh wait... Claude Mythos beat me to that. Oh well.

## Architecture?
  - I haven't decided on this yet. I'm stuck between `qwen3`, a variant of `llama`, or an entirely custom (or "archless") design altogether.
  - Container should be GGUF v3 (easily llama.cpp loadable).
