"""Generate text from a trained checkpoint.

TODO (Plan step 2.7):
- Load a checkpoint from checkpoints/ and rebuild the model and tokenizer from it.
- Add arguments: --prompt, --max-new-tokens, --temperature, --top-k.
- Temperature: divide the logits by T before softmax (T < 1 is more conservative, T > 1 more random).
- Top-k: keep only the k largest logits and set the rest to -inf before sampling.
- Save a few samples to ../assets/ for the README.
"""
