"""Tests for the tokenizer and GPT model.

TODO: once tokenizer.py and model.py exist, add tests (use the "debug" preset so they run fast) that:
- check decode(encode(s)) == s
- check output logits have shape (batch, block_size, vocab_size)
- check the loss at init is close to ln(vocab_size)
- check causality: changing token t must not change the logits at positions < t
"""
