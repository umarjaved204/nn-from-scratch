"""Character-level GPT model.

TODO, built up in stages:

2.2 Bigram baseline
    - An nn.Embedding(vocab_size, vocab_size) that maps each token straight to next-token logits.
    - forward(idx, targets=None) -> (logits, loss), using F.cross_entropy.
    - generate(idx, max_new_tokens): repeatedly sample the next token and append it.
    - Check: loss at init ≈ ln(vocab_size) ≈ 4.17.

2.3 Single causal self-attention head
    - Linear projections to query, key and value (head_size each).
    - Scores = q @ k^T / sqrt(head_size); mask future positions with -inf; softmax; @ v.
    - Check: changing a future token must not change outputs at earlier positions.

2.4 Multi-head attention + feed-forward
    - Run several heads in parallel, concatenate, project back to n_embd.
    - Feed-forward: Linear(n_embd, 4 * n_embd) -> GELU -> Linear(4 * n_embd, n_embd).

2.5 Transformer block and full GPT
    - Block: x = x + attn(ln1(x)); x = x + ffwd(ln2(x))   (pre-LayerNorm + residuals)
    - GPT: token embedding + positional embedding -> N blocks -> final LayerNorm -> lm_head.
    - Take hyperparameters from config.GPTConfig.
"""
