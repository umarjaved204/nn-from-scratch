# Project Plan

## Phase 0 – Setup
- [x] Project structure, `.gitignore`, `LICENSE`, `requirements.txt`
- [x] Create virtual environment and install dependencies
- [x] Install CUDA build of PyTorch and confirm `torch.cuda.is_available()`
- [x] Create GitHub repo and push initial commit

## Phase 1 – NumPy MLP (`part1_numpy/`)
- [x] 1.1 Forward pass: `Linear`, `ReLU`, softmax + cross-entropy (log-sum-exp) — *check: loss ≈ ln(10) ≈ 2.30 at init*
- [x] 1.2 Backward pass for each layer — *check: softmax-CE gradient = `probs − one_hot`*
- [x] 1.3 Gradient check against finite differences — *check: relative error < 1e-6*
- [x] 1.4 Training loop: He init, shuffled minibatches, SGD, evaluation — *check: overfits one batch to ~0 loss*
- [x] 1.5 Train 784 → 128 → 10 on MNIST — *result: 97.5% test accuracy*
- [x] Turn the checks into `pytest` tests (`part1_numpy/tests/test_gradients.py`)
- [ ] Stretch: momentum, Adam
- [x] Write `part1_numpy/README.md` with results and plot
- [x] Write the "What I learned" section

## Phase 2 – Character-level GPT (`part2_gpt/`)
- [x] 2.1 Tokenizer and data: vocab, `encode`/`decode`, train/val split, `get_batch` — *check: `decode(encode(s)) == s`*
- [x] 2.2 Bigram baseline + training loop + `generate()` — *result: val loss 2.50 (init 4.73: nn.Embedding starts with large weights)*
- [x] 2.3 Single causal self-attention head — *check: changing future tokens doesn't change earlier outputs*
- [x] 2.4 Multi-head attention + feed-forward MLP
- [x] 2.5 Transformer block (pre-LayerNorm, residuals), stacked, with positional embeddings
- [x] 2.6 AdamW, warmup + cosine LR, gradient clipping, dropout, train/val loss estimate, checkpoints
- [x] 2.7 Sampling with temperature and top-k
- [x] `pytest` tests: round trip, shapes, loss ≈ ln(65) at init, causality, generation past block_size
- [x] 2.8 Train the presets — *results: val loss 1.59 (cpu, 3 min), 1.51 (gpu, 14 min)*
- [ ] Optional: train the `large` preset (~1 hour)
- [x] Write `part2_gpt/README.md` with loss curve and samples
- [x] Write the "What I learned" section

## Phase 3 – Experiments (optional)
- [x] Remove √d_k scaling — *result: 1.600 vs 1.590 val loss; sharper attention (max weight 0.60 vs 0.45)*
- [ ] Remove other pieces: positional embeddings, residuals
- [x] Visualize attention maps — *previous-char and two-back heads in layer 0, long-range heads in layer 1*
- [ ] Compare char-level vs BPE (`tiktoken`)
- [ ] Train on a different corpus

## Sanity checks for any model
1. Loss at init ≈ ln(number of classes / vocab size)
2. Can overfit a single batch to ~0 loss
3. Watch the train/val gap for overfitting
