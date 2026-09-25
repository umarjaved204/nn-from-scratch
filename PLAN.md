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
- [ ] Turn the checks into `pytest` tests (`part1_numpy/tests/test_gradients.py`)
- [ ] Stretch: momentum, Adam
- [x] Write `part1_numpy/README.md` with results and plot
- [ ] Write the "What I learned" section in your own words

## Phase 2 – Character-level GPT (`part2_gpt/`)
- [ ] 2.1 Tokenizer and data: vocab, `encode`/`decode`, train/val split, `get_batch` — *check: `decode(encode(s)) == s`*
- [ ] 2.2 Bigram baseline + training loop + `generate()` — *check: loss ≈ ln(65) ≈ 4.17 at init, ≈ 2.5 trained*
- [ ] 2.3 Single causal self-attention head — *check: changing future tokens doesn't change earlier outputs*
- [ ] 2.4 Multi-head attention + feed-forward MLP
- [ ] 2.5 Transformer block (pre-LayerNorm, residuals), stacked, with positional embeddings
- [ ] 2.6 AdamW, warmup + cosine LR, gradient clipping, dropout, train/val loss estimate, checkpoints
- [ ] 2.7 Sampling with temperature and top-k
- [ ] 2.8 Scale up — *target: val loss < 2.0 (CPU) / ≈ 1.5 (GPU)*
- [ ] Write `part2_gpt/README.md` with loss curve and samples

## Phase 3 – Experiments (optional)
- [ ] Remove pieces one at a time: positional embeddings, √d_k scaling, residuals
- [ ] Visualize attention maps
- [ ] Compare char-level vs BPE (`tiktoken`)
- [ ] Train on a different corpus

## Sanity checks for any model
1. Loss at init ≈ ln(number of classes / vocab size)
2. Can overfit a single batch to ~0 loss
3. Watch the train/val gap for overfitting
