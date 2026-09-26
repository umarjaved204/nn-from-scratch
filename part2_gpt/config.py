"""Hyperparameter presets for the character-level GPT."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class GPTConfig:
    # Model
    block_size: int = 256  # context length: max number of tokens the model sees at once
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    vocab_size: int | None = None  # set from the tokenizer at runtime

    # Training
    batch_size: int = 32
    max_iters: int = 5000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4  # floor for the cosine schedule
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 200  # batches averaged when estimating train/val loss
    seed: int = 1337


# Timings measured on a GeForce MX550 (2 GB). Every preset runs on the GPU when one is available.
PRESETS: dict[str, GPTConfig] = {
    # Tiny model for tests and quick debugging runs
    "debug": GPTConfig(
        block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0,
        batch_size=4, max_iters=100, warmup_iters=10, eval_interval=50, eval_iters=10,
    ),
    # ~0.8M params; small enough to train on a laptop CPU. ~6 min on the MX550.
    "cpu": GPTConfig(
        block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0,
        batch_size=32, max_iters=3000, eval_interval=250, eval_iters=50,
    ),
    # ~3.3M params; ~20 min and ~0.5 GB on the MX550
    "gpu": GPTConfig(
        n_layer=4, n_head=4, n_embd=256, batch_size=16, eval_interval=500, eval_iters=50,
    ),
    # ~10.8M params, the classic nanoGPT Shakespeare size; ~1 hour and ~1.1 GB on the MX550.
    # (batch_size 32 needs slightly more than 2 GB, which slows the MX550 to a crawl)
    "large": GPTConfig(batch_size=16, eval_interval=500, eval_iters=50),
}


def get_config(name: str) -> GPTConfig:
    """Return a fresh copy of a preset so callers can modify it safely."""
    if name not in PRESETS:
        raise ValueError(f"Unknown config {name!r}. Choose from: {', '.join(PRESETS)}")
    return replace(PRESETS[name])
