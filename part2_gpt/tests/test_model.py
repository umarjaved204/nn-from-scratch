"""Tests for the tokenizer and GPT model (use the tiny "debug" preset so they run in seconds)."""

import math

import torch

from config import get_config
from model import GPT
from tokenizer import CharTokenizer


def make_model():
    cfg = get_config("debug")
    cfg.vocab_size = 65
    torch.manual_seed(0)
    return GPT(cfg).eval(), cfg


def test_tokenizer_round_trip():
    tokenizer = CharTokenizer("hello world")
    text = "low hello"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_logits_shape():
    model, cfg = make_model()
    idx = torch.randint(cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(idx)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss is None


def test_initial_loss_is_ln_vocab_size():
    # Small initial weights mean the untrained model guesses evenly across the vocabulary
    model, cfg = make_model()
    idx = torch.randint(cfg.vocab_size, (8, cfg.block_size))
    targets = torch.randint(cfg.vocab_size, (8, cfg.block_size))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.1


def test_causality():
    # Changing the token at position 5 must not change the predictions at positions 0-4
    model, cfg = make_model()
    idx = torch.randint(cfg.vocab_size, (1, cfg.block_size))
    changed = idx.clone()
    changed[0, 5] = (idx[0, 5] + 1) % cfg.vocab_size
    before, _ = model(idx)
    after, _ = model(changed)
    assert torch.allclose(before[0, :5], after[0, :5], atol=1e-6)
    assert not torch.allclose(before[0, 5], after[0, 5])


def test_generate_runs_past_block_size():
    # generate() must crop its context to block_size, or position embeddings would run out
    model, cfg = make_model()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=3 * cfg.block_size)
    assert out.shape == (1, 1 + 3 * cfg.block_size)
