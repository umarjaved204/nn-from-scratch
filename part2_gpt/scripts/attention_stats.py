"""Measure what trained models' attention heads focus on, using real validation text.

Usage: python part2_gpt/scripts/attention_stats.py gpt_cpu gpt_cpu_noscale   (how sharp is attention?)
       python part2_gpt/scripts/attention_stats.py gpt_gpu --per-head         (what does each head look at?)

Sharpness: for every attention row, the largest weight and the entropy (how spread out the
weights are). Sharper attention = larger max weight, lower entropy.
Per head: share of attention on the previous character, on the character two back, and on
spaces/newlines, plus how far back the head looks on average.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so we can import from part2_gpt/
from sample import CHECKPOINT_DIR, load_model  # noqa: E402
from tokenizer import load_data  # noqa: E402

SKIP_ROWS = 10  # the first positions have very few earlier characters to choose from


def run_model(run, val_data, n_windows):
    """Load a checkpoint and run it on random validation windows. Returns (model, tokenizer, idx)."""
    path = CHECKPOINT_DIR / f"{run}.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}")
    model, tokenizer, _ = load_model(path, "cpu")
    torch.manual_seed(0)  # the same windows for every run, so comparisons are fair
    T = model.block_size
    starts = torch.randint(len(val_data) - T, (n_windows,))
    idx = torch.stack([val_data[s : s + T] for s in starts])
    with torch.no_grad():
        model(idx)
    return model, tokenizer, idx


def sharpness(model):
    max_weights, entropies = [], []
    for block in model.blocks:
        for head in block.attn.heads:
            w = head.weights[:, SKIP_ROWS:, :]
            max_weights.append(w.max(dim=-1).values.mean().item())
            entropies.append(-(w * w.clamp_min(1e-12).log()).sum(dim=-1).mean().item())
    return sum(max_weights) / len(max_weights), sum(entropies) / len(entropies)


def per_head(model, tokenizer, idx):
    T = idx.shape[1]
    pos = torch.arange(T)
    separators = torch.tensor([tokenizer.stoi[" "], tokenizer.stoi["\n"]])
    is_separator = torch.isin(idx, separators).float()        # (B, T): 1 where the text has a space/newline
    distance = (pos[:, None] - pos[None, :]).clamp(min=0).float()  # how far back each column is

    print(f"separators (spaces/newlines) are {is_separator.mean().item():.0%} of the text\n")
    print("head    prev  2 back  separators  avg distance")
    for L, block in enumerate(model.blocks):
        for H, head in enumerate(block.attn.heads):
            w = head.weights                                   # (B, T, T)
            prev = w[:, pos[1:], pos[1:] - 1].mean().item()
            two_back = w[:, pos[2:], pos[2:] - 2].mean().item()
            on_separators = (w * is_separator[:, None, :]).sum(-1).mean().item()
            avg_distance = (w * distance).sum(-1).mean().item()
            print(f"L{L}H{H}    {prev:4.0%}  {two_back:6.0%}  {on_separators:10.0%}  {avg_distance:8.1f} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="checkpoint names, e.g. gpt_cpu gpt_cpu_noscale")
    parser.add_argument("--per-head", action="store_true", help="break the numbers down by head")
    parser.add_argument("--windows", type=int, default=16, help="number of random validation windows")
    args = parser.parse_args()

    _, _, val_data = load_data()
    for run in args.runs:
        model, tokenizer, idx = run_model(run, val_data, args.windows)
        if args.per_head:
            print(f"=== {run}")
            per_head(model, tokenizer, idx)
        else:
            max_weight, entropy = sharpness(model)
            print(f"{run:24s} average max weight {max_weight:.2f}   average entropy {entropy:.2f} nats")


if __name__ == "__main__":
    main()
