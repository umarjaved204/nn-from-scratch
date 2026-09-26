"""Draw what each attention head looks at, for one line of text.

Usage: python part2_gpt/scripts/plot_attention.py --config gpu                    (every head, as a grid)
       python part2_gpt/scripts/plot_attention.py --config gpu --layer 0 --head 2  (one head, large)
       python part2_gpt/scripts/plot_attention.py --config gpu --text "ROMEO:\\nBut soft"

Row = the character doing the looking; column = an earlier character it can look at.
Darker = more attention. Everything above the diagonal is blank: the future is masked out.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files; no window needed

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so we can import from part2_gpt/
from sample import CHECKPOINT_DIR, load_model  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
DEFAULT_TEXT = "First Citizen:\nBefore we proceed"  # the opening line of the training text
# Sequential single-hue scale: near-white (no attention) to dark blue (all attention)
CMAP = LinearSegmentedColormap.from_list("attention", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])


def show(ch):
    """Make spaces and newlines visible in tick labels."""
    return {" ": "␣", "\n": "⏎"}.get(ch, ch)


def draw_head(ax, weights, labels, show_x, show_y, fontsize):
    image = ax.imshow(weights, cmap=CMAP, vmin=0, vmax=1)
    ticks = range(len(labels))
    ax.set_xticks(ticks, labels if show_x else [], fontsize=fontsize, fontfamily="monospace")
    ax.set_yticks(ticks, labels if show_y else [], fontsize=fontsize, fontfamily="monospace")
    ax.tick_params(length=0, colors="#52514e")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gpu", help="which trained preset to load")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to analyse (type \\n for a newline)")
    parser.add_argument("--layer", type=int, help="plot a single head: its layer (from 0)")
    parser.add_argument("--head", type=int, help="plot a single head: its index within the layer")
    parser.add_argument("--out", help="file name inside assets/ (default depends on the options)")
    args = parser.parse_args()

    path = CHECKPOINT_DIR / f"gpt_{args.config}.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}\nTrain one first: python part2_gpt/train.py --config {args.config}")
    model, tokenizer, _ = load_model(path, "cpu")

    text = args.text.replace("\\n", "\n")[: model.block_size]
    unknown = sorted(set(text) - set(tokenizer.chars))
    if unknown:
        raise SystemExit(f"The text contains characters the model has never seen: {unknown}")

    # One forward pass; every Head keeps its attention weights in head.weights
    with torch.no_grad():
        model(torch.tensor([tokenizer.encode(text)]))
    weights = [[head.weights[0].numpy() for head in block.attn.heads] for block in model.blocks]
    labels = [show(ch) for ch in text]

    if args.layer is not None and args.head is not None:
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        image = draw_head(ax, weights[args.layer][args.head], labels, True, True, fontsize=10)
        ax.set_title(f"Layer {args.layer}, head {args.head}", loc="left")
        ax.set_xlabel("attends to (earlier character)", color="#52514e")
        ax.set_ylabel("current character", color="#52514e")
        out_name = args.out or f"attention_gpt_{args.config}_L{args.layer}H{args.head}.png"
    else:
        n_layer, n_head = len(weights), len(weights[0])
        fig, axes = plt.subplots(n_layer, n_head, figsize=(3.2 * n_head, 3.2 * n_layer),
                                 squeeze=False, layout="constrained")
        for layer in range(n_layer):
            for h in range(n_head):
                ax = axes[layer][h]
                image = draw_head(ax, weights[layer][h], labels,
                                  show_x=layer == n_layer - 1, show_y=h == 0, fontsize=6)
                ax.set_title(f"layer {layer}, head {h}", fontsize=9, color="#52514e")
        fig.suptitle("Attention weights for every head (row = current character, column = earlier character)")
        out_name = args.out or f"attention_gpt_{args.config}.png"

    fig.colorbar(image, ax=fig.axes, shrink=0.6, label="attention weight")
    out_path = ASSETS_DIR / out_name
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
