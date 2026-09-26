"""Train a character-level language model on Tiny Shakespeare.

Usage: python part2_gpt/train.py --config cpu                            (or: debug, gpu, large)
       python part2_gpt/train.py --config cpu --model bigram --lr 1e-2   (bigram baseline)

Saves the best checkpoint to part2_gpt/checkpoints/ and a loss plot to assets/.
"""

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files; no window needed

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import torch  # noqa: E402

from config import get_config  # noqa: E402
from model import GPT, BigramLanguageModel  # noqa: E402
from tokenizer import get_batch, load_data  # noqa: E402

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    """Average loss over eval_iters random batches, for both splits (less noisy than one batch)."""
    model.eval()  # turns dropout off
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step, cfg):
    """Learning rate schedule: linear warmup, then cosine decay down to min_lr."""
    # Warmup: start small while the weights are random and the gradients are unreliable
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.warmup_iters
    # Cosine decay: take smaller and smaller steps as training settles into a minimum
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))  # goes smoothly from 1 to 0
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


def make_optimizer(model, cfg):
    """AdamW, with weight decay on weight matrices only (not on biases or LayerNorm parameters)."""
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate)


def save_checkpoint(path, model, model_type, cfg, tokenizer, step, val_loss):
    """Save everything sample.py needs to rebuild the model and tokenizer."""
    path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "chars": tokenizer.chars,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def plot_losses(history, title, path):
    """Save train and validation loss curves to an image for the README."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["train"], color="#2a78d6", linewidth=2, label="Train")
    ax.plot(history["step"], history["val"], color="#eb6834", linewidth=2, label="Validation")
    ax.set_title(title, loc="left")
    ax.set_xlabel("Step", color="#52514e")
    ax.set_ylabel("Cross-entropy loss", color="#52514e")
    ax.annotate(
        f"{history['val'][-1]:.2f}", (history["step"][-1], history["val"][-1]),
        xytext=(0, 8), textcoords="offset points", ha="center", color="#52514e",
    )
    ax.legend(frameon=False)
    ax.grid(color="#e8e8e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu", help="preset from config.py: debug, cpu, gpu or large")
    parser.add_argument("--model", default="gpt", choices=["gpt", "bigram"])
    parser.add_argument("--lr", type=float, help="override the preset's learning rate")
    args = parser.parse_args()

    cfg = get_config(args.config)
    if args.lr is not None:
        cfg.learning_rate = args.lr
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, train_data, val_data = load_data()
    cfg.vocab_size = tokenizer.vocab_size

    if args.model == "gpt":
        model = GPT(cfg).to(device)
    else:
        model = BigramLanguageModel(cfg.vocab_size).to(device)
    optimizer = make_optimizer(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model} ({n_params:,} params) | config: {args.config} "
          f"| device: {device} | lr: {cfg.learning_rate}")

    run_name = f"{args.model}_{args.config}"
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"
    history = {"step": [], "train": [], "val": []}
    best_val = float("inf")
    start_time = time.time()

    for step in range(cfg.max_iters + 1):
        # Set this step's learning rate from the schedule
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Every eval_interval steps (and at the end): report losses, keep the best model so far
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"step {step:5d}  train loss {losses['train']:.4f}  val loss {losses['val']:.4f}  "
                  f"lr {lr:.1e}  {time.time() - start_time:5.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(checkpoint_path, model, args.model, cfg, tokenizer, step, best_val)

        if step == cfg.max_iters:
            break

        # One learning step on a random batch
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)                  # forward: get the loss
        optimizer.zero_grad(set_to_none=True)  # clear the previous step's gradients
        loss.backward()                        # backprop: compute every parameter's gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)  # cap rare huge gradients
        optimizer.step()                       # update the weights

    print(f"\nbest val loss {best_val:.4f} | checkpoint saved to {checkpoint_path}")
    if args.config != "debug":  # debug runs are smoke tests; keep them out of the README assets
        plot_path = ASSETS_DIR / f"loss_{run_name}.png"
        plot_losses(history, f"Loss: {args.model}, {args.config} preset", plot_path)
        print(f"loss plot saved to {plot_path}")

    # Generate text, starting from a single newline character (id 0)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- sample ---")
    print(tokenizer.decode(model.generate(context, max_new_tokens=300)[0].tolist()))


if __name__ == "__main__":
    main()
