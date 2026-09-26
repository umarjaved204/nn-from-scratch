"""Train a character-level language model on Tiny Shakespeare.

Usage: python part2_gpt/train.py --config cpu            (or: gpu, debug)
       python part2_gpt/train.py --config cpu --lr 1e-2  (override the learning rate)
"""

import argparse

import torch

from config import get_config
from model import BigramLanguageModel
from tokenizer import get_batch, load_data


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    """Average loss over eval_iters random batches, for both splits (less noisy than one batch)."""
    model.eval()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu", help="preset from config.py: debug, cpu or gpu")
    parser.add_argument("--lr", type=float, help="override the preset's learning rate")
    args = parser.parse_args()

    cfg = get_config(args.config)
    if args.lr is not None:
        cfg.learning_rate = args.lr
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"config: {args.config} | device: {device} | lr: {cfg.learning_rate}")

    tokenizer, train_data, val_data = load_data()
    cfg.vocab_size = tokenizer.vocab_size

    model = BigramLanguageModel(cfg.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    for step in range(cfg.max_iters + 1):
        # Report average train/val loss every eval_interval steps
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            print(f"step {step:5d}  train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")

        # One learning step on a random batch
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)                  # forward: get the loss
        optimizer.zero_grad(set_to_none=True)  # clear the previous step's gradients
        loss.backward()                        # backprop: compute every parameter's gradient
        optimizer.step()                       # update the weights

    # Generate text, starting from a single newline character (id 0)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- sample ---")
    print(tokenizer.decode(model.generate(context, max_new_tokens=300)[0].tolist()))


if __name__ == "__main__":
    main()
