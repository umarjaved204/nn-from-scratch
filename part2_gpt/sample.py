"""Generate text from a trained checkpoint.

Usage: python part2_gpt/sample.py --config cpu
       python part2_gpt/sample.py --config gpu --prompt "ROMEO:" --temperature 0.8 --top-k 20
       python part2_gpt/sample.py --config gpu --save      (also write the samples to assets/)
"""

import argparse
from pathlib import Path

import torch

from config import GPTConfig
from model import GPT, BigramLanguageModel
from tokenizer import CharTokenizer

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def load_model(path, device):
    """Rebuild the model and tokenizer saved by train.py."""
    checkpoint = torch.load(path, map_location=device)
    cfg = GPTConfig(**checkpoint["config"])
    tokenizer = CharTokenizer("".join(checkpoint["chars"]))
    if checkpoint["model_type"] == "gpt":
        model = GPT(cfg)
    else:
        model = BigramLanguageModel(cfg.vocab_size)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cpu", help="which trained preset to load")
    parser.add_argument("--model", default="gpt", choices=["gpt", "bigram"])
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint path (overrides --config/--model)")
    parser.add_argument("--prompt", default="\n", help="text to continue from")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0, help="<1 safer, >1 more random")
    parser.add_argument("--top-k", type=int, help="only sample from the k most likely characters")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save", action="store_true", help="also write the samples to assets/")
    args = parser.parse_args()

    run_name = f"{args.model}_{args.config}"
    path = args.checkpoint or CHECKPOINT_DIR / f"{run_name}.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}\nTrain one first: python part2_gpt/train.py --config {args.config}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, checkpoint = load_model(path, device)
    print(f"loaded {path.name} (step {checkpoint['step']}, val loss {checkpoint['val_loss']:.4f})")

    unknown = sorted(set(args.prompt) - set(tokenizer.chars))
    if unknown:
        raise SystemExit(f"The prompt contains characters the model has never seen: {unknown}")
    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    samples = []
    for i in range(args.num_samples):
        out = model.generate(context, args.max_new_tokens, args.temperature, args.top_k)
        samples.append(tokenizer.decode(out[0].tolist()))
        print(f"\n--- sample {i + 1} ---\n{samples[-1]}")

    if args.save:
        out_path = ASSETS_DIR / f"sample_{run_name}.txt"
        out_path.parent.mkdir(exist_ok=True)
        header = (f"# {run_name}: val loss {checkpoint['val_loss']:.4f}, "
                  f"temperature {args.temperature}, top-k {args.top_k}\n")
        out_path.write_text(header + "\n\n---\n\n".join(samples) + "\n", encoding="utf-8")
        print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
