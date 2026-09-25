"""Train the character-level GPT.

Usage: python part2_gpt/train.py --config cpu   (or: gpu, debug)

TODO (Plan steps 2.2 and 2.6):
- Parse --config and load it with config.get_config(name).
- Set seeds (torch.manual_seed) so runs are reproducible.
- Pick the device: "cuda" if torch.cuda.is_available() else "cpu".
- Build the tokenizer, set config.vocab_size, create the model and move it to the device.
- Optimizer: AdamW with weight decay.
- Learning rate: linear warmup for warmup_iters, then cosine decay to min_lr.
- Training loop: get_batch -> forward -> loss.backward() -> clip gradients -> optimizer.step().
- Every eval_interval steps: estimate the mean train/val loss over eval_iters batches
  inside torch.no_grad() with model.eval(), then switch back to model.train().
- Save checkpoints (model state, config, vocab) to checkpoints/ (gitignored).
- Log losses so you can plot a loss curve into ../assets/ for the README.

Checks:
- Overfit a single batch to ~0 loss before running full training.
- Target: val loss < 2.0 (cpu preset) or ≈ 1.5 (gpu preset).
"""
