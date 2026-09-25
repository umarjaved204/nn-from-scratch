# Part 2 – Character-level GPT in PyTorch

A decoder-only transformer trained on Tiny Shakespeare to generate text one character at a time.

## Results

_TBD: train/val loss curves, generated samples at several checkpoints._

## Architecture

_TBD: tokenizer, token + positional embeddings, causal multi-head self-attention, transformer blocks, language-model head._

## Run

```bash
python part2_gpt/scripts/download_data.py
python part2_gpt/train.py --config cpu      # or: --config gpu
python part2_gpt/sample.py
pytest part2_gpt/tests
```
