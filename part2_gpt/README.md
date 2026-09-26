# Part 2 – Character-level GPT in PyTorch

A decoder-only transformer trained on [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) (1.1M characters) to generate text one character at a time. Built up step by step, starting from a bigram baseline.

## Results

| Model | Parameters | Context | Validation loss | Training time* |
|---|---|---|---|---|
| Random guessing | – | – | 4.17 (= ln 65) | – |
| Bigram baseline | 4,225 | 1 character | 2.50 | 15 s |
| GPT, `cpu` preset | 0.8M | 64 characters | 1.59 | 3 min |
| GPT, `gpu` preset | 3.3M | 256 characters | **1.51** | 14 min |

<sub>*On a laptop GeForce MX550 (2 GB).</sub>

![Train and validation loss for the gpu preset](../assets/loss_gpt_gpu.png)

### Sample

From the `gpu` preset at the end of training (temperature 1.0):

```
On is truesty time, to the suit of steem;
Proved the couldey that standery for the fleere but of
We will posses, may constr, willieve undertain brunce
Can thou cert overy, men, on denying and powely:
Would he sought in y thing conscqual to golden-roor.

BUCKINGHARDINE:
Gester, I did-not gine own,
```

The bigram model, which only sees one character, produces word-shaped gibberish (`Wanthar u qur, vet?`). With attention, the model spells most words correctly, invents plausible speaker names, and keeps the rhythm of verse, though the text doesn't mean anything yet.

## Architecture

```
character ids
  → token embedding + position embedding
  → N × Block:
        x = x + MultiHeadAttention(LayerNorm(x))   # gather information from earlier characters
        x = x + FeedForward(LayerNorm(x))          # process it at each position
  → LayerNorm → Linear → scores for the next character
```

| Piece | What it does |
|---|---|
| **Tokenizer** ([`tokenizer.py`](tokenizer.py)) | Maps each of the 65 distinct characters to an id. Targets are the inputs shifted one character left. |
| **Causal self-attention** ([`model.py`](model.py) `Head`) | `softmax(Q·Kᵀ / √d + mask) · V`. Each position scores every earlier position by how well its query matches their key, then takes a weighted average of their values. The mask stops it seeing the future. |
| **Scaling by √d** | Keeps dot products small so softmax doesn't saturate into all-or-nothing weights with vanishing gradients. |
| **Multi-head attention** | Several smaller heads side by side, each free to learn a different pattern. |
| **Position embeddings** | Attention is order-blind (a weighted average), so each position gets a learned vector added to it. |
| **Feed-forward** | `Linear → GELU → Linear` applied to each position independently. |
| **Residual connections + pre-LayerNorm** | Each sub-layer adds a correction to `x` rather than replacing it, giving gradients a direct path through deep stacks. |

## Training

- **AdamW**, with weight decay on weight matrices only (not biases or LayerNorm)
- **Learning rate**: linear warmup over 100 steps, then cosine decay from 1e-3 to 1e-4
- **Gradient clipping** at norm 1.0
- **Dropout** 0.2 (`gpu` and `large` presets)
- **Checkpointing**: the model with the best validation loss is saved

Sanity checks, all in [`tests/test_model.py`](tests/test_model.py): the untrained model's loss is ≈ ln(65); changing a character never changes predictions for earlier positions; generation works past the context length.

## Presets

Defined in [`config.py`](config.py):

| Preset | Layers | Heads | Embedding | Context | Batch | Steps |
|---|---|---|---|---|---|---|
| `debug` | 1 | 2 | 16 | 8 | 4 | 100 |
| `cpu` | 4 | 4 | 128 | 64 | 32 | 3,000 |
| `gpu` | 4 | 4 | 256 | 256 | 16 | 5,000 |
| `large` | 6 | 6 | 384 | 256 | 16 | 5,000 |

## Run

```bash
python part2_gpt/scripts/download_data.py                  # Tiny Shakespeare, ~1 MB
python part2_gpt/train.py --config gpu                     # train; saves checkpoint + loss plot
python part2_gpt/sample.py --config gpu --prompt "ROMEO:" --temperature 0.8 --top-k 20
python part2_gpt/train.py --config cpu --model bigram --lr 1e-2   # bigram baseline
pytest part2_gpt/tests
```

## What I learned

<!-- Write this in your own words: what clicked about attention, what surprised you, bugs worth remembering. -->
_TBD_
