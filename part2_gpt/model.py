"""Character-level language models: a bigram baseline and a GPT built on causal self-attention."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """Shared by every model here: the loss calculation and text generation."""

    block_size = None  # longest context the model can read; None means no limit

    @staticmethod
    def compute_loss(logits, targets):
        # cross_entropy expects one row per prediction, so flatten batch and time together
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Extend idx (B, T) by sampling one new character at a time.

        temperature < 1 makes the text safer and more repetitive, > 1 more random.
        top_k, if set, only allows the k most likely characters at each step.
        """
        for _ in range(max_new_tokens):
            # Models with positional embeddings can only read the last block_size characters
            context = idx if self.block_size is None else idx[:, -self.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature               # last position only: (B, C)
            if top_k is not None:
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = F.softmax(logits, dim=-1)                     # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)     # sample one id per row: (B, 1)
            idx = torch.cat([idx, next_id], dim=1)                # append it: (B, T + 1)
        return idx


class BigramLanguageModel(LanguageModel):
    """Predicts the next character from the current character only.

    The whole model is one (vocab_size, vocab_size) table: row i holds the scores
    for which character comes after character i.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """idx and targets are (B, T) tensors of character ids. Returns (logits, loss)."""
        logits = self.table(idx)  # each id's row of scores: (B, T, vocab_size)
        loss = None if targets is None else self.compute_loss(logits, targets)
        return logits, loss


class Head(nn.Module):
    """One head of causal self-attention.

    Every position emits a query ("what am I looking for?"), a key ("what do I contain?")
    and a value ("what will I pass on?"). Each position then takes a weighted average of
    the values of itself and all earlier positions, weighted by how well its query matches their keys.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular matrix of ones: position t may look at positions 0..t, never later ones
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # How well each query matches each key, scaled so softmax doesn't saturate: (B, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        # Block the future: a position may not attend to anything after it
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        # Each row becomes attention weights that add up to 1
        weights = F.softmax(scores, dim=-1)

        self.weights = weights  # kept so the attention pattern can be inspected or plotted
        weights = self.dropout(weights)
        return weights @ v  # weighted average of the values: (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads side by side, each free to focus on something different."""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head  # heads split the embedding so the total size is unchanged
        self.heads = nn.ModuleList(
            [Head(cfg.n_embd, head_size, cfg.block_size, cfg.dropout) for _ in range(cfg.n_head)]
        )
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # mixes the heads' outputs back together
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A small MLP applied to each position on its own: attention gathers, this layer thinks."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),  # a smooth version of ReLU
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention then feed-forward, each with LayerNorm and a residual."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x):
        # "x +" is the residual connection: each sub-layer adds a correction instead of
        # replacing x, which gives gradients a direct path back through deep stacks
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(LanguageModel):
    """Token + position embeddings -> N transformer blocks -> scores for the next character."""

    def __init__(self, cfg):
        super().__init__()
        self.block_size = cfg.block_size
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # what each character is
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)  # where it sits in the window
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)                                # final normalization
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)                # vectors -> next-char scores
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Small random weights so the untrained model guesses evenly: loss starts at ln(vocab_size)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """idx and targets are (B, T) tensors of character ids. Returns (logits, loss)."""
        B, T = idx.shape
        tok = self.token_embedding(idx)                                    # (B, T, n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = self.dropout(tok + pos)     # each vector now knows both the character and its position
        x = self.blocks(x)              # N rounds of: gather from earlier characters, then process
        logits = self.lm_head(self.ln_f(x))                                # (B, T, vocab_size)
        loss = None if targets is None else self.compute_loss(logits, targets)
        return logits, loss


if __name__ == "__main__":
    torch.manual_seed(0)
    head = Head(n_embd=16, head_size=8, block_size=8)

    # Causality check: changing position 5 must not change the outputs at positions 0-4
    x = torch.randn(1, 8, 16)
    out1 = head(x)
    x2 = x.clone()
    x2[0, 5] = torch.randn(16)
    out2 = head(x2)
    print("earlier positions unchanged:", torch.allclose(out1[0, :5], out2[0, :5]))    # expect True
    print("changed position differs:  ", not torch.allclose(out1[0, 5], out2[0, 5]))  # expect True

    # Attention weights for a 4-character input: zeros above the diagonal, each row sums to 1
    head(torch.randn(1, 4, 16))
    print(head.weights[0].detach().round(decimals=2))
