"""Character-level language models: a bigram baseline, and (from step 2.3) a GPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
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

        if targets is None:
            return logits, None

        # cross_entropy expects one row per prediction, so flatten batch and time together
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Extend idx (B, T) by sampling one new character at a time."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                             # last position only: (B, C)
            probs = F.softmax(logits, dim=-1)                     # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)     # sample one id per row: (B, 1)
            idx = torch.cat([idx, next_id], dim=1)                # append it: (B, T + 1)
        return idx
