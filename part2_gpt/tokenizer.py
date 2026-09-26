"""Character-level tokenizer and batching."""

from pathlib import Path

import torch

DATA_PATH = Path(__file__).parent / "data" / "input.txt"


class CharTokenizer:
    """Maps each distinct character in a text to an integer id, and back."""

    def __init__(self, text):
        # Every distinct character, sorted so the ids are the same on every run
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        # Lookup tables in both directions
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # character -> id, e.g. {"a": 39}
        self.itos = dict(enumerate(self.chars))                 # id -> character, e.g. {39: "a"}

    def encode(self, text):
        """Turn a string into a list of integer ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """Turn a list of integer ids back into a string."""
        return "".join(self.itos[i] for i in ids)


def load_data(path=DATA_PATH, val_fraction=0.1):
    """Read the corpus, build the tokenizer, and split the encoded text into train/val tensors."""
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # First 90% for training, last 10% held out for validation
    n = int(len(data) * (1 - val_fraction))
    train_data = data[:n]
    val_data = data[n:]
    return tokenizer, train_data, val_data


def get_batch(data, block_size, batch_size, device="cpu"):
    """Sample random windows of text. Returns inputs x and targets y, each (batch_size, block_size).

    y is x shifted one position to the left: at every position, the target is the next character.
    """
    starts = torch.randint(len(data) - block_size, (batch_size,))  # random window start positions
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])  # same windows, one step later
    return x.to(device), y.to(device)


if __name__ == "__main__":
    torch.manual_seed(1337)
    tokenizer, train_data, val_data = load_data()

    print("vocab size:", tokenizer.vocab_size)                    # expect 65
    print("characters:", repr("".join(tokenizer.chars)))

    s = "Hello, world!"
    print(tokenizer.encode(s))
    print("round trip ok:", tokenizer.decode(tokenizer.encode(s)) == s)   # expect True

    print("train / val tokens:", len(train_data), len(val_data))  # expect 1003854 111540

    x, y = get_batch(train_data, block_size=8, batch_size=2)
    print(x)
    print(y)
    print(repr(tokenizer.decode(x[0].tolist())), "->", repr(tokenizer.decode(y[0].tolist())))
