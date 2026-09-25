"""Character-level tokenizer and batching.

TODO (Plan step 2.1):
- Build the vocabulary from the sorted set of characters in the corpus.
- Two lookup tables: char -> integer id (stoi) and integer id -> char (itos).
- encode(text) -> list[int] and decode(ids) -> str.
- Split the encoded corpus 90% train / 10% val.
- get_batch(split): pick batch_size random start positions and return
  x = data[i : i + block_size] and y = data[i + 1 : i + block_size + 1],
  both shaped (batch_size, block_size). y is x shifted left by one.

Check:
- decode(encode(s)) == s for any string s from the corpus.
"""
