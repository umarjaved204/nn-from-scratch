"""Tests for the NumPy layers.

TODO: once layers.py and gradcheck.py exist, add tests that:
- check each layer's backward pass against numerical gradients (relative error < 1e-6)
- check the loss at random init is close to ln(10)
- check softmax stays finite for very large logits (e.g. 1000)
"""
