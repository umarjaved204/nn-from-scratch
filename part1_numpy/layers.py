"""Neural network layers with explicit forward and backward passes.

TODO (Plan steps 1.1 and 1.2):
- Linear: y = x @ W + b. Cache the input in forward; in backward compute dW, db and dx.
  Use He initialization for W: randn(in, out) * sqrt(2 / in).
- ReLU: y = max(0, x). Backward passes the gradient only where x > 0.
- SoftmaxCrossEntropy: combined softmax + cross-entropy loss.
  Forward: subtract the row max before exp (log-sum-exp trick) so it never overflows.
  Backward: (probs - one_hot) / batch_size.

Checks:
- With random weights on MNIST, the loss should be close to ln(10) ≈ 2.30.
- Every backward pass must pass the gradient check in gradcheck.py.
"""

import numpy as np  # noqa: F401
