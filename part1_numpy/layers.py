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

class Linear:
    def __init__(self, in_features, out_features):
        # He initialization: random normal values scaled by sqrt(2 / in_features)
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features) # shape (in_features, out_features)
        self.b = np.zeros(out_features)  # shape (out_features,), all zeros

    def forward(self, x):
        self.x = x  # saved for backward()
        return x @ self.W + self.b # shape (batch_size, out_features)

    def backward(self, dout):
        # dout: (batch_size, out_features)
        self.dW = self.x.T @ dout  # shape (in_features, out_features), same as W
        self.db = dout.sum(axis=0)  # shape (out_features,), same as b
        return dout @ self.W.T  # dx: shape (batch_size, in_features), same as x


class ReLU:
    def forward(self, x):
        self.x = x  # saved for backward()
        return np.maximum(0, x)  # negatives become 0, everything else unchanged

    def backward(self, dout):
        # dout has the same shape as x
        return dout * (self.x > 0)  # dout where x > 0, zero elsewhere


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        # logits: (batch_size, num_classes) raw scores
        # y:      (batch_size,) correct class for each row, e.g. [5, 0, 4]

        # 1. Overflow trick: subtract each row's max
        shifted = logits - logits.max(axis=1, keepdims=True)

        # 2. Softmax: exp, then divide each row by its sum
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)  # shape (batch_size, num_classes)

        # 3. Probability the model gave to the correct class, one per row
        correct_probs = probs[np.arange(len(y)), y]

        # 4. Loss: average of -log(correct_probs)
        loss = -np.log(correct_probs).mean()

        self.probs = probs  # saved for backward()
        self.y = y
        return loss

    def backward(self):
        n = len(self.y)
        grad = self.probs.copy()  # copy, so we don't change the saved probs
        grad[np.arange(n), self.y] -= 1  # subtract 1 at each row's correct class
        return grad / n


if __name__ == "__main__":
    layer = Linear(784, 128)
    x = np.random.randn(32, 784)
    print(layer.forward(x).shape)                     # expect (32, 128)

    relu = ReLU()
    print(relu.forward(np.array([-2.0, 0.0, 3.0])))   # expect [0. 0. 3.]

    loss_fn = SoftmaxCrossEntropy()

    # All scores equal -> every class gets 0.1 -> loss = ln(10)
    print(loss_fn.forward(np.zeros((4, 10)), np.array([0, 3, 7, 9])))   # expect ~2.302
    print(loss_fn.probs.sum(axis=1))                                    # expect [1. 1. 1. 1.]

    # Huge scores must not break (no nan / inf)
    print(loss_fn.forward(np.array([[1000.0, 0.0, 0.0]]), np.array([0])))  # expect ~0

    # ReLU backward: gradient passes only where the input was positive
    relu.forward(np.array([-2.0, 0.0, 3.0]))
    print(relu.backward(np.array([5.0, 5.0, 5.0])))   # expect [0. 0. 5.]

    # Softmax-CE backward
    loss_fn.forward(np.zeros((2, 3)), np.array([0, 2]))
    print(loss_fn.backward())
    # expect [[-0.333  0.167  0.167]
    #         [ 0.167  0.167 -0.333]]
    
    # Linear backward: each gradient must match the shape of what it belongs to
    layer = Linear(4, 3)
    layer.forward(np.random.randn(5, 4))
    dx = layer.backward(np.ones((5, 3)))
    print(layer.dW.shape, layer.db.shape, dx.shape)   # expect (4, 3) (3,) (5, 4)
    print(layer.db)                                   # expect [5. 5. 5.]


