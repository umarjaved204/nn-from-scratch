"""Train a 784 -> 128 -> 10 MLP on MNIST using only NumPy."""

import numpy as np

from layers import Linear, ReLU, SoftmaxCrossEntropy
from mnist import load_mnist

LEARNING_RATE = 0.1
SEED = 0


class MLP:
    """Linear -> ReLU -> Linear, with softmax cross-entropy loss."""

    def __init__(self, in_features, hidden, num_classes):
        self.l1 = Linear(in_features, hidden)
        self.relu = ReLU()
        self.l2 = Linear(hidden, num_classes)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        """Return logits, shape (batch_size, num_classes)."""
        h = self.l1.forward(x)
        h = self.relu.forward(h)
        return self.l2.forward(h)

    def backward(self):
        """Backpropagate from the loss through every layer."""
        grad = self.loss_fn.backward()
        grad = self.l2.backward(grad)
        grad = self.relu.backward(grad)
        self.l1.backward(grad)

    def step(self, lr):
        """Gradient descent: move each parameter a small step against its gradient."""
        for layer in (self.l1, self.l2):
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def train_step(self, x, y, lr):
        """One full learning step on a batch. Returns the loss before the update."""
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, y)
        self.backward()
        self.step(lr)
        return loss


def overfit_one_batch(X, y, steps=200):
    """Sanity check: a working model should memorize a tiny batch (loss -> ~0)."""
    model = MLP(784, 128, 10)
    x_batch, y_batch = X[:32], y[:32]
    for step in range(steps + 1):
        loss = model.train_step(x_batch, y_batch, LEARNING_RATE)
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss:.4f}")


if __name__ == "__main__":
    np.random.seed(SEED)
    X_train, y_train, X_test, y_test = load_mnist()

    print("Overfitting one batch of 32 images:")
    overfit_one_batch(X_train, y_train)
