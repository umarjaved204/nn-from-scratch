"""Train a 784 -> 128 -> 10 MLP on MNIST using only NumPy."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files; no window needed

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np
from matplotlib.ticker import PercentFormatter

from layers import Linear, ReLU, SoftmaxCrossEntropy
from mnist import load_mnist

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1
SEED = 0
PLOT_PATH = Path(__file__).resolve().parents[1] / "assets" / "mnist_training.png"


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

    def accuracy(self, X, y):
        """Fraction of rows in X whose predicted class matches the label in y."""
        predictions = self.forward(X).argmax(axis=1)  # column with the highest score per row
        return (predictions == y).mean()


def overfit_one_batch(X, y, steps=200):
    """Sanity check: a working model should memorize a tiny batch (loss -> ~0)."""
    model = MLP(784, 128, 10)
    x_batch, y_batch = X[:32], y[:32]
    for step in range(steps + 1):
        loss = model.train_step(x_batch, y_batch, LEARNING_RATE)
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss:.4f}")


def train(model, X_train, y_train, X_test, y_test):
    """Train for EPOCHS passes over the data. Returns per-epoch history for plotting."""
    history = {"loss": [], "train_acc": [], "test_acc": []}
    n = len(X_train)

    for epoch in range(1, EPOCHS + 1):
        order = np.random.permutation(n)  # new random order every epoch

        losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            loss = model.train_step(X_train[idx], y_train[idx], LEARNING_RATE)
            losses.append(loss)

        history["loss"].append(np.mean(losses))
        history["train_acc"].append(model.accuracy(X_train, y_train))
        history["test_acc"].append(model.accuracy(X_test, y_test))
        print(
            f"epoch {epoch:2d}  loss {history['loss'][-1]:.4f}  "
            f"train acc {history['train_acc'][-1]:.2%}  test acc {history['test_acc'][-1]:.2%}"
        )
    return history


def plot_history(history, path=PLOT_PATH):
    """Save loss and accuracy curves side by side (two charts, never one dual-axis chart)."""
    epochs = list(range(1, len(history["loss"]) + 1))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, history["loss"], color="#2a78d6", linewidth=2)
    ax_loss.set_title("Training loss (cross-entropy)", loc="left")

    ax_acc.plot(epochs, history["train_acc"], color="#2a78d6", linewidth=2, label="Train")
    ax_acc.plot(epochs, history["test_acc"], color="#eb6834", linewidth=2, label="Test")
    ax_acc.set_title("Accuracy", loc="left")
    ax_acc.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_acc.legend(frameon=False, loc="lower right")
    final_test = history["test_acc"][-1]
    ax_acc.annotate(
        f"{final_test:.1%}", (epochs[-1], final_test), xytext=(0, -14),
        textcoords="offset points", ha="center", color="#52514e",
    )

    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("Epoch", color="#52514e")
        ax.set_xticks(epochs)
        ax.grid(color="#e8e8e6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#52514e")

    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    np.random.seed(SEED)
    X_train, y_train, X_test, y_test = load_mnist()

    print("Sanity check: overfitting one batch of 32 images")
    overfit_one_batch(X_train, y_train)

    print("\nTraining on all 60,000 images")
    model = MLP(784, 128, 10)
    history = train(model, X_train, y_train, X_test, y_test)
    plot_history(history)
