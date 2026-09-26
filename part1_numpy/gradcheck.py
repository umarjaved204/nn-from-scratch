"""Numerical gradient checking.

Compares the gradients from our hand-written backward passes against gradients
measured directly by nudging each number and watching how the loss changes.
"""

import numpy as np

from layers import Linear, ReLU, SoftmaxCrossEntropy


def numerical_gradient(f, x, eps=1e-5):
    """Estimate the gradient of f() with respect to the array x.

    f takes no arguments: it re-runs the forward pass and returns the loss.
    x is changed in place (e.g. a layer's weights), so f() sees each nudge.
    """
    grad = np.zeros_like(x)
    for i in range(x.size):
        old_value = x.flat[i]

        x.flat[i] = old_value + eps  # nudge up
        f_plus = f()
        x.flat[i] = old_value - eps  # nudge down
        f_minus = f()
        x.flat[i] = old_value        # put it back

        grad.flat[i] = (f_plus - f_minus) / (2 * eps)  # centered difference: rise over run
    return grad


def relative_error(a, b):
    """Largest element-wise relative difference between two arrays."""
    return np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8))


def check_gradients(seed=0):
    """Compare analytic and numerical gradients on a tiny network. Returns {name: relative error}."""
    np.random.seed(seed)

    # A tiny network: 4 inputs -> 5 hidden -> 3 classes, batch of 2
    x = np.random.randn(2, 4)
    y = np.array([0, 2])
    l1, relu, l2, ce = Linear(4, 5), ReLU(), Linear(5, 3), SoftmaxCrossEntropy()

    def loss():
        h = relu.forward(l1.forward(x))
        return ce.forward(l2.forward(h), y)

    # Analytic gradients: one forward pass, then backward through every layer
    loss()
    grad = ce.backward()        # dloss/dlogits
    grad = l2.backward(grad)    # dloss/dh (after ReLU)
    grad = relu.backward(grad)  # dloss/dh (before ReLU)
    dx = l1.backward(grad)      # dloss/dx

    # Compare against numerical gradients
    checks = {
        "l1.dW": (l1.dW, numerical_gradient(loss, l1.W)),
        "l1.db": (l1.db, numerical_gradient(loss, l1.b)),
        "l2.dW": (l2.dW, numerical_gradient(loss, l2.W)),
        "l2.db": (l2.db, numerical_gradient(loss, l2.b)),
        "dx":    (dx,    numerical_gradient(loss, x)),
    }
    return {name: relative_error(analytic, numeric) for name, (analytic, numeric) in checks.items()}


if __name__ == "__main__":
    for name, err in check_gradients().items():
        print(f"{name:6s} relative error: {err:.2e}  {'OK' if err < 1e-6 else 'FAIL'}")
