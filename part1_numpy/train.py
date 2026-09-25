"""Train an MLP on MNIST using only NumPy.

TODO (Plan steps 1.4 and 1.5):
- Load data with `from mnist import load_mnist`.
- Build a 784 -> 128 -> 10 network from the layers in layers.py.
- Loop over epochs: shuffle, split into minibatches, forward, backward, SGD update.
- After each epoch, report train loss and test accuracy.
- Save a loss/accuracy plot to ../assets/ for the README.

Checks:
- Before full training, confirm the model can overfit a single small batch to ~0 loss.
- Target: > 97% test accuracy.
"""
