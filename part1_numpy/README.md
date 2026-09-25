# Part 1 – MLP from Scratch in NumPy

A multi-layer perceptron with hand-derived backpropagation, trained on MNIST using only NumPy. No autograd, no deep learning framework.

## Results

| Metric | Value |
|---|---|
| Test accuracy (10,000 unseen images) | **97.5%** |
| Train accuracy | 98.8% |
| Training time | ~4 min on a laptop CPU (10 epochs) |

![Training loss and accuracy over 10 epochs](../assets/mnist_training.png)

Train and test accuracy rise together for the first five epochs. After that, train accuracy keeps climbing while test accuracy levels off: the start of overfitting.

## How it works

### Architecture

```
image (784 pixels) → Linear(784, 128) → ReLU → Linear(128, 10) → softmax → 10 digit probabilities
```

### Layers ([`layers.py`](layers.py))

Every layer has a `forward` pass and a `backward` pass. Backward receives `dout`, the gradient of the loss with respect to the layer's output, and returns the gradient with respect to its input.

| Layer | Forward | Backward |
|---|---|---|
| `Linear` | `y = x @ W + b` | `dW = x.T @ dout`<br>`db = dout.sum(axis=0)`<br>`dx = dout @ W.T` |
| `ReLU` | `y = max(0, x)` | `dx = dout * (x > 0)` |
| `SoftmaxCrossEntropy` | `loss = mean(-log(softmax(logits)[correct class]))` | `dlogits = (probs - one_hot) / batch_size` |

- **Linear backward** formulas follow from matching shapes: each gradient has the same shape as the thing it belongs to.
- **Softmax** subtracts each row's maximum before `exp` so large scores can't overflow.
- **Weights** use He initialization (`randn * sqrt(2 / in_features)`), which suits ReLU.

### Gradient check ([`gradcheck.py`](gradcheck.py))

Every backward pass is verified against a numerical estimate: nudge each parameter by ±ε and measure how the loss changes, `(f(w + ε) − f(w − ε)) / 2ε`. All layers agree with the analytic gradients to a relative error below 1e-7.

### Training ([`train.py`](train.py))

- Mini-batch stochastic gradient descent: batch size 64, learning rate 0.1, 10 epochs
- Data reshuffled every epoch
- Sanity check before training: the model must overfit a single batch of 32 images to near-zero loss

## Run

```bash
python part1_numpy/layers.py      # quick checks for each layer
python part1_numpy/gradcheck.py   # verify backprop numerically
python part1_numpy/train.py       # train and save the plot to assets/
```

MNIST (~11 MB) is downloaded automatically on the first run.

## What I learned

<!-- Write this in your own words: what clicked, what surprised you, and bugs worth remembering. -->
_TBD_
