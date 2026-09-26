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

- **Softmax** subtracts each row's maximum before `exp` so large scores can't overflow.
- **Weights** use He initialization (`randn * sqrt(2 / in_features)`), which suits ReLU.

### The shape trick: deriving the Linear backward pass

The three `Linear` backward formulas can be worked out without calculus, from one rule:

> **A gradient always has the same shape as the thing it's the gradient of.**

For a batch of `N` inputs going through `Linear(in, out)`:

| Array | Shape |
|---|---|
| `x` (input, saved in forward) | `(N, in)` |
| `W` | `(in, out)` |
| `b` | `(out,)` |
| `dout` (gradient arriving from the next layer) | `(N, out)` |

Each gradient is built from these pieces, arranged so that the matrix multiplication works (inner sizes match) and the result has the right shape (outer sizes):

**`dW` must be `(in, out)`**, made from `x` and `dout`. `x @ dout` is `(N, in) @ (N, out)`, where the inner sizes don't match. Transposing `x` is the only arrangement that works:

```
x.T  @  dout   →   (in, N) @ (N, out)   →   (in, out)  ✓
```

**`db` must be `(out,)`**, made from `dout` `(N, out)`. The bias was added to every row in the forward pass, so its gradient adds up `dout` over all the rows. Summing over the batch dimension removes `N`:

```
dout.sum(axis=0)   →   (N, out) → (out,)  ✓
```

**`dx` must be `(N, in)`**, made from `dout` and `W`. `W` has to be transposed:

```
dout  @  W.T   →   (N, out) @ (out, in)   →   (N, in)  ✓
```

The forward pass sent data from `in` to `out` through `W`, and the backward pass sends the gradient from `out` back to `in` through `W.T`: the same connections, in reverse.

The shapes give the formula but can't prove it's correct. That's what the gradient check below is for.

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

Before this, I knew backprop as a diagram with arrows. Writing it by hand is what made it click.

- **Most of my bugs were shape bugs.** The rule that stuck with me: a gradient always has the same shape as the thing it belongs to. That's enough to work out the Linear layer's backward pass without calculus. `dW` has to be `x.T @ dout`, because nothing else gives the right shape.
- **The right shape doesn't mean the right numbers.** My backward passes had the right shapes well before I trusted them. The gradient check (nudge each weight, measure how the loss changes) was slow, but it's the only reason I believe the gradients are correct.
- **Some of my mistakes, so I don't repeat them:** I sized the bias by the batch size (`np.zeros(32)`) instead of the number of neurons. I left `.shape` inside a matrix multiply. And I called `l1.backward` twice and got a shape error I couldn't read at first. Reading the traceback from the bottom up is what finally helped.
- **Check the boring numbers first.** A random network should have a loss of about ln(10) = 2.30, and a working one should be able to memorize 32 images. Both checks take seconds and catch problems before a 4-minute training run.
- **You can see overfitting.** After about five epochs, train accuracy kept climbing but test accuracy flattened out around 97.5%. The gap between those two lines is what overfitting looks like.
