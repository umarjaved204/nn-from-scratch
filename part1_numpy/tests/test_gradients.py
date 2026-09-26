"""Tests for the NumPy layers."""

import math

import numpy as np
import pytest

from gradcheck import check_gradients
from layers import Linear, ReLU, SoftmaxCrossEntropy


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backprop_matches_numerical_gradients(seed):
    for name, err in check_gradients(seed).items():
        assert err < 1e-6, f"{name}: relative error {err:.2e}"


def test_initial_loss_is_ln_num_classes():
    # Equal scores give every class probability 1/10, so the loss is -ln(1/10) = ln(10)
    loss = SoftmaxCrossEntropy().forward(np.zeros((4, 10)), np.array([0, 3, 7, 9]))
    assert loss == pytest.approx(math.log(10))


def test_softmax_is_stable_for_huge_scores():
    loss_fn = SoftmaxCrossEntropy()
    loss = loss_fn.forward(np.array([[1000.0, 0.0, 0.0]]), np.array([0]))
    assert np.isfinite(loss)
    assert np.allclose(loss_fn.probs.sum(axis=1), 1.0)


def test_linear_gradient_shapes_match_parameters():
    layer = Linear(4, 3)
    layer.forward(np.random.randn(5, 4))
    dx = layer.backward(np.ones((5, 3)))
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape
    assert dx.shape == (5, 4)


def test_relu_blocks_gradient_for_non_positive_inputs():
    relu = ReLU()
    relu.forward(np.array([-2.0, 0.0, 3.0]))
    assert np.array_equal(relu.backward(np.array([5.0, 5.0, 5.0])), [0.0, 0.0, 5.0])
