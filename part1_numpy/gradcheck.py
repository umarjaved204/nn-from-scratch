"""Numerical gradient checking.

TODO (Plan step 1.3):
- numerical_gradient(f, x, eps=1e-5): for each element of x, nudge it by +eps and -eps
  and estimate the derivative with the centered difference (f(x+eps) - f(x-eps)) / (2 * eps).
  Restore the original value after each nudge.
- relative_error(a, b): |a - b| / max(|a| + |b|, tiny) so small gradients are compared fairly.

Check:
- Use float64 inputs. The relative error between your analytic and numerical gradients
  should be < 1e-6 for every layer.
"""

import numpy as np  # noqa: F401
