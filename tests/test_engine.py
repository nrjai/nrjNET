"""
Tests for nrjNET.

Run from the project root:
    python -m unittest discover tests -v

Every gradient is checked two ways:
  1. against "nudge and measure" numerical derivatives (always runs)
  2. against PyTorch (runs only if torch is installed)
"""

import math
import random
import unittest

from nrjnet import Value, MLP

try:
    import torch
except ImportError:
    torch = None


def numerical_grad(f, inputs, i, h=1e-6):
    """Slope of f with respect to inputs[i], by nudging it a tiny bit both ways."""
    up = list(inputs); up[i] += h
    down = list(inputs); down[i] -= h
    return (f(*up) - f(*down)) / (2 * h)


class TestValueOps(unittest.TestCase):

    def test_forward_arithmetic(self):
        a, b = Value(2.0), Value(-3.0)
        self.assertAlmostEqual((a + b).data, -1.0)
        self.assertAlmostEqual((a * b).data, -6.0)
        self.assertAlmostEqual((a - b).data, 5.0)
        self.assertAlmostEqual((a / b).data, -2 / 3)
        self.assertAlmostEqual((a ** 3).data, 8.0)
        self.assertAlmostEqual((2 * a + 1).data, 5.0)   # number on the left
        self.assertAlmostEqual((1 - a).data, -1.0)
        self.assertAlmostEqual((6 / a).data, 3.0)

    def test_simple_expression_grads(self):
        # d = a*b + c  ->  dd/da = b, dd/db = a, dd/dc = 1
        a, b, c = Value(2.0), Value(-3.0), Value(10.0)
        d = a * b + c
        d.backward()
        self.assertEqual((a.grad, b.grad, c.grad), (-3.0, 2.0, 1.0))

    def test_value_used_twice_accumulates(self):
        # b = a + a -> db/da = 2 (gradient must ADD from both paths)
        a = Value(3.0)
        b = a + a
        b.backward()
        self.assertEqual(a.grad, 2.0)

    def test_neuron_grads_match_numerical(self):
        def f(x1, x2, w1, w2, b):
            return math.tanh(x1 * w1 + x2 * w2 + b)

        inputs = [2.0, 0.0, -3.0, 1.0, 6.8813735870195432]
        vals = [Value(v) for v in inputs]
        x1, x2, w1, w2, b = vals
        o = (x1 * w1 + x2 * w2 + b).tanh()
        o.backward()

        self.assertAlmostEqual(o.data, f(*inputs))
        for i, v in enumerate(vals):
            self.assertAlmostEqual(v.grad, numerical_grad(f, inputs, i), places=5)

    def test_complex_expression_matches_numerical(self):
        # Uses every operation at least once
        def f_value(a, b):
            c = a * b + b ** 3
            d = (c / a - 2).tanh() + (a * 0.5).exp()
            return -d * c - 1 / b

        def f_float(a, b):
            c = a * b + b ** 3
            d = math.tanh(c / a - 2) + math.exp(a * 0.5)
            return -d * c - 1 / b

        inputs = [-1.5, 2.0]
        a, b = Value(inputs[0]), Value(inputs[1])
        out = f_value(a, b)
        out.backward()

        self.assertAlmostEqual(out.data, f_float(*inputs))
        self.assertAlmostEqual(a.grad, numerical_grad(f_float, inputs, 0), places=4)
        self.assertAlmostEqual(b.grad, numerical_grad(f_float, inputs, 1), places=4)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestAgainstPyTorch(unittest.TestCase):

    def test_neuron_matches_pytorch(self):
        inputs = [2.0, 0.0, -3.0, 1.0, 6.8813735870195432]

        vals = [Value(v) for v in inputs]
        x1, x2, w1, w2, b = vals
        o = (x1 * w1 + x2 * w2 + b).tanh()
        o.backward()

        ts = [torch.tensor([v], dtype=torch.double, requires_grad=True) for v in inputs]
        tx1, tx2, tw1, tw2, tb = ts
        to = torch.tanh(tx1 * tw1 + tx2 * tw2 + tb)
        to.backward()

        self.assertAlmostEqual(o.data, to.item())
        for v, t in zip(vals, ts):
            self.assertAlmostEqual(v.grad, t.grad.item())


class TestMLP(unittest.TestCase):

    def test_parameter_count(self):
        # 4*(3+1) + 4*(4+1) + 1*(4+1) = 41
        self.assertEqual(len(MLP(3, [4, 4, 1]).parameters()), 41)

    def test_training_reduces_loss(self):
        random.seed(0)
        xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
        ys = [1.0, -1.0, -1.0, 1.0]
        model = MLP(3, [4, 4, 1])

        def loss_fn():
            return sum((model(x) - y) ** 2 for x, y in zip(xs, ys))

        first = loss_fn().data
        for _ in range(50):
            loss = loss_fn()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= 0.05 * p.grad
        self.assertLess(loss_fn().data, first * 0.1)


if __name__ == "__main__":
    unittest.main()
