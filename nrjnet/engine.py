"""
engine.py - the autograd engine of nrjNET.

`Value` wraps a single number and remembers how it was computed. Calling
`.backward()` on a result fills in `.grad` on every Value that led to it,
using the chain rule (backpropagation).

This file knows nothing about neurons or networks. It is just maths with gradients.
"""

import math


class Value:
    """A single number that remembers how it was computed, so it can do backprop."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data                  # the actual number, e.g. 2.0
        self.grad = 0.0                   # d(output)/d(this). 0 until backward() runs
        self._backward = lambda: None     # pushes my grad to my children (set by each operation)
        self._prev = set(_children)       # the Values that were combined to make me
        self._op = _op                    # the operation that made me ('+', '*', ...) - for drawing
        self.label = label                # an optional name - for drawing

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---------------- Basic operations ----------------
    # Each operation does TWO things:
    #   1. FORWARD : compute the result number
    #   2. BACKWARD: define how the gradient flows back to the inputs
    #                (local derivative x out.grad  ->  the chain rule)
    # Gradients use += because a Value used twice receives gradient from both paths.

    def __add__(self, other):                                   # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1 -> gradient passes through unchanged
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):                                   # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a -> each input gets "the other one" x out.grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):                                   # self ** number
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # ---------------- Activation functions ----------------

    def tanh(self):
        # Squashes any number into (-1, 1). Formula: (e^(2x) - 1) / (e^(2x) + 1).
        # math.tanh is used because the formula overflows for large x.
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # ---------------- Convenience operations ----------------
    # Built from the operations above, so they get backprop for free.

    def __neg__(self):              return self * -1            # -self
    def __sub__(self, other):       return self + (-other)      # self - other
    def __truediv__(self, other):   return self * other ** -1   # self / other

    # "r" versions handle a plain number on the LEFT, e.g. 2 * Value
    def __radd__(self, other):      return self + other         # other + self
    def __rmul__(self, other):      return self * other         # other * self
    def __rsub__(self, other):      return other + (-self)      # other - self
    def __rtruediv__(self, other):  return other * self ** -1   # other / self

    # ---------------- Backpropagation ----------------

    def backward(self):
        """Compute the gradient of THIS value with respect to every Value that led to it."""

        # 1. Topological order: every node comes AFTER the nodes it was made from.
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2. d(self)/d(self) = 1 - the gradient starts here.
        self.grad = 1.0

        # 3. Walk from the output back to the inputs, applying the chain rule at each node.
        for node in reversed(topo):
            node._backward()
