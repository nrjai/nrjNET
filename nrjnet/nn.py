"""
nn.py - the neural network building blocks of nrjNET.

Built on top of engine.Value:
    Neuron -> Layer -> MLP
"""

import random

from nrjnet.engine import Value


class Module:
    """Shared base class: anything that has trainable parameters."""

    def parameters(self):
        return []

    def zero_grad(self):
        # Reset every gradient to 0 before a new backward pass.
        # Needed because Value accumulates gradients with +=.
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """One neuron: out = tanh(w1*x1 + w2*x2 + ... + b)"""

    def __init__(self, nin):
        # nin = number of inputs. One weight per input, plus one bias,
        # started at random so different neurons can learn different things.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w . x + b  (sum starts at b instead of 0), then squash with tanh
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer(Module):
    """A row of `nout` neurons, each looking at the same `nin` inputs."""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # A single-neuron layer returns the Value itself instead of a 1-item list
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer({len(self.neurons[0].w)} -> {len(self.neurons)})"


class MLP(Module):
    """Multi-Layer Perceptron: layers connected one after another.

    Example: MLP(3, [4, 4, 1]) builds layers 3->4, 4->4, 4->1.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        # Feed forward: output of each layer becomes input of the next
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return "MLP[" + ", ".join(str(layer) for layer in self.layers) + "]"
