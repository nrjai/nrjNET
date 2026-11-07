"""
Train nrjNET on a tiny 4-example dataset.

Run from the project root:
    python examples/train.py
"""

import random

from nrjnet import MLP

random.seed(42)  # same starting weights every run

# Dataset: 4 examples with 3 inputs each, target +1 or -1
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])
print(model, "|", len(model.parameters()), "parameters\n")

learning_rate = 0.05
steps = 100

for step in range(steps):
    # 1. Forward pass: predictions and loss (sum of squared errors)
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # 2. Zero the gradients
    model.zero_grad()

    # 3. Backward pass
    loss.backward()

    # 4. Update: step every parameter against its gradient (downhill)
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == steps - 1:
        print(f"step {step:3d}   loss {loss.data:.6f}")

print("\nResults:")
for x, target in zip(xs, ys):
    print(f"  input {x!s:<18} target {target:+.1f}   prediction {model(x).data:+.4f}")
