"""
Train nrjNET and save a picture of it learning to assets/training.png.

Run from the project root (needs matplotlib):
    python examples/plot_training.py
"""

import os
import random

import matplotlib.pyplot as plt

from nrjnet import MLP

random.seed(42)

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
model = MLP(3, [4, 4, 1])

before = [model(x).data for x in xs]
losses = []
for step in range(100):
    loss = sum((model(x) - y) ** 2 for x, y in zip(xs, ys))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad
    losses.append(loss.data)
after = [model(x).data for x in xs]

# ---- plot ----
PINK, PURPLE, MINT, INK, GRID = "#ff7eb6", "#8b5cf6", "#34d399", "#3b3355", "#ece8f5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")

ax1.plot(losses, color=PURPLE, linewidth=2.5)
ax1.fill_between(range(len(losses)), losses, color=PURPLE, alpha=0.12)
ax1.set_title("Loss goes down", fontweight="bold", pad=12)
ax1.set_xlabel("training step")
ax1.set_ylabel("loss")
ax1.grid(color=GRID)
ax1.annotate(f"{losses[0]:.2f}", (0, losses[0]), xytext=(10, -4), textcoords="offset points", color=PURPLE)
ax1.annotate(f"{losses[-1]:.3f}", (len(losses) - 1, losses[-1]), xytext=(-30, 10),
             textcoords="offset points", color=PURPLE)

idx = range(len(xs))
w = 0.26
ax2.bar([i - w for i in idx], ys, w, color=INK, alpha=0.18, label="target")
ax2.bar(list(idx), before, w, color=PINK, label="before training")
ax2.bar([i + w for i in idx], after, w, color=MINT, label="after training")
ax2.axhline(0, color=INK, linewidth=0.8, alpha=0.4)
ax2.set_xticks(list(idx), [f"example {i + 1}" for i in idx])
ax2.set_ylim(-1.25, 1.25)
ax2.set_title("Predictions vs targets", fontweight="bold", pad=12)
ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
ax2.grid(axis="y", color=GRID)

fig.tight_layout(pad=2)
os.makedirs("assets", exist_ok=True)
fig.savefig("assets/training.png", dpi=160)
print(f"saved assets/training.png  (loss {losses[0]:.3f} -> {losses[-1]:.4f})")
