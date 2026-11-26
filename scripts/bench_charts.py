#!/usr/bin/env python3
"""bench-val.png + bench-heldout.png — objavljeni brojevi, mjereno (docs/TRAINING.md).

val: sve na v03-val (217, isti set) → direktno uporedivo.
held-out: v01 (73) vs v02 (87) su RAZLIČITI setovi → set pise uz svaki bar.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- v03-val (217), isti set ---
val_labels = ["2B\nbaza", "2B v3\n(objavljen)", "4B\nbaza", "4B\n(objavljen)",
              "4B\n+sys", "8B\nbaza", "8B+sys\n(objavljen)"]
val_scores = [37.3, 46.1, 87.6, 90.3, 92.2, 88.9, 93.5]
val_colors = ["#7f7f7f", "#1f77b4", "#7f7f7f", "#2ca02c", "#98df8a",
              "#7f7f7f", "#9467bd"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(val_labels, val_scores, color=val_colors)
ax.set_title("BH-Vision serija 1 — v03-val (217 uzoraka, isti set)")
ax.set_ylabel("tačnost % (exact-match, normalizirano)")
ax.set_ylim(0, 100)
for i, v in enumerate(val_scores):
    ax.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("docs/bench-val.png", dpi=100)

# --- held-out (v01=73 vs v02=87, RAZLIČITI setovi) ---
ho_labels = ["2B baza\n(v01)", "2B v3\n(v01)", "4B\n(v01)", "4B+sys\n(v01)",
             "4B+sys\n(v02)", "8B baza\n(v02)", "8B+sys\n(v02)"]
ho_scores = [21.9, 23.3, 49.3, 54.8, 52.9, 59.8, 58.6]
ho_colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#98df8a", "#98df8a",
             "#7f7f7f", "#9467bd"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(ho_labels, ho_scores, color=ho_colors)
ax.set_title("BH-Vision serija 1 — held-out (v01: 73, v02: 87 — različiti setovi!)")
ax.set_ylabel("tačnost % (pravi bhas/FZS dokumenti)")
ax.set_ylim(0, 70)
for i, v in enumerate(ho_scores):
    ax.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("docs/bench-heldout.png", dpi=100)
print("charts-OK")
