
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DATA
# ============================================================

data = {
    "Layer": ["Layer 12", "Layer 13", "Layer 14", "Layer 15"],
    "Helpfulness": [0.07, 0.08, 0.08, 0.09],
    "Correctness": [-0.01, 0.13, 0.12, 0.00],
    "Coherence": [0.06, 0.14, 0.11, 0.05],
    "Complexity": [-0.03, 0.02, 0.06, -0.02],
    "Verbosity": [-0.10, -0.17, -0.03, -0.10],
    "Generation time": [0.06, 0.05, 0.05, 0.02],
}

df = pd.DataFrame(data)

# ============================================================
# PLOT
# ============================================================

metrics = [
    "Helpfulness",
    "Correctness",
    "Coherence",
    "Complexity",
    "Verbosity",
    "Generation time",
]

fig, ax = plt.subplots(figsize=(10, 6))

for metric in metrics:
    ax.plot(
        df["Layer"],
        df[metric],
        marker="o",
        linewidth=2,
        markersize=6,
        label=metric
    )

# Baseline reference
ax.axhline(y=0, linewidth=1, linestyle="--")

ax.set_title(
    "Layer Comparison: Quality and Generation Time",
    fontsize=14
)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Change Relative to Baseline", fontsize=11)

ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()

plt.savefig(
    "layer_quality_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DATA
# ============================================================

data = {
    "Layer": ["Layer 12", "Layer 13", "Layer 14", "Layer 15"],
    "Helpfulness": [0.07, 0.08, 0.08, 0.09],
    "Correctness": [-0.01, 0.13, 0.12, 0.00],
    "Coherence": [0.06, 0.14, 0.11, 0.05],
    "Complexity": [-0.03, 0.02, 0.06, -0.02],
    "Verbosity": [-0.10, -0.17, -0.03, -0.10],
    "Generation time": [0.06, 0.05, 0.05, 0.02],
    "Win rate": [5.77,5.77 ,3.85, 3.37 ]
}

df = pd.DataFrame(data)

# ============================================================
# FLIPPED AXES: METRICS ON X, VALUES ON Y
# ============================================================

metrics = [
    "Helpfulness",
    "Correctness",
    "Coherence",
    "Complexity",
    "Verbosity",
    "Generation time",
    "Win rate"
]

fig, ax = plt.subplots(figsize=(11, 6))

# Each layer is now a separate line across the metrics
for _, row in df.iterrows():
    ax.plot(
        metrics,
        [row[metric] for metric in metrics],
        marker="o",
        linewidth=2,
        markersize=6,
        label=row["Layer"]
    )

# Baseline reference
ax.axhline(y=0, color="black", linewidth=1, linestyle="--")

ax.set_title(
    "Layer Comparison: Quality and Generation Time",
    fontsize=14
)
ax.set_xlabel("Metric", fontsize=11)
ax.set_ylabel("Change Relative to Baseline", fontsize=11)

plt.xticks(rotation=30, ha="right")
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(title="Layer")

plt.tight_layout()

plt.savefig(
    "layer_comparison_flipped_axes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
