
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. DEFINE BASELINE SCORES
# ============================================================

baseline = {
    "Truthful %": 46.34,
    "Informative %": 67.07,
    "Both %": 39.02,
    "Win Rate %": 31.73,
    "Time (s)": 8.12
}

# ============================================================
# 2. GROUP THE DATASETS
# ============================================================

Layer12 = pd.DataFrame({
    "Layer": [12, 12, 12, 12],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Truthful %": [47.56, 48.78, 46.95, 46.95],
    "Informative %": [67.68, 68.29, 69.51, 67.68],
    "Both %": [40.85, 39.63, 39.63, 39.02],
    "Win Rate %": [27.44, 28.05, 31.71, 28.66],
    "Time (s)": [7.24, 7.25, 7.27, 7.24]
})

Layer13 = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Truthful %": [45.73, 48.17, 47.56, 47.56],
    "Informative %": [67.07, 64.63, 67.07, 64.02],
    "Both %": [35.98, 37.20, 39.02, 34.76],
    "Win Rate %": [30.49, 31.71, 29.88, 28.05],
    "Time (s)": [7.25, 7.26, 7.24, 7.25]
})

Layer14 = pd.DataFrame({
    "Layer": [14, 14, 14, 14],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Truthful %": [48.78, 45.12, 48.17, 46.34],
    "Informative %": [66.46, 67.07, 69.51, 68.90],
    "Both %": [39.63, 38.41, 40.85, 38.41],
    "Win Rate %": [31.10, 29.27, 28.05, 30.49],
    "Time (s)": [9.94, 7.35, 7.26, 7.32]
})

Layer15 = pd.DataFrame({
    "Layer": [15, 15, 15, 15],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Truthful %": [45.73, 46.95, 46.95, 45.12],
    "Informative %": [65.85, 67.07, 65.85, 65.24],
    "Both %": [37.20, 37.80, 36.59, 37.80],
    "Win Rate %": [28.66, 31.10, 27.44, 28.66],
    "Time (s)": [7.25, 7.25, 7.26, 7.25]
})


datasets = [
    ("Layer 12", Layer12),
    ("Layer 13", Layer13),
    ("Layer 14", Layer14),
    ("Layer 15", Layer15)
]

# ============================================================
# 3. DEFINE METRICS AND IMPROVEMENT DIRECTIONS
# ============================================================

metrics = list(baseline.keys())

# +1 = higher is better
# -1 = lower is better
direction = {
    "Truthful %": 1,
    "Informative %": 1,
    "Both %": 1,
    "Win Rate %": 1,
    "Time (s)": -1
}

display_metrics = {
    "Truthful %": "Truthfulness (pp)",
    "Informative %": "Informativeness (pp)",
    "Both %": "Both (pp)",
    "Win Rate %": "Win rate (pp)",
    "Time (s)": "Time saved (s)"
}

# ============================================================
# 4. CALCULATE DIFFERENCES FROM BASELINE
# ============================================================

frames = []

for layer_name, data in datasets:
    frame = data[["Steer"] + metrics].copy()
    frame["Layer"] = layer_name

    for metric in metrics:
        frame[metric] = frame[metric] - baseline[metric]

    frames.append(frame)

results = pd.concat(frames, ignore_index=True)

# Positive values mean improvement for every metric,
# including time (where a reduction is an improvement).
for metric in metrics:
    results[metric] = results[metric] * direction[metric]

# ============================================================
# 5. CREATE GROUPED BAR CHART
# ============================================================



# ============================================================
# GROUPED BAR CHART — 2 ROWS, 3 COLUMNS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. SETTINGS
# ============================================================

datasets = [
    ("Layer 12", Layer12),
    ("Layer 13", Layer13),
    ("Layer 14", Layer14),
    ("Layer 15", Layer15)
]

metrics = list(baseline.keys())

direction = {
    "Truthful %": 1,
    "Informative %": 1,
    "Both %": 1,
    "Win Rate %": 1,
    "Time (s)": -1
}

metric_titles = {
    "Truthful %": "Truthfulness",
    "Informative %": "Informativeness",
    "Both %": "Both",
    "Win Rate %": "Win rate %",
    "Time (s)": "Time (s)"
}

# ============================================================
# 2. CALCULATE BASELINE DIFFERENCES
# ============================================================

frames = []

for layer_name, data in datasets:
    frame = data[["Steer"] + metrics].copy()
    frame["Layer"] = layer_name

    for metric in metrics:
        frame[metric] = (
            frame[metric] - baseline[metric]
        ) * direction[metric]

    frames.append(frame)

results = pd.concat(frames, ignore_index=True)

# ============================================================
# 3. CREATE 2 x 3 SUBPLOT GRID
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 8.5),
    sharex=True
)

axes = axes.flatten()

steer_values = [100, 200, 300, 400]
layer_names = [name for name, _ in datasets]

x = np.arange(len(steer_values))
bar_width = 0.18

# ============================================================
# 4. PLOT EACH METRIC
# ============================================================

for ax, metric in zip(axes, metrics):

    for i, layer_name in enumerate(layer_names):

        layer_data = (
            results[results["Layer"] == layer_name]
            .set_index("Steer")
            .reindex(steer_values)
        )

        values = layer_data[metric].to_numpy()

        positions = x + (
            i - (len(layer_names) - 1) / 2
        ) * bar_width

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=layer_name,
            edgecolor="black",
            linewidth=0.35
        )

        # Numerical annotations
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:+.2f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    value
                ),
                xytext=(
                    0,
                    3 if value >= 0 else -3
                ),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=6,
                rotation=90
            )

    # Baseline reference
    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_title(
        metric_titles[metric],
        fontsize=10,
        fontweight="bold",
        pad=5
    )

    ax.set_ylabel(
        "Seconds saved" if metric == "Time (s)"
        else "Change (pp)",
        fontsize=8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(v) for v in steer_values],
        fontsize=8
    )

    ax.set_xlabel(
        "Steering value",
        fontsize=8
    )

    ax.tick_params(axis="y", labelsize=8)

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.5,
        alpha=0.3
    )

    ax.set_axisbelow(True)
    ax.margins(y=0.22)

# ============================================================
# 5. REMOVE UNUSED SIXTH PANEL
# ============================================================

for ax in axes[len(metrics):]:
    ax.remove()

# ============================================================
# 6. FIGURE TITLE
# ============================================================

fig.suptitle(
    "Performance Change Relative to Baseline — Layers 12–15",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

# ============================================================
# 7. SHARED LEGEND AT THE BOTTOM
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Layer",
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=4,
    fontsize=8,
    title_fontsize=8,
    frameon=True
)

# ============================================================
# 8. LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.07,
    right=0.99,
    top=0.90,
    bottom=0.12,
    wspace=0.22,
    hspace=0.32
)

# ============================================================
# 9. SAVE AND DISPLAY
# ============================================================

plt.savefig(
    "layers12_15_grouped_bar_grid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
