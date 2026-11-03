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

# ============================================================
# 4. CALCULATE DIFFERENCES FROM BASELINE
# ============================================================

frames = []

for group_name, data in datasets:

    frame = data.set_index("Steer")[metrics].copy()

    for metric in metrics:
        frame[metric] = frame[metric] - baseline[metric]

    # Label rows with layer and steering value
    frame.index = [
        f"{group_name} | {steer:.1f}"
        for steer in data["Steer"]
    ]

    frames.append(frame)

# Combine all layers
heatmap_data = pd.concat(frames)

# Round differences for display
heatmap_data = heatmap_data.round(3)

# ============================================================
# 5. CONVERT TO IMPROVEMENT SCORES
# ============================================================

improvement = heatmap_data.copy()

for metric in metrics:
    improvement[metric] = (
        heatmap_data[metric] * direction[metric]
    )

# ============================================================
# 6. NORMALIZE EACH METRIC INDEPENDENTLY
# ============================================================
#
# -1 = largest relative deterioration
#  0 = equal to baseline
# +1 = largest relative improvement
#
# Normalization is relative within each metric.
# ============================================================

normalized = improvement.copy()

for metric in metrics:

    max_abs = np.abs(improvement[metric]).max()

    if max_abs > 0:
        normalized[metric] = improvement[metric] / max_abs
    else:
        normalized[metric] = 0.0

# ============================================================
# 7. CREATE HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(9, 10))

image = ax.imshow(
    normalized.values,
    aspect="auto",
    cmap="RdYlGn",
    vmin=-1,
    vmax=1,
    interpolation="nearest"
)

# ============================================================
# 8. X AXIS
# ============================================================

display_metrics = [
    "Truthful Δ (pp)",
    "Informative Δ (pp)",
    "Both Δ (pp)",
    "Win Rate Δ (pp)",
    "Time Δ (s)"
]

ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(
    display_metrics,
    rotation=40,
    ha="right",
    rotation_mode="anchor",
    fontsize=9
)

# ============================================================
# 9. Y AXIS
# ============================================================

ax.set_yticks(np.arange(len(heatmap_data)))
ax.set_yticklabels(
    heatmap_data.index,
    rotation=0,
    fontsize=8
)

# ============================================================
# 10. ADD CELL BORDERS
# ============================================================

ax.set_xticks(
    np.arange(-0.5, len(metrics), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(heatmap_data), 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linestyle="-",
    linewidth=0.8
)

ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(axis="both", which="major", length=0, pad=4)

# ============================================================
# 11. ANNOTATE CELLS WITH ACTUAL DIFFERENCES
# ============================================================

for row in range(heatmap_data.shape[0]):

    for col in range(heatmap_data.shape[1]):

        actual_value = heatmap_data.iloc[row, col]
        color_value = normalized.iloc[row, col]

        text_color = (
            "white"
            if abs(color_value) > 0.72
            else "black"
        )

        ax.text(
            col,
            row,
            f"{actual_value:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=text_color
        )

# ============================================================
# 12. TITLE
# ============================================================

ax.set_title(
    "Performance Change Relative to Baseline — Layers 12–15",
    fontsize=12,
    pad=12
)

ax.set_xlabel("")
ax.set_ylabel("")

# ============================================================
# 13. COLOR BAR
# ============================================================

colorbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.035,
    pad=0.035,
    shrink=0.95
)

colorbar.set_label(
    "Relative improvement score\n"
    "(red = worse, yellow = baseline, green = better)",
    fontsize=9
)

colorbar.ax.tick_params(labelsize=8)

# ============================================================
# 14. FINAL FORMATTING
# ============================================================

ax.set_xlim(-0.5, len(metrics) - 0.5)
ax.set_ylim(len(heatmap_data) - 0.5, -0.5)

plt.tight_layout()

# ============================================================
# 15. SAVE AND DISPLAY
# ============================================================

plt.savefig(
    "layers12_15_performance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 16. PRINT DIFFERENCES FOR VERIFICATION
# ============================================================

print("\nDifferences from baseline:")
print(heatmap_data.to_string())