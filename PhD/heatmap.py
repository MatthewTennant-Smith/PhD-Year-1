
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. DEFINE BASELINE SCORES
# ============================================================

baseline = {
    "Win rate %": 35.42,
    "Helpful": 2.15,
    "Honesty": 2.19,
    "Instruction": 2.13,
    "Truthful": 2.12,
    "Time (s)": 8.12
}



# ============================================================
# 2. LAYER 14 RAW RESULTS
# ============================================================



# ============================================================
# LAYER 12 — ALL OBJECTIVES
# ============================================================

all_objectives = pd.DataFrame({
    "Layer": [15, 15, 15, 15],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [35.62, 35.10, 36.15, 36.67],
    "Helpful": [2.14, 2.14, 2.06, 2.12],
    "Honesty": [2.18, 2.16, 2.07, 2.12],
    "Instruction": [2.12, 2.11, 2.00, 2.05],
    "Truthful": [2.08, 2.07, 1.98, 2.03],
    "Time (s)": [7.75, 7.75, 7.73, 7.73]
})


# ============================================================
# LAYER 15 — HELPFUL + HONESTY
# ============================================================

helpful_honesty = pd.DataFrame({
    "Layer": [15, 15, 15, 15],
    "Objectives": ["Helpful + Honesty"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [33.44, 36.35, 35.73, 35.52],
    "Helpful": [2.08, 1.97, 1.98, 2.12],
    "Honesty": [2.12, 1.97, 2.00, 2.13],
    "Instruction": [2.05, 1.93, 1.95, 2.08],
    "Truthful": [2.03, 1.89, 1.90, 2.08],
    "Time (s)": [7.75, 7.74, 7.74, 7.72]
})


# ============================================================
# LAYER 15 — HELPFUL ONLY
# ============================================================

helpful_only = pd.DataFrame({
    "Layer": [15, 15, 15, 15],
    "Objectives": ["Helpful"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [37.40, 38.54, 37.92, 38.23],
    "Helpful": [2.08, 1.99, 2.14, 2.01],
    "Honesty": [2.10, 2.00, 2.19, 2.03],
    "Instruction": [2.06, 1.93, 2.13, 1.98],
    "Truthful": [2.03, 1.92, 2.13, 1.97],
    "Time (s)": [7.73, 7.74, 7.73, 7.73]
})



# ============================================================
# 3. GROUP THE DATASETS
# ============================================================

datasets = [
    ("All", all_objectives),
    ("Helpful + Honest", helpful_honesty),
    ("Helpful", helpful_only)
]


# ============================================================
# 4. DEFINE METRICS AND IMPROVEMENT DIRECTIONS
# ============================================================

metrics = list(baseline.keys())

# +1 = higher is better
# -1 = lower is better
direction = {
    "Win rate %": 1,
    "Helpful": 1,
    "Honesty": 1,
    "Instruction": 1,
    "Truthful": 1,
    "Time (s)": -1
}


# ============================================================
# 5. CALCULATE DIFFERENCES FROM BASELINE
# ============================================================

frames = []

for group_name, data in datasets:

    # Calculate raw difference: score - baseline
    frame = data.set_index("Steer")[metrics].copy()

    for metric in metrics:
        frame[metric] = frame[metric] - baseline[metric]

    # Create labels for each experiment and steering value
    frame.index = [
        f"{group_name} | {steer:.1f}"
        for steer in data["Steer"]
    ]

    frames.append(frame)


# Combine all groups
heatmap_data = pd.concat(frames)

# Keep these values as the actual differences from baseline
heatmap_data = heatmap_data.round(3)


# ============================================================
# 6. CONVERT TO IMPROVEMENT SCORES
# ============================================================

improvement = heatmap_data.copy()

for metric in metrics:
    improvement[metric] = (
        heatmap_data[metric] * direction[metric]
    )


# ============================================================
# 7. NORMALIZE EACH METRIC INDEPENDENTLY
# ============================================================
#
# -1 = worst relative change within that metric
#  0 = equal to baseline
# +1 = best relative change within that metric
#
# This is relative normalization, not a percentage.
# ============================================================

normalized = improvement.copy()

for metric in metrics:

    max_abs = np.abs(improvement[metric]).max()

    if max_abs > 0:
        normalized[metric] = improvement[metric] / max_abs
    else:
        normalized[metric] = 0.0


# ============================================================
# 8. CREATE HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6.5))

image = ax.imshow(
    normalized.values,
    aspect="auto",
    cmap="RdYlGn",
    vmin=-1,
    vmax=1,
    interpolation="nearest"
)


# ============================================================
# 9. X AXIS
# ============================================================

display_metrics = [
    "Win rate Δ (pp)",
    "Helpful Δ",
    "Honest Δ",
    "Instruction Δ",
    "Truthful Δ",
    "Time Δ (s)"
]

ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(
    display_metrics,
    rotation=45,
    ha="right",
    rotation_mode="anchor",
    fontsize=7
)


# ============================================================
# 10. Y AXIS
# ============================================================

ax.set_yticks(np.arange(len(heatmap_data)))
ax.set_yticklabels(
    heatmap_data.index,
    rotation=0,
    fontsize=6.5
)


# ============================================================
# 11. ADD CELL BORDERS
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
    linewidth=0.6
)

ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(axis="both", which="major", length=0, pad=3)


# ============================================================
# 12. ANNOTATE CELLS WITH ACTUAL DIFFERENCES
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
            fontsize=6,
            color=text_color
        )


# ============================================================
# 13. TITLE
# ============================================================

ax.set_title(
    "Layer 15: Performance Change Relative to Baseline",
    fontsize=10,
    pad=8
)

ax.set_xlabel("")
ax.set_ylabel("")


# ============================================================
# 14. COLOR BAR
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
    fontsize=7
)

colorbar.ax.tick_params(labelsize=7)


# ============================================================
# 15. FINAL FORMATTING
# ============================================================

ax.set_xlim(-0.5, len(metrics) - 0.5)
ax.set_ylim(len(heatmap_data) - 0.5, -0.5)

plt.tight_layout()


# ============================================================
# 16. SAVE AND DISPLAY
# ============================================================

plt.savefig(
    "layer14_performance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. PRINT DIFFERENCES FOR VERIFICATION
# ============================================================

print("\nDifferences from baseline:")
print(heatmap_data.to_string())
