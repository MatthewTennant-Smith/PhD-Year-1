import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# BASELINE
# ============================================================

baseline = {
    "Win rate %": 27.88,
    "Helpful": 3.23,
    "Correct": 3.21,
    "Coherent": 3.30,
    "Complex": 2.63,
    "Verbose": 2.66,
    "Time (s)": 7.9
}


# ============================================================
# LAYER 12 DATA
# ============================================================

# ============================================================
# LAYER 12 — ALL OBJECTIVES
# ============================================================

layer12_all = pd.DataFrame({
    "Layer": [12, 12, 12, 12],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [28.37, 29.33, 27.88, 28.85],
    "Helpful": [3.16, 3.26, 3.28, 3.26],
    "Correct": [3.11, 3.22, 3.15, 3.22],
    "Coherent": [3.24, 3.33, 3.30, 3.32],
    "Complex": [2.48, 2.62, 2.62, 2.60],
    "Verbose": [2.54, 2.46, 2.55, 2.59],
    "Time (s)": [7.96, 7.95, 7.95, 7.95]
})


# ============================================================
# LAYER 12 — HELPFUL + CORRECT + COHERENT
# ============================================================

layer12_hcc = pd.DataFrame({
    "Layer": [12, 12, 12, 12],
    "Objectives": ["Helpful + Correct + Coherent"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [26.92, 27.88, 26.92, 26.92],
    "Helpful": [3.24, 3.27, 3.26, 3.26],
    "Correct": [3.27, 3.19, 3.16, 3.15],
    "Coherent": [3.33, 3.30, 3.27, 3.27],
    "Complex": [2.64, 2.60, 2.60, 2.55],
    "Verbose": [2.56, 2.59, 2.57, 2.54],
    "Time (s)": [7.93, 7.95, 7.94, 7.97]
})


# ============================================================
# LAYER 12 — HELPFUL ONLY
# ============================================================

layer12_helpful = pd.DataFrame({
    "Layer": [12, 12, 12, 12],
    "Objectives": ["Helpful"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [28.37, 27.40, 29.33, 29.81],
    "Helpful": [3.39, 3.28, 3.41, 3.37],
    "Correct": [3.34, 3.20, 3.29, 3.36],
    "Coherent": [3.47, 3.30, 3.40, 3.46],
    "Complex": [2.70, 2.64, 2.70, 2.71],
    "Verbose": [2.68, 2.58, 2.61, 2.70],
    "Time (s)": [7.95, 7.96, 7.96, 10.32]
})



# ============================================================
# LAYER 13 DATA
# ============================================================

# ============================================================
# LAYER 13 — ALL OBJECTIVES
# ============================================================

layer13_all = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [25.48, 26.44, 28.37, 28.37],
    "Helpful": [3.15, 3.22, 3.12, 3.28],
    "Correct": [3.13, 3.18, 3.10, 3.26],
    "Coherent": [3.27, 3.26, 3.22, 3.38],
    "Complex": [2.59, 2.55, 2.55, 2.66],
    "Verbose": [2.54, 2.62, 2.55, 2.61],
    "Time (s)": [7.96, 7.97, 7.96, 7.96]
})


# ============================================================
# LAYER 13 — HELPFUL + CORRECT + COHERENT
# ============================================================

layer13_hcc = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["Helpful + Correct + Coherent"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [26.92, 27.40, 26.92, 30.29],
    "Helpful": [3.19, 3.26, 3.24, 3.34],
    "Correct": [3.16, 3.21, 3.23, 3.24],
    "Coherent": [3.26, 3.30, 3.28, 3.37],
    "Complex": [2.60, 2.63, 2.57, 2.67],
    "Verbose": [2.55, 2.61, 2.54, 2.61],
    "Time (s)": [7.98, 7.95, 7.97, 7.95]
})


# ============================================================
# LAYER 13 — HELPFUL ONLY
# ============================================================

layer13_helpful = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["Helpful"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [26.92, 29.81, 27.88, 30.77],
    "Helpful": [3.21, 3.29, 3.30, 3.29],
    "Correct": [3.21, 3.28, 3.21, 3.21],
    "Coherent": [3.26, 3.38, 3.31, 3.30],
    "Complex": [2.60, 2.64, 2.60, 2.61],
    "Verbose": [2.58, 2.57, 2.54, 2.62],
    "Time (s)": [7.97, 7.96, 7.97, 7.97]
})


# ============================================================
# COMBINE ALL RESULTS
# ============================================================

df = pd.concat([
    layer12_all,
    layer12_hcc,
    layer12_helpful,
    layer13_all,
    layer13_hcc,
    layer13_helpful
], ignore_index=True)


# ============================================================
# CALCULATE DIFFERENCE FROM BASELINE
# ============================================================

df_diff = df.copy()

for metric, baseline_value in baseline.items():
    df_diff[metric] = df_diff[metric] - baseline_value


# ============================================================
# SETTINGS
# ============================================================

metrics = [
    "Win rate %",
    "Helpful",
    "Correct",
    "Coherent",
    "Complex",
    "Verbose",
    "Time (s)"
]

steer_values = [100, 200, 300, 400]

configurations = [
    ("All", 12),
    ("All", 13),
    ("Helpful + Correct + Coherent", 12),
    ("Helpful + Correct + Coherent", 13),
    ("Helpful", 12),
    ("Helpful", 13)
]

labels = [
    "All — L12",
    "All — L13",
    "H+C+C — L12",
    "H+C+C — L13",
    "Helpful — L12",
    "Helpful — L13"
]


# ============================================================
# ALL GRAPHS ON ONE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(22, 11)
)

axes = axes.flatten()

x = np.arange(len(steer_values))

width = 0.12

offsets = (
    np.arange(len(configurations))
    - (len(configurations) - 1) / 2
) * width


# ============================================================
# PLOT EACH METRIC
# ============================================================

for ax, metric in zip(axes, metrics):

    for i, ((objective, layer), label) in enumerate(
        zip(configurations, labels)
    ):

        data = df_diff[
            (df_diff["Objectives"] == objective) &
            (df_diff["Layer"] == layer)
        ].sort_values("Steer")

        values = data[metric].values

        ax.bar(
            x + offsets[i],
            values,
            width,
            label=label
        )

    # Zero = baseline
    ax.axhline(
        y=0,
        linewidth=1.5
    )

    # Title
    ax.set_title(
        metric,
        fontsize=13,
        fontweight="bold"
    )

    # X-axis
    ax.set_xlabel(
        "Steering steps"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        steer_values
    )

    # Y-axis
    if metric == "Win rate %":

        ax.set_ylabel(
            "Change (percentage points)"
        )

    elif metric == "Time (s)":

        ax.set_ylabel(
            "Change (seconds)"
        )

    else:

        ax.set_ylabel(
            "Change from baseline"
        )

    # Grid
    ax.grid(
        axis="y",
        alpha=0.3
    )


# ============================================================
# REMOVE UNUSED 8TH SUBPLOT
# ============================================================

fig.delaxes(
    axes[-1]
)


# ============================================================
# ONE SHARED LEGEND
# ============================================================

handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    title="Configuration",
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.01)
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Performance Change Relative to Baseline: Layers 12 and 13",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0.08, 1, 0.95]
)


# ============================================================
# SHOW
# ============================================================

plt.show()