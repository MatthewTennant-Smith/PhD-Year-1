import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. BASELINE
# ============================================================

baseline = {
    "Win rate %": 27.88,
    "Helpful": 3.23,
    "Correct": 3.21,
    "Coherent": 3.30,
    "Complex": 2.63,
    "Verbose": 2.66,
    "Time (s)": 7.272
}

metrics = list(baseline.keys())

all_objectives = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["All"] * 4,
    "Steer": [0.8, 0.9, 1.0, 1.2],

    "Win rate %": [33.17, 33.65, 25.96, 25.00],
    "Helpful": [3.41, 3.31, 3.30, 3.17],
    "Correct": [3.35, 3.30, 3.27, 3.11],
    "Coherent": [3.45, 3.40, 3.38, 3.29],
    "Complex": [2.63, 2.63, 2.60, 2.47],
    "Verbose": [2.63, 2.55, 2.56, 2.48],
    "Time (s)": [7.97, 7.97, 7.96, 7.96]
})


# ============================================================
# TABLE 2: HELPFUL + CORRECT + COHERENT
# ============================================================

helpful_correct_coherent = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["Helpful + Correct + Coherent"] * 4,
    "Steer": [0.8, 0.9, 1.0, 1.2],

    "Win rate %": [32.21, 33.65, 31.25, 32.69],
    "Helpful": [3.37, 3.31, 3.42, 3.26],
    "Correct": [3.34, 3.34, 3.32, 3.15],
    "Coherent": [3.43, 3.44, 3.42, 3.27],
    "Complex": [2.68, 2.65, 2.68, 2.55],
    "Verbose": [2.60, 2.49, 2.60, 2.54],
    "Time (s)": [7.95, 7.95, 7.94, 7.96]
})


# ============================================================
# TABLE 3: HELPFUL ONLY
# ============================================================

helpful_only = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["Helpful"] * 4,
    "Steer": [0.8, 0.9, 1.0, 1.2],

    "Win rate %": [26.92, 26.92, 30.77, 28.85],
    "Helpful": [3.32, 3.30, 3.36, 3.31],
    "Correct": [3.29, 3.30, 3.34, 3.30],
    "Coherent": [3.35, 3.36, 3.42, 3.40],
    "Complex": [2.64, 2.60, 2.67, 2.63],
    "Verbose": [2.60, 2.56, 2.62, 2.57],
    "Time (s)": [7.95, 7.97, 7.97, 10.18]
})


# ============================================================
# 2. COMBINE YOUR THREE EXISTING DATAFRAMES
# ============================================================
#
# Assumes you already created:
#
# all_objectives
# helpful_correct_coherent
# helpful_only
#

df = pd.concat(
    [
        all_objectives,
        helpful_correct_coherent,
        helpful_only
    ],
    ignore_index=True
)


# ============================================================
# 3. CALCULATE DIFFERENCE FROM BASELINE
# ============================================================

diff = df.copy()

for metric, baseline_value in baseline.items():
    diff[metric] = diff[metric] - baseline_value


# ============================================================
# 4. CREATE ROW LABELS
# ============================================================

diff["Experiment"] = (
    diff["Objectives"]
    + " | "
    + diff["Steer"].astype(str)
)


# ============================================================
# 5. CREATE HEATMAP DATA
# ============================================================

heatmap_data = (
    diff
    .set_index("Experiment")[metrics]
)


# ============================================================
# 6. DEFINE WHICH DIRECTION IS "BETTER"
# ============================================================
#
# +1 = higher is better
# -1 = lower is better
#
# Here:
#   Higher Win Rate = better
#   Higher Helpful = better
#   Higher Correct = better
#   Higher Coherent = better
#   Lower Time = better
#
# Complex and Verbose are left unchanged here.

direction = {
    "Win rate %": 1,
    "Helpful": 1,
    "Correct": 1,
    "Coherent": 1,
    "Complex": 1,
    "Verbose": 1,
    "Time (s)": -1
}


# ============================================================
# 7. CONVERT DIFFERENCES INTO "IMPROVEMENT" SCORES
# ============================================================
#
# Example:
#
# Helpful:
# +0.14 -> improvement
#
# Time:
# +0.70 seconds -> worse
# -0.70 seconds -> improvement
#

improvement = heatmap_data.copy()

for metric in metrics:
    improvement[metric] = (
        heatmap_data[metric] * direction[metric]
    )


# ============================================================
# 8. NORMALIZE EACH METRIC INDEPENDENTLY
# ============================================================
#
# This is important because:
#
# Win Rate might change by +5.77
# Helpful might change by +0.19
#
# Without normalization, Win Rate dominates the colors.
#
# After normalization:
#
# -1 = worst change for that metric
#  0 = baseline
# +1 = best change for that metric
#

normalized = improvement.copy()

for metric in metrics:

    max_abs = np.max(
        np.abs(improvement[metric].values)
    )

    if max_abs != 0:
        normalized[metric] = (
            improvement[metric] / max_abs
        )
    else:
        normalized[metric] = 0


# ============================================================
# 9. CREATE HEATMAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)

image = ax.imshow(
    normalized.values,
    aspect="auto",
    cmap="RdYlGn",
    vmin=-1,
    vmax=1
)


# ============================================================
# 10. X AXIS
# ============================================================

ax.set_xticks(
    np.arange(len(metrics))
)

ax.set_xticklabels(
    metrics,
    rotation=45,
    ha="right"
)


# ============================================================
# 11. Y AXIS
# ============================================================

ax.set_yticks(
    np.arange(len(heatmap_data))
)

ax.set_yticklabels(
    heatmap_data.index
)


# ============================================================
# 12. WRITE ACTUAL DIFFERENCES IN CELLS
# ============================================================
#
# IMPORTANT:
#
# The colors use the normalized improvement score,
# but the text shows the REAL difference from baseline.
#
# Example:
#
# +0.14
# -0.06
# +5.77
#

for row in range(heatmap_data.shape[0]):

    for col in range(heatmap_data.shape[1]):

        value = heatmap_data.iloc[row, col]

        ax.text(
            col,
            row,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=9
        )


# ============================================================
# 13. TITLE
# ============================================================

ax.set_title(
    "Performance Change Relative to HelpSteer2 Baseline",
    fontsize=14
)


# ============================================================
# 14. COLOR BAR
# ============================================================

colorbar = plt.colorbar(
    image,
    ax=ax
)

colorbar.set_label(
    "Relative performance change\n"
    "(red = worse, green = better)"
)


# ============================================================
# 15. FINISH
# ============================================================

plt.tight_layout()

plt.show()


# ============================================================
# BAR CHARTS
# One chart per metric
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


# Order the experiments consistently
objective_order = [
    "All",
    "Helpful + Correct + Coherent",
    "Helpful"
]

# Steering coefficients
steer_values = [0.8, 0.9, 1.0, 1.2]

# Position of groups on x-axis
x = np.arange(len(steer_values))

# Width of each bar
width = 0.25


# ============================================================
# CREATE ONE BAR CHART FOR EACH METRIC
# ============================================================

for metric in metrics:

    fig, ax = plt.subplots(figsize=(9, 5))

    # --------------------------------------------------------
    # Add bars for each objective configuration
    # --------------------------------------------------------

    for i, objective in enumerate(objective_order):

        # Get data for this objective
        data = diff[
            diff["Objectives"] == objective
        ].sort_values("Steer")

        # Position bars side-by-side
        offset = (i - 1) * width

        bars = ax.bar(
            x + offset,
            data[metric],
            width,
            label=objective
        )

        # ----------------------------------------------------
        # Add values above/below bars
        # ----------------------------------------------------

        for bar, value in zip(bars, data[metric]):

            # Put text slightly above positive values
            # and below negative values
            offset_text = 3 if value >= 0 else -12

            ax.annotate(
                f"{value:+.2f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height()
                ),
                xytext=(0, offset_text),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8
            )


    # ========================================================
    # BASELINE LINE
    # ========================================================

    ax.axhline(
        y=0,
        linewidth=1
    )


    # ========================================================
    # AXES
    # ========================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        steer_values
    )

    ax.set_xlabel(
        "Steering coefficient"
    )

    ax.set_ylabel(
        "Difference from baseline"
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        f"{metric} — Change from HelpSteer2 Baseline"
    )


    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(
        title="Objectives Steered"
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout()

    plt.show()

    import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# LINE CHARTS
# ============================================================

objective_order = [
    "All",
    "Helpful + Correct + Coherent",
    "Helpful"
]

# Metrics to plot
metrics = [
    "Win rate %",
    "Helpful",
    "Correct",
    "Coherent",
    "Complex",
    "Verbose",
    "Time (s)"
]

# Steering values
steer_values = [0.8, 0.9, 1.0, 1.2]


# ============================================================
# CREATE ONE LINE CHART PER METRIC
# ============================================================

for metric in metrics:

    fig, ax = plt.subplots(figsize=(9, 5))

    for objective in objective_order:

        data = (
            diff[
                diff["Objectives"] == objective
            ]
            .sort_values("Steer")
        )

        ax.plot(
            data["Steer"],
            data[metric],
            marker="o",
            linewidth=2,
            label=objective
        )

    # Baseline = 0 difference
    ax.axhline(
        0,
        linewidth=1
    )

    # Labels
    ax.set_xlabel(
        "Steering coefficient"
    )

    ax.set_ylabel(
        "Difference from baseline"
    )

    ax.set_title(
        f"{metric} — Change from HelpSteer2 baseline"
    )

    ax.set_xticks(
        steer_values
    )

    ax.legend(
        title="Objectives Steered"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


    import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================

objective_order = [
    "All",
    "Helpful + Correct + Coherent",
    "Helpful"
]

steer_values = [0.8, 0.9, 1.0, 1.2]

# Radar 1: core objectives
core_metrics = [
    "Win rate %",
    "Helpful",
    "Correct",
    "Coherent"
]

# Radar 2: extended objectives
extended_metrics = [
    "Win rate %",
    "Helpful",
    "Correct",
    "Coherent",
    "Complex",
    "Verbose"
]


# ============================================================
# Radar function
# ============================================================

def make_radar(data, objective, metrics):

    # Select one objective configuration
    radar_data = data[
        data["Objectives"] == objective
    ].copy()

    # Make sure steering values are ordered
    radar_data = radar_data.sort_values("Steer")

    # Convert to numeric
    for metric in metrics:
        radar_data[metric] = radar_data[metric].astype(float)

    # --------------------------------------------------------
    # Normalize each metric across steering coefficients
    # --------------------------------------------------------
    
    normalized = radar_data.copy()

    for metric in metrics:

        min_val = radar_data[metric].min()
        max_val = radar_data[metric].max()

        if max_val != min_val:
            normalized[metric] = (
                radar_data[metric] - min_val
            ) / (max_val - min_val)
        else:
            normalized[metric] = 0.5

    # --------------------------------------------------------
    # Radar angles
    # --------------------------------------------------------

    angles = np.linspace(
        0,
        2 * np.pi,
        len(metrics),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(polar=True)
    )

    # --------------------------------------------------------
    # Plot each steering coefficient
    # --------------------------------------------------------

    for steer in steer_values:

        row = normalized[
            normalized["Steer"] == steer
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        values = [
            row[metric]
            for metric in metrics
        ]

        values += values[:1]

        ax.plot(
            angles,
            values,
            marker="o",
            linewidth=2,
            label=f"Steer = {steer}"
        )

        ax.fill(
            angles,
            values,
            alpha=0.05
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    ax.set_ylim(0, 1)

    ax.set_yticks([
        0,
        0.25,
        0.50,
        0.75,
        1.00
    ])

    ax.set_yticklabels([
        "0",
        "0.25",
        "0.50",
        "0.75",
        "1.00"
    ])

    ax.set_title(
        f"{objective}\nPerformance Profile",
        pad=25
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1)
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Generate all radar charts
# ============================================================

for objective in objective_order:

    # Core radar
    make_radar(
        df,
        objective,
        core_metrics
    )

    # Extended radar
    make_radar(
        df,
        objective,
        extended_metrics
    )