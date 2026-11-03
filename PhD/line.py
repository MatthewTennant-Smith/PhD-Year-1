import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame({
    "Layer": [12, 13, 14, 15],
    "Win rate %": [3.12, 6.66, 4.37, 4.79],
    "Helpful": [0.06, 0.14, 0.2, 0.18],
    "Honesty": [0.06, 0.15, 0.23, 0.18],
    "Instruction": [0.09, 0.14, 0.22, 0.17],
    "Truthful": [0.09, 0.18, 0.1, 0.17],
})

metrics = [
    "Win rate %",
    "Helpful",
    "Honesty",
    "Instruction",
    "Truthful"
]

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(16, 9)
)

axes = axes.flatten()

for ax, metric in zip(axes, metrics):

    plot_data = df.dropna(subset=[metric])

    ax.plot(
        plot_data["Layer"],
        plot_data[metric],
        marker="o",
        linewidth=2,
        markersize=7
    )

    # Label each point
    for _, row in plot_data.iterrows():
        ax.annotate(
            f'{row[metric]:+.2f}',
            (row["Layer"], row[metric]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    # Baseline reference
    ax.axhline(
        y=0,
        color="black",
        linewidth=1,
        linestyle="--",
        alpha=0.7
    )

    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Change from baseline")
    ax.set_xticks([12, 13, 14, 15])
    ax.grid(True, alpha=0.3)

# Remove the unused sixth subplot
fig.delaxes(axes[-1])

fig.suptitle(
    "Performance Across Layers — Helpful + Honest, Steering = 1.0",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Print values used in the plots
print(df.to_string(index=False))