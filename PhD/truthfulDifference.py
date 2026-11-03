import pandas as pd

# ============================================================
# HELPSTEER2 BASELINE
# ============================================================

baseline = {
    "Truthful %": 46.34,
    "Informative %": 67.07,
    "Both %": 39.02,
    "Win Rate %": 31.73,
    "Time (s)": 8.12
}


# ============================================================
# TABLE 1: ALL OBJECTIVES
# ============================================================

import pandas as pd

# ============================================================
# TABLE 1: ALL OBJECTIVES
# ============================================================

# ============================================================
# LAYER 13 — ALL OBJECTIVES
# ============================================================

# ============================================================
# LAYER 11 — ALL OBJECTIVES
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

df = pd.concat([
    Layer12,
    Layer13,
    Layer14,
    Layer15
], ignore_index=True)

# ============================================================
# CALCULATE DIFFERENCE FROM BASELINE
# ============================================================

result = df.copy()

for column, baseline_value in baseline.items():
    result[column] = result[column] - baseline_value


# ============================================================
# FORMAT DIFFERENCES WITH + / -
# ============================================================

for column in baseline:
    result[column] = result[column].map(
        lambda x: f"{x:+.2f}"
    )


# ============================================================
# PRINT EACH EXPERIMENT AS A SEPARATE GRID
# ============================================================

for (layer, objectives), table in result.groupby(
    ["Layer", "Objectives"],
    sort=False
):

    print("\n" + "=" * 100)
    print(f"Layer {layer} | Objectives Steered: {objectives}")
    print("=" * 100)

    display_table = table.drop(
        columns=["Layer", "Objectives"]
    )

    print(display_table.to_string(index=False))



    import pandas as pd

# ============================================================
# HELPSTEER2 BASELINE
# ============================================================

baseline = {
    "Truthful %": 46.34,
    "Informative %": 67.07,
    "Both %": 39.02,
    "Win Rate %": 31.73,
    "Time (s)": 8.12
}


# ============================================================
# LAYER 12 — HELPFUL + HONESTY
# ============================================================
Layer12 = pd.DataFrame({
    "Layer": [12, 12, 12, 12],
    "Objectives": ["All"] * 4,
    "Steer": [0.8, 0.9, 1.0, 1.2],
    "Truthful %": [53.05, 46.34, 48.78, 51.22],
    "Informative %": [80.49, 75.61, 78.66, 75.61],
    "Both %": [45.73, 38.42, 41.46, 42.07],
    "Win Rate %": [32.930, 30.49, 32.32, 33.54],
    "Time (s)": [7.25, 7.24, 7.26, 7.23]
})

Layer13 = pd.DataFrame({
    "Layer": [13, 13, 13, 13],
    "Objectives": ["All"] * 4,
    "Steer": [0.8, 0.9, 1.0, 1.2],
    "Truthful %": [47.56, 45.73, 46.95, 46.34],
    "Informative %": [75.00, 78.66, 78.05, 78.05],
    "Both %": [41.46, 42.07, 42.07, 40.85],
    "Win Rate %": [27.44, 31.10, 32.32, 30.49],
    "Time (s)": [7.25, 7.25, 7.23, 7.22]
})

Layer14 = pd.DataFrame({
    "Layer": [14, 14, 14, 14],
    "Objectives": ["All"] * 4,
    "Steer": [0.8, 0.9, 1.0, 1.2],
    "Truthful %": [48.78, 49.39, 53.05, 48.17],
    "Informative %": [70.73, 73.78, 73.78, 73.78],
    "Both %": [38.41, 39.63, 44.51, 42.07],
    "Win Rate %": [31.10, 31.10, 30.49, 32.93],
    "Time (s)": [9.87, 7.34, 7.25, 7.32]
})

Layer15 = pd.DataFrame({
    "Layer": [15, 15, 15, 15],
    "Objectives": ["All"] * 4,
    "Steer": [0.8, 0.9, 1.0, 1.2],
    "Truthful %": [49.39, 48.78, 51.22, 54.88],
    "Informative %": [73.78, 72.56, 71.95, 71.95],
    "Both %": [40.85, 39.02, 41.46, 45.73],
    "Win Rate %": [34.76, 30.49, 29.88, 37.20],
    "Time (s)": [7.24, 7.24, 7.26, 7.24]
})

# ============================================================
# COMBINE ALL RESULTS
# ============================================================

df = pd.concat([
    Layer12,
    Layer13,
    Layer14,
    Layer15
], ignore_index=True)


# ============================================================
# CALCULATE DIFFERENCE FROM BASELINE
# ============================================================

result = df.copy()

for column, baseline_value in baseline.items():
    result[column] = result[column] - baseline_value


# ============================================================
# FORMAT DIFFERENCES WITH + / -
# ============================================================

for column in baseline:
    result[column] = result[column].map(
        lambda x: f"{x:+.2f}"
    )


# ============================================================
# PRINT EACH EXPERIMENT AS A SEPARATE GRID
# ============================================================

for (layer, objectives), table in result.groupby(
    ["Layer", "Objectives"],
    sort=False
):

    print("\n" + "=" * 100)
    print(f"Layer {layer} | Objectives Steered: {objectives}")
    print("=" * 100)

    display_table = table.drop(
        columns=["Layer", "Objectives"]
    )

    print(display_table.to_string(index=False))