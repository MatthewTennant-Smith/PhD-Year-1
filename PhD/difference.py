import pandas as pd

# ============================================================
# HELPSTEER2 BASELINE
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

#===============================================
# COMBINE ALL RESULTS INTO ONE DATAFRAME
# ============================================================

df = pd.concat([
    all_objectives,
    helpful_honesty,
    helpful_only
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
    "Win rate %": 35.42,
    "Helpful": 2.15,
    "Honesty": 2.19,
    "Instruction": 2.13,
    "Truthful": 2.12,
    "Time (s)": 8.12
}



# ============================================================
# LAYER 12 — HELPFUL + HONESTY
# ============================================================
all_objectives = pd.DataFrame({
    "Layer": [14, 14, 14, 14],
    "Objectives": ["All"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [36.67, 35.83, 35.83, 35.42],
    "Helpful": [2.08, 2.18, 2.15, 2.13],
    "Honesty": [2.10, 2.25, 2.15, 2.14],
    "Instruction": [2.04, 2.17, 2.10, 2.07],
    "Truthful": [2.01, 2.15, 2.07, 2.04],
    "Time (s)": [7.75, 7.72, 7.74, 7.72]
})


# ============================================================
# LAYER 14 — HELPFUL + HONESTY
# ============================================================

helpful_honesty = pd.DataFrame({
    "Layer": [14, 14, 14, 14],
    "Objectives": ["Helpful + Honesty"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [36.56, 37.81, 36.56, 36.25],
    "Helpful": [2.09, 2.09, 2.14, 2.19],
    "Honesty": [2.11, 2.12, 2.14, 2.21],
    "Instruction": [2.04, 2.05, 2.06, 2.11],
    "Truthful": [2.01, 2.03, 2.03, 2.13],
    "Time (s)": [7.74, 7.73, 7.73, 7.75]
})


# ============================================================
# LAYER 14 — HELPFUL ONLY
# ============================================================

helpful_only = pd.DataFrame({
    "Layer": [14, 14, 14, 14],
    "Objectives": ["Helpful"] * 4,
    "Steer": [100, 200, 300, 400],
    "Win rate %": [36.15, 36.25, 36.25, 35.42],
    "Helpful": [2.12, 2.16, 2.15, 2.05],
    "Honesty": [2.15, 2.19, 2.15, 2.07],
    "Instruction": [2.09, 2.14, 2.09, 2.01],
    "Truthful": [2.05, 2.09, 2.05, 1.99],
    "Time (s)": [7.72, 7.74, 7.73, 7.74]
})


# ============================================================
# COMBINE ALL RESULTS
# ============================================================

df = pd.concat([
    all_objectives,
    helpful_honesty,
    helpful_only
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