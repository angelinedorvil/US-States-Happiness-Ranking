import pandas as pd
import numpy as np
import os
import glob
from utils.metrics_utils import process_chr_metric, normalize

# Years of data to consider
YEARS = range(2020, 2026)  # 2015–2025 inclusive

# === Target metrics from County Health Rankings ===
TARGET_METRICS = {
    # Health outcomes
    "v001_rawvalue": True,   # Premature death (lower is better)
    "v127_rawvalue": True,   # Premature age-adjusted mortality (lower is better)
    "v128_rawvalue": True,   # Child mortality (lower is better)
    "v129_rawvalue": True,   # Infant mortality (lower is better)
    "v135_rawvalue": True,   # Injury deaths (lower is better)
    "v161_rawvalue": True,   # Suicides (lower is better)
    "v015_rawvalue": True,   # Homicides (lower is better)
    "v039_rawvalue": True,   # Motor vehicle crash deaths (lower is better)
    "v148_rawvalue": True,   # Firearm fatalities (lower is better)
    
    # Environmental / infrastructure factors
    "v125_rawvalue": True,   # Air pollution (PM2.5) (lower is better)
    "v124_rawvalue": True,   # Drinking water violations (lower is better)
    "v179_rawvalue": False,  # Access to parks (higher is better)
    "v182_rawvalue": True,   # Adverse climate events (lower is better)
    "v166_rawvalue": False,  # Broadband access (higher is better)
    "v181_rawvalue": False,  # Library access (higher is better)
    "v156_rawvalue": True,   # Traffic volume (lower is better)
    
    # Behavioral / social outcomes
    "v134_rawvalue": True,   # Alcohol-impaired driving deaths (lower is better)
    "v139_rawvalue": True,   # Food insecurity (lower is better)
    "v083_rawvalue": True,   # Limited access to healthy foods (lower is better)
    "v133_rawvalue": False,  # Food environment index (higher is better)
    "v155_rawvalue": False,  # Flu vaccination (higher is better)
}

# === Target metric weights (editable later) ===
TARGET_METRIC_WEIGHTS = {
    # Health outcomes
    "v001_rawvalue": 1.0,   # Premature death
    "v127_rawvalue": 1.0,   # Premature age-adjusted mortality
    "v128_rawvalue": 1.0,   # Child mortality
    "v129_rawvalue": 1.0,   # Infant mortality
    "v135_rawvalue": 1.0,   # Injury deaths
    "v161_rawvalue": 1.0,   # Suicides
    "v015_rawvalue": 1.0,   # Homicides
    "v039_rawvalue": 1.0,   # Motor vehicle crash deaths
    "v148_rawvalue": 1.0,   # Firearm fatalities

    # Environmental / infrastructure
    "v125_rawvalue": 1.0,   # Air pollution (PM2.5)
    "v124_rawvalue": 1.0,   # Drinking water violations
    "v179_rawvalue": 1.0,   # Access to parks
    "v182_rawvalue": 1.0,   # Adverse climate events
    "v166_rawvalue": 1.0,   # Broadband access
    "v181_rawvalue": 1.0,   # Library access
    "v156_rawvalue": 1.0,   # Traffic volume

    # Behavioral / social outcomes
    "v134_rawvalue": 1.0,   # Alcohol-impaired driving deaths
    "v139_rawvalue": 1.0,   # Food insecurity
    "v083_rawvalue": 1.0,   # Limited access to healthy foods
    "v133_rawvalue": 1.0,   # Food environment index
    "v155_rawvalue": 1.0,   # Flu vaccination
}


# Classify states based on happiness index
def classify_percentiles(df, column="env_safety_index", n_classes=5):
    """
    Adds a percentile-based class column.
    n_classes = 3 (tertile), 4 (quartile), 5 (quintile), etc.
    """
    df = df.copy()
    # labels = [f"Tier_{i}" for i in range(1, n_classes + 1)]
    labels = list(range(1, n_classes + 1))
    df["Percentile_Class"] = pd.qcut(df[column], q=n_classes, labels=labels)
    return df

# Process hate crime data
def process_hate_crime(path="data/hate_crime.csv"):
    df = pd.read_csv(path)
    df = df[df["data_year"] >= 2020]
    df_state = df.groupby("state_name", as_index=False)["incident_id"].count()
    df_state.rename(columns={"state_name": "State", "incident_id": "hate_crime_count"}, inplace=True)
    df_state["hate_crime_norm"] = normalize(df_state["hate_crime_count"], reverse=True)

    # Save to file
    results_dir = "results/norm_targets"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_hate_crime_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Hate Crime Data Processed")
    return df_state

# Process all target metrics across years
def process_target_metrics_all_years(data_dir="data"):
    """
    Loops over all analytic_data_YEAR.csv files (2015–2025),
    computes weighted state averages for all TARGET_METRICS,
    and aggregates by state across years.
    """
    yearly_results = []

    for year in YEARS:
        file_path = os.path.join(data_dir, f"analytic_data{year}.csv")
        if not os.path.exists(file_path):
            print(f"Missing file for {year}, skipping.")
            continue

        df = pd.read_csv(file_path)
        df = df[df["state"] != "US"]

        state_year = pd.DataFrame({"state": df["state"].unique()})
        for var, reverse in TARGET_METRICS.items():
            if var not in df.columns:
                print(f"{var} not found in {year}, skipping.")
                continue
            metric_df = process_chr_metric(df, var, reverse=reverse)
            state_year = state_year.merge(metric_df, on="state", how="left")

        state_year["year"] = year
        yearly_results.append(state_year)

    # === Combine all years ===
    df_all = pd.concat(yearly_results, ignore_index=True)

    # Average each normalized metric by state
    agg_cols = [c for c in df_all.columns if c.endswith("_norm")]
    df_state = (
        df_all.groupby("state")[agg_cols]
        .mean()
        .reset_index()
    )
    df_state.rename(columns={"state": "State"}, inplace=True)

    results_dir = "results/norm_targets"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_target_metrics_by_state_all_years.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    return df_state

# Combine datasets
def build_target_index():
    """
    Combines all CHR target metrics (2015–2025) and hate crime
    into one weighted composite index by state.
    """
    chr_df = process_target_metrics_all_years()
    hate_df = process_hate_crime()

    # === Merge ===
    merged = chr_df.merge(hate_df, on="State", how="left")

    # === Normalize weights ===
    total_weight = sum(TARGET_METRIC_WEIGHTS.values()) + 1.0  # +1 for hate crime
    weights = {k: v / total_weight for k, v in TARGET_METRIC_WEIGHTS.items()}
    weights["hate_crime_norm"] = 1.0 / total_weight

    # === Compute composite ===
    merged["target_index"] = 0
    for col, w in weights.items():
        col_norm = col if col.endswith("_norm") else f"{col}_norm"
        if col_norm in merged.columns:
            merged["target_index"] += merged[col_norm].fillna(0) * w
        else:
            print(f"Missing {col_norm} in merged data.")

    merged = classify_percentiles(merged, column="target_index", n_classes=5)

    # === Save outputs ===
    os.makedirs("results/norm_targets", exist_ok=True)
    out_csv = "results/norm_targets/final_target_index_5_years.csv"
    out_txt = "results/norm_targets/final_target_index_5_years.txt"

    merged.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(merged[["State", "target_index", "Percentile_Class"]].to_string(index=False))

    print(f"Final Target Index (2020–2025) saved to {out_csv}")
    return merged

# Testing functions
build_target_index()