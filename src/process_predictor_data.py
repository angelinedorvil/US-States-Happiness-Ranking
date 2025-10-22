import pandas as pd
from sklearn.linear_model import LinearRegression
import us
import os
from utils.metrics_utils import process_chr_metric, normalize

# Years of data to consider
YEARS = range(2015, 2026)  # 2015–2025 inclusive

# === Predictor metrics from County Health Rankings ===
PREDICTOR_METRICS = {
    # --- Health behaviors ---
    "v009_rawvalue": True,    # Adult smoking (lower is better)
    "v011_rawvalue": True,    # Adult obesity (lower is better)
    "v049_rawvalue": True,    # Excessive drinking (lower is better)
    "v070_rawvalue": True,    # Physical inactivity (lower is better)
    "v045_rawvalue": True,    # Sexually transmitted infections (lower is better)
    "v014_rawvalue": True,    # Teen births (lower is better)
    "v138_rawvalue": True,    # Drug overdose deaths (lower is better)
    "v060_rawvalue": True,    # Diabetes prevalence (lower is better)
    "v061_rawvalue": True,    # HIV prevalence (lower is better)
    "v132_rawvalue": False,   # Access to exercise opportunities (higher is better)
    "v183_rawvalue": True,    # Feelings of loneliness (lower is better)
    "v143_rawvalue": True,    # Insufficient sleep (lower is better)

    # --- Clinical care ---
    "v004_rawvalue": True,    # Ratio of population to primary care physicians (lower is better)
    "v062_rawvalue": True,    # Ratio of population to mental health providers (lower is better)
    "v088_rawvalue": True,    # Ratio of population to dentists (lower is better)
    "v005_rawvalue": True,    # Preventable hospital stays (lower is better)
    "v085_rawvalue": True,    # Uninsured (lower is better)
    "v131_rawvalue": True,    # Ratio of population to other primary care providers (lower is better)

    # --- Social & economic factors ---
    "v024_rawvalue": True,    # Children in poverty (lower is better)
    "v044_rawvalue": True,    # Income inequality (lower is better)
    "v069_rawvalue": False,   # Some college education (higher is better)
    "v168_rawvalue": False,   # High school completion (higher is better)
    "v023_rawvalue": True,    # Unemployment (lower is better)
    "v140_rawvalue": False,   # Social associations (higher is better)
    "v171_rawvalue": True,    # Child care cost burden (lower is better)
    "v151_rawvalue": True,    # Gender pay gap (lower is better)
    "v063_rawvalue": False,   # Median household income (higher is better)
    "v170_rawvalue": False,   # Living wage (higher is better)
    "v172_rawvalue": False,   # Child care centers (higher is better)
    "v141_rawvalue": True,    # Residential segregation (Black/White) (lower is better)
    "v149_rawvalue": True,    # Disconnected youth (lower is better)
    "v184_rawvalue": True,    # Lack of social and emotional support (lower is better)
    "v177_rawvalue": False,   # Voter turnout (higher is better)

    # --- Physical environment & housing ---
    "v136_rawvalue": True,    # Severe housing problems (lower is better)
    "v153_rawvalue": False,   # Home ownership (higher is better)
    "v154_rawvalue": True,    # Severe housing cost burden (lower is better)
    "v067_rawvalue": True,    # Driving alone to work (lower is better)
    "v137_rawvalue": True,    # Long commute (lower is better)

    # --- Education & community context ---
    "v167_rawvalue": True,    # School segregation (lower is better)
    "v169_rawvalue": False,   # School funding adequacy (higher is better)

    # --- Health outcomes (as predictive indicators) ---
    "v036_rawvalue": True,    # Poor physical health days (lower is better)
    "v042_rawvalue": True,    # Poor mental health days (lower is better)
    "v144_rawvalue": True,    # Frequent physical distress (lower is better)
    "v145_rawvalue": True,    # Frequent mental distress (lower is better)
    "v147_rawvalue": False,   # Life expectancy (higher is better)
}

# === Predictor metric weights ===
PREDICTOR_METRIC_WEIGHTS = {
    # Health behaviors
    "v009_rawvalue": 1.0,
    "v011_rawvalue": 1.0,
    "v049_rawvalue": 1.0,
    "v070_rawvalue": 1.0,
    "v045_rawvalue": 1.0,
    "v014_rawvalue": 1.0,
    "v138_rawvalue": 1.0,
    "v060_rawvalue": 1.0,
    "v061_rawvalue": 1.0,
    "v132_rawvalue": 1.0,
    "v183_rawvalue": 0.8,
    "v143_rawvalue": 1.0,

    # Clinical care
    "v004_rawvalue": 1.0,
    "v062_rawvalue": 1.0,
    "v088_rawvalue": 1.0,
    "v005_rawvalue": 1.0,
    "v085_rawvalue": 1.0,
    "v131_rawvalue": 1.0,

    # Social & economic
    "v024_rawvalue": 1.0,
    "v044_rawvalue": 1.0,
    "v069_rawvalue": 1.0,
    "v168_rawvalue": 1.0,
    "v023_rawvalue": 1.0,
    "v140_rawvalue": 1.0,
    "v171_rawvalue": 1.0,
    "v151_rawvalue": 1.0,
    "v063_rawvalue": 1.0,
    "v170_rawvalue": 1.0,
    "v172_rawvalue": 1.0,
    "v141_rawvalue": 1.0,
    "v149_rawvalue": 1.0,
    "v184_rawvalue": 1.0,
    "v177_rawvalue": 1.0,

    # Physical environment & housing
    "v136_rawvalue": 1.0,
    "v153_rawvalue": 1.0,
    "v154_rawvalue": 1.0,
    "v067_rawvalue": 1.0,
    "v137_rawvalue": 1.0,

    # Education & community context
    "v167_rawvalue": 1.0,
    "v169_rawvalue": 1.0,

    # Health outcomes (predictive indicators)
    "v036_rawvalue": 1.0,
    "v042_rawvalue": 1.0,
    "v144_rawvalue": 1.0,
    "v145_rawvalue": 1.0,
    "v147_rawvalue": 1.0,
}


# Process all predictor metrics across years
def process_predictor_metrics_all_years(data_dir="data"):
    """
    Loops over all analytic_data_YEAR.csv files (2015–2025),
    computes weighted state averages for all PREDICTOR_METRICS,
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
        for var, reverse in PREDICTOR_METRICS.items():
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

    results_dir = "results/norm_predictors"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_predictors_metrics_by_state_all_years.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    return df_state

# Combine datasets
def build_predictor_index():
    """
    Combines all CHR target metrics (2015–2025) and hate crime
    into one weighted composite index by state.
    """
    chr_df = process_predictor_metrics_all_years()

    # === Normalize weights ===
    total_weight = sum(PREDICTOR_METRIC_WEIGHTS.values()) 
    weights = {k: v / total_weight for k, v in PREDICTOR_METRIC_WEIGHTS.items()}

    # === Compute composite ===
    chr_df["predictor_index"] = 0
    for col, w in weights.items():
        col_norm = col if col.endswith("_norm") else f"{col}_norm"
        if col_norm in chr_df.columns:
            chr_df["predictor_index"] += chr_df[col_norm].fillna(0) * w
        else:
            print(f"Missing {col_norm} in chr_df data.")

    # === Save outputs ===
    os.makedirs("results/norm_predictors", exist_ok=True)
    out_csv = "results/norm_predictors/final_predictor_index_all_years.csv"
    out_txt = "results/norm_predictors/final_predictor_index_all_years.txt"

    chr_df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(chr_df[["State", "predictor_index"]].to_string(index=False))

    print(f"Final Predicotr Index (2015–2025) saved to {out_csv}")
    return chr_df

# Testing functions
build_predictor_index()