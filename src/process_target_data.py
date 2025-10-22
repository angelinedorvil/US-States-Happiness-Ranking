import pandas as pd
import numpy as np
import os
import glob


# Optional normalization helper
def normalize(series, reverse=False):
    norm = (series - series.min()) / (series.max() - series.min())
    if reverse:
        norm = 1 - norm
    return norm * 100

# --- State abbreviation ↔ full name mappings ---
STATE_MAP = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
}

# Standardize state names function
def standardize_state_names(df):
    """
    Ensures that state names are standardized to full names.
    Works whether states are provided as abbreviations or full names.
    """
    df = df.copy()
    df["State"] = df["State"].str.strip()
    df["State"] = df["State"].replace(STATE_MAP)  # Convert abbreviations to full names
    df["State"] = df["State"].str.title()  # Capitalize properly (Alabama, not ALABAMA)
    return df

# Classify states based on happiness index
def classify_percentiles(df, column="env_safety_index", n_classes=5):
    """
    Adds a percentile-based class column.
    n_classes = 3 (tertile), 4 (quartile), 5 (quintile), etc.
    """
    df = df.copy()
    labels = [f"Tier_{i}" for i in range(1, n_classes + 1)]
    df["Percentile_Class"] = pd.qcut(df[column], q=n_classes, labels=labels)
    return df

# Process air pollution data
def process_pm25(path="data/target_data/PM2.5_highest_annual_average_concentration_states_2018_to_2020.csv"):
    df = pd.read_csv(path)
    df = (
        df.groupby("State", as_index=False)["Value"]
          .mean()
          .rename(columns={"Value": "pm25_avg"})
    )
    df["pm25_norm"] = normalize(df["pm25_avg"], reverse=True)  # lower = better

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_air_pollution_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    print("PM2.5 Data Processed")
    return df

# Process air quality index data
def process_aqi(
    paths=[
        "data/target_data/annual_aqi_by_county_2023.csv",
        "data/target_data/annual_aqi_by_county_2024.csv",
        "data/target_data/annual_aqi_by_county_2025.csv"
    ]
):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["percent_good"] = (df["Good Days"] / df["Days with AQI"]) * 100
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    df_state = (
        df_all.groupby("State", as_index=False)["percent_good"]
              .mean()
              .rename(columns={"percent_good": "aqi_gooddays"})
    )
    df_state["aqi_norm"] = normalize(df_state["aqi_gooddays"], reverse=False)  # higher = better'

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_aqi_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("AQI Data Processed")
    return df_state

# Process traffic fatality data
def process_traffic_fatalities(
    paths=[
        "data/target_data/accident_2020.csv",
        "data/target_data/accident_2021.csv",
        "data/target_data/accident_2022.csv"
    ]
):
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="latin1")
        frames.append(df[["STATENAME", "PERSONS"]])
    df_all = pd.concat(frames, ignore_index=True)
    df_state = df_all.groupby("STATENAME", as_index=False)["PERSONS"].sum()
    df_state.rename(columns={"STATENAME": "State", "PERSONS": "fatalities"}, inplace=True)
    df_state["fatalities_norm"] = normalize(df_state["fatalities"], reverse=True)

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_traffic_fatalities_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Traffic Fatalities Data Processed")
    return df_state

# Process violent crime data
def process_violent_crime():
    """
    Processes all cleaned FBI crime CSVs from:
        data/target_data/Table_5_Crime_in_the_United_States_by_State_*.csv

    Combines all years, computes weighted composite score (equal weights by default),
    and normalizes it so that lower = better.
    """

    folder = os.path.join("data", "target_data")
    pattern = os.path.join(folder, "Table_5_Crime_in_the_United_States_by_State_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No FBI crime CSV files found in {folder}")

    # print(f"📂 Found {len(files)} FBI crime CSV file(s):")
    # for f in files:
    #     print("   •", os.path.basename(f))

    # === Define weights (equal by default; editable later) ===
    weights = {
        "Violent_crime": 1.0,
        "Murder": 1.0,
        "Rape": 1.0,
        "Robbery": 1.0,
        "Aggravated_assault": 1.0,
        "Property_crime": 1.0,
        "Burglary": 1.0,
        "Larceny_theft": 1.0,
        "Motor_vehicle_theft": 1.0,
    }
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # === Read and clean all files ===
    frames = []
    for p in files:
        df = pd.read_csv(p, quotechar='"', thousands=',')
        keep_cols = ["State"] + list(weights.keys())
        df = df[keep_cols].copy()

        # clean numbers
        for col in weights.keys():
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)

        frames.append(df)

    # === Combine across years ===
    combined = pd.concat(frames, ignore_index=True)
    df_avg = combined.groupby("State", as_index=False).mean(numeric_only=True)

    # === Compute weighted composite ===
    df_avg["crime_composite"] = sum(df_avg[col] * w for col, w in weights.items())

    # === Normalize (lower = better) ===
    df_avg["crime_norm"] = normalize(df_avg["crime_composite"], reverse=True)

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_violent_crime_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_avg[["State", "crime_composite", "crime_norm"]].to_string(index=False))

    print("Violent Crime Data Processed")
    return df_avg

# Process hate crime data
def process_hate_crime(path="data/target_data/hate_crime.csv"):
    df = pd.read_csv(path)
    df = df[df["data_year"] >= 2022]
    df_state = df.groupby("state_name", as_index=False)["incident_id"].count()
    df_state.rename(columns={"state_name": "State", "incident_id": "hate_crime_count"}, inplace=True)
    df_state["hate_crime_norm"] = normalize(df_state["hate_crime_count"], reverse=True)

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_hate_crime_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Hate Crime Data Processed")
    return df_state

# Process drinking water violations data
def process_drinking_water(path="data/target_data/SDWA_PN_VIOLATION_ASSOC.csv"):
    """
    Processes Safe Drinking Water Act (SDWA) violation data.
    Extracts state abbreviation from PWSID.
    Excludes EPA region codes (tribal/non-state systems).
    Counts unique violations per state since 2023.
    """

    df = pd.read_csv(path, dtype=str)
    df["PWSID"] = df["PWSID"].str.strip().str.upper()
    df["State"] = df["PWSID"].str[:2]

    # --- Filter to valid US state abbreviations ---
    valid_states = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA",
        "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM",
        "NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA",
        "WV","WI","WY"
    }
    df = df[df["State"].isin(valid_states)]

    # --- Filter valid reported violations ---
    df = df[df["LAST_REPORTED_DATE"].notna()]
    df["year"] = df["LAST_REPORTED_DATE"].str.extract(r"(\d{4})").astype(float)
    df = df[df["year"] >= 2023]

    # --- Aggregate per state ---
    df_state = (
        df.groupby("State", as_index=False)["PN_VIOLATION_ID"]
          .nunique()
          .rename(columns={"PN_VIOLATION_ID": "water_violations"})
    )

    # --- Normalize (lower = better) ---
    df_state["water_norm"] = normalize(df_state["water_violations"], reverse=True)

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_drinking_water_violations_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Drinking Water Violations Processed")

    return df_state

# Process natural hazard data
def process_natural_hazard(
    paths=[
        "data/target_data/NRI_Table_Counties_2020.csv",
        "data/target_data/NRI_Table_Counties_2021.csv",
        "data/target_data/NRI_Table_Counties_2023.csv"
    ]
):
    """
    Processes FEMA National Risk Index (NRI) data across multiple years.
    - Computes average RISK_SCORE per state across available datasets.
    - If a state appears only in some years (e.g., territories), uses available data.
    - Lower RISK_SCORE = better (normalized reversed).
    """

    frames = []

    for p in paths:
        df = pd.read_csv(p)
        # Average within file by state
        df_state = (
            df.groupby("STATE", as_index=False)["RISK_SCORE"]
              .mean()
              .rename(columns={"STATE": "State", "RISK_SCORE": "hazard_score"})
        )
        frames.append(df_state)

    # Combine and average across available years
    df_all = pd.concat(frames, ignore_index=True)
    df_avg = df_all.groupby("State", as_index=False)["hazard_score"].mean()  # ✅ default skipna=True

    # Normalize (lower = better)
    df_avg["hazard_norm"] = normalize(df_avg["hazard_score"], reverse=True)

    # Save to file
    results_dir = "results" 
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_natural_hazard_risk_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_avg.to_string(index=False))

    print("Natural Hazard Risk Data Processed")
    return df_avg

# Process broadband access data
def process_broadband(path="data/target_data/bdc_comparison_fixed_state_total_all_terrestrial_r_100_20_D24.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"geography_desc_full": "State", "percent_coverage": "broadband_coverage"})
    df["broadband_norm"] = normalize(df["broadband_coverage"], reverse=False)

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_broadband_access_by_state.txt")   
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df[["State", "broadband_coverage", "broadband_norm"]].to_string(index=False))

    print("Broadband Access Data Processed")
    return df[["State", "broadband_coverage", "broadband_norm"]]

# Process food insecurity data
def process_food_access(path="data/target_data/FoodAccessResearchAtlasData2019.csv"):
    df = pd.read_csv(path)
    df_state = (
        df.groupby("State", as_index=False)["LILATracts_1And10"]
          .mean()
          .rename(columns={"LILATracts_1And10": "food_low_access"})
    )
    df_state["food_norm"] = normalize(df_state["food_low_access"], reverse=True)

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_food_insecurity_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Food Insecurity Data Processed")
    return df_state

# Process park data
def process_parks(path="data/target_data/analytic_data2025_v2.csv"):
    """
    Aggregates Access to Parks (v179_rawvalue) to the state level
    using a population-weighted average.
    Higher = better.
    """

    df = pd.read_csv(path)
    df = df[df["state"] != "US"]

    # Compute weighted mean by state
    df["weighted_value"] = df["v179_rawvalue"] * df["v179_denominator"]
    df_state = (
        df.groupby("state", as_index=False)
          .apply(lambda g: pd.Series({
              "parks_access": g["weighted_value"].sum() / g["v179_denominator"].sum()
          }))
          .reset_index(drop=True)
    )

    df_state.rename(columns={"state": "State"}, inplace=True)
    df_state["parks_norm"] = normalize(df_state["parks_access"], reverse=False)

    # Save to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "norm_parks_access_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Parks Access Data Processed (population-weighted)")
    return df_state

# Combine datasets
def build_target_index():
    """
    Combines all processed target datasets (environment & safety metrics)
    into one weighted composite index by state.

    You can edit the weights dictionary below to rebalance contributions.
    All weights automatically normalize to sum = 1.
    """

    # === Step 1: collect all processed dataframes ===
    dfs = {
        "pm25": process_pm25(),
        "aqi": process_aqi(),
        "traffic": process_traffic_fatalities(),
        "crime": process_violent_crime(),
        "hate": process_hate_crime(),
        "water": process_drinking_water(),
        "hazard": process_natural_hazard(),
        "broadband": process_broadband(),
        "food": process_food_access(),
        "parks": process_parks(),
    }

    # === Step 2: merge all dataframes on 'State' ===
    merged = None
    for name, df in dfs.items():
        df = standardize_state_names(df)
        dfs[name] = df
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="State", how="outer")

    # === Step 3: define weights (editable) ===
    # Set all equal for now; you can tune later.
    weights = {
        "pm25_norm": 1.0,
        "aqi_norm": 1.0,
        "fatalities_norm": 1.0,
        "crime_norm": 1.0,
        "hate_crime_norm": 1.0,
        "water_norm": 1.0,
        "hazard_norm": 1.0,
        "broadband_norm": 1.0,
        "food_norm": 1.0,
        "parks_norm": 1.0,
    }

    # normalize weights to sum = 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # === Step 4: compute weighted composite index ===
    merged["env_safety_index"] = 0
    for col, w in weights.items():
        if col in merged.columns:
            merged["env_safety_index"] += merged[col].fillna(0) * w
        else:
            print(f"⚠️ Warning: column {col} not found in merged dataset")

    merged = classify_percentiles(merged, column="env_safety_index", n_classes=5)

    # === Step 5: save results ===
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "env_safety_index_by_state.csv")
    out_txt = os.path.join(results_dir, "env_safety_index_by_state.txt")

    merged[["State", "env_safety_index", "Percentile_Class"]].to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(merged[["State", "env_safety_index", "Percentile_Class"]].to_string(index=False))

    print(f"\n✅ Environment & Safety Index saved to:\n  {out_csv}\n  {out_txt}")
    return merged


# Testing functions
# process_pm25()
# process_aqi()
# process_traffic_fatalities()
# process_violent_crime()
# process_hate_crime()
# process_drinking_water()
# process_natural_hazard()
# process_broadband()
# process_food_access()
# process_parks()
build_target_index()