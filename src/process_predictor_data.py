import pandas as pd
from sklearn.linear_model import LinearRegression
import us
import os


# === Helper ===
def normalize(series, reverse=False):
    norm = (series - series.min()) / (series.max() - series.min())
    if reverse:
        norm = 1 - norm
    return norm * 100

# Keep only states + DC
VALID_STATE_ABBRS = {s.abbr for s in us.states.STATES} | {"DC"}
def filter_us_states(df, col="State_abbr"):
    """Keep only 50 states + D.C."""
    return df[df[col].isin(VALID_STATE_ABBRS)].copy()

# === Helper: Percentile Classification ===
def classify_percentiles(df, column="env_safety_index", n_classes=5):
    """
    Adds a percentile-based class column.
    n_classes = 3 (tertile), 4 (quartile), 5 (quintile), etc.
    """
    df = df.copy()
    labels = [f"Tier_{i}" for i in range(1, n_classes + 1)]
    df["Percentile_Class"] = pd.qcut(df[column], q=n_classes, labels=labels)
    return df

# === 1️⃣ CDC Mental Health Data ===
def process_mental_health(path="data/predictor_data/Behavioral_Risk_Factor_Surveillance_System_(BRFSS)_-_Mental_Health_Indicators_20251019.csv"):
    df = pd.read_csv(path)
    df = df[df['Demographics_Value'] == 'Total'].copy()
    df['Percent'] = pd.to_numeric(df['Percent'], errors='coerce')

    # Define polarity: True = reverse (higher = worse)
    QUESTION_POLARITY = {
        "poor physical or mental health keep you from doing": True,
        "depressive disorder": True,
        "socially isolated": True,
        "lonely": True,
        "social and emotional support": False,
        "satisfied with your life": False,
        "mental health not good": True,
        "physical health not good": True,
    }

    # Tag rows for reversal
    def detect_reverse(question):
        q_lower = question.lower()
        for phrase, reverse_flag in QUESTION_POLARITY.items():
            if phrase in q_lower:
                return reverse_flag
        return False  # default: not reversed

    df['reverse'] = df['Question'].apply(detect_reverse)

    # Normalize each question individually so "higher = better"
    df['norm'] = 0.0
    for question_text, group in df.groupby('Question'):
        reverse_flag = detect_reverse(question_text)
        normed = normalize(group['Percent'], reverse=reverse_flag)
        df.loc[group.index, 'norm'] = normed

    # Average normalized values across all questions and years for each state
    df_state = (
        df.groupby(['Area', 'Area_abbr'], as_index=False)['norm']
          .mean()
          .rename(columns={'Area': 'State', 'Area_abbr': 'State_abbr', 'norm': 'mental_health_norm'})
    )

    # Also keep raw mean of Percent for reference
    df_raw = (
        df.groupby(['Area', 'Area_abbr'], as_index=False)['Percent']
          .mean()
          .rename(columns={'Area': 'State', 'Area_abbr': 'State_abbr', 'Percent': 'mental_health_index'})
    )

    # ✅ Merge safely (now both sides have same column names)
    df_state = df_state.merge(df_raw, on=['State', 'State_abbr'], how='left')

    # Save snapshot
    os.makedirs("results/norm_predictors", exist_ok=True)
    out_txt = os.path.join("results/norm_predictors", "norm_mental_health_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Mental Health Data Processed ✅")
    return df_state

# === 2️⃣ Life Expectancy ===
def process_life_expectancy(path="data/predictor_data/U.S._State_Life_Expectancy_by_Sex,_2019_20251019.csv"):
    df = pd.read_csv(path)

    # Keep only Total rows (ignore Male/Female)
    df = df[df['Sex'].str.lower() == 'total'][['State', 'LEB']]

    # Drop 'United States' national average
    df = df[df['State'].str.lower() != 'united states']

    # Clean column names
    df.rename(columns={'LEB': 'life_expectancy'}, inplace=True)
    df['life_expectancy'] = pd.to_numeric(df['life_expectancy'], errors='coerce')

    # Add abbreviations (handle D.C. manually)
    df['State_abbr'] = df['State'].map(lambda s: us.states.lookup(s).abbr if us.states.lookup(s) else None)
    df.loc[df['State'].str.lower().str.contains("district of columbia"), 'State_abbr'] = "DC"

    # Normalize (higher = better)
    df["life_expectancy_norm"] = normalize(df["life_expectancy"], reverse=False)

    # Save snapshot for verification
    os.makedirs("results/norm_predictors", exist_ok=True)
    out_txt = os.path.join("results/norm_predictors", "norm_life_expectancy_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    print("Life Expectancy Data Processed ✅")
    #print("Missing abbreviations:", df['State_abbr'].isna().sum())
    return df

# === 3️⃣ Poverty Data ===
def process_poverty(path="data/predictor_data/Poverty2023.csv"):
    df = pd.read_csv(path)
    df = df[df['Stabr'] != 'US']
    df = df[df['Attribute'] == 'PCTPOVALL_2023'].copy()
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    df_state = (
        df.groupby('Stabr', as_index=False)['Value']
          .mean()
          .rename(columns={'Stabr': 'State_abbr', 'Value': 'poverty_rate_2023'})
    )

    # Normalize (lower = better)
    df_state["poverty_norm"] = normalize(df_state["poverty_rate_2023"], reverse=True)

    # Save snapshot
    os.makedirs("results/norm_predictors", exist_ok=True)
    out_txt = os.path.join("results/norm_predictors", "norm_poverty_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Poverty Data Processed ✅")
    return df_state

# === 4️⃣ Education Data ===
def process_education(path="data/predictor_data/Education2023.csv"):
    """
    Processes county-level education attainment data (2019–2023).
    Creates a scaled Education Index per state:
      - Not HS graduate: lower is better, weight = 1
      - HS graduate: higher is better, weight = 2
      - Some college/associate: higher is better, weight = 3
      - Bachelor's or higher: higher is better, weight = 4
    """

    df = pd.read_csv(path, encoding="latin1")

    # Keep only 2019–23 indicators
    df = df[df["Attribute"].str.contains("2019-23", na=False)].copy()

    # Clean numeric column
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Define mapping: keyword → (reverse?, scale)
    EDU_MAP = {
        "not high school graduate": (True, 1),
        "high school graduate": (False, 2),
        "some college or associate": (False, 3),
        "bachelor": (False, 4),
    }

    # Tag each row
    df["reverse"] = df["Attribute"].apply(
        lambda x: next((v[0] for k, v in EDU_MAP.items() if k in x.lower()), None)
    )
    df["scale"] = df["Attribute"].apply(
        lambda x: next((v[1] for k, v in EDU_MAP.items() if k in x.lower()), None)
    )

    # Drop rows without valid matches
    df = df.dropna(subset=["reverse", "scale"])

    # Normalize each group (per Attribute)
    df["norm"] = 0.0
    for attr, group in df.groupby("Attribute"):
        reverse = group["reverse"].iloc[0]
        normed = normalize(group["Value"], reverse=reverse)
        df.loc[group.index, "norm"] = normed

    # Apply scaling weight
    df["scaled_score"] = df["norm"] * df["scale"]

    # Aggregate by state (mean of all counties)
    df_state = (
        df.groupby("State", as_index=False)["scaled_score"]
          .mean()
          .rename(columns={"scaled_score": "education_index_raw"})
    )

    # Normalize final index so all states = 0–100
    df_state["education_index_norm"] = normalize(df_state["education_index_raw"], reverse=False)

    # Add abbreviations (for consistency with other predictors)
    df_state["State_abbr"] = df_state["State"].str.upper()

    # Save snapshot
    os.makedirs("results/norm_predictors", exist_ok=True)
    out_txt = os.path.join("results/norm_predictors", "norm_education_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Education Data Processed ✅")
    return df_state

# === 5 Unemployment Data ===
def process_unemployment(path="data/predictor_data/Unemployment2023.csv"):
    df = pd.read_csv(path)
    df = df[df["Attribute"].str.contains("Unemployment_rate_", na=False)]
    df["Year"] = df["Attribute"].str.extract(r"(\d{4})").astype(int)
    df = df[df["Year"].between(2021, 2023)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    df_state = (
        df.groupby("State", as_index=False)["Value"]
          .mean()
          .rename(columns={"Value": "unemployment_rate_avg"})
    )

    df_state["unemployment_norm"] = normalize(df_state["unemployment_rate_avg"], reverse=True)
    df_state["State_abbr"] = df_state["State"].str.upper()

    os.makedirs("results/norm_predictors", exist_ok=True)
    out_txt = os.path.join("results/norm_predictors", "norm_unemployment_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Unemployment Data Processed ✅")
    return df_state

# === 6 Median Income Data ===
def process_income(path="data/predictor_data/Unemployment2023.csv"):
    df = pd.read_csv(path)
    df = df[df["Attribute"].str.contains("Median_Household_Income_", na=False)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Keep most recent (2022) if multiple years exist
    df["Year"] = df["Attribute"].str.extract(r"(\d{4})").astype(int)
    df = df[df["Year"] == 2022]

    df_state = (
        df.groupby("State", as_index=False)["Value"]
          .mean()
          .rename(columns={"Value": "median_income"})
    )

    df_state["income_norm"] = normalize(df_state["median_income"], reverse=False)
    df_state["State_abbr"] = df_state["State"].str.upper()

    os.makedirs("results/norm_predictors", exist_ok=True)
    out_txt = os.path.join("results/norm_predictors", "norm_income_by_state.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df_state.to_string(index=False))

    print("Household Income Data Processed ✅")
    return df_state

# === 7 Combine all predictor datasets ===
def build_predictor_dataset():
    """
    Combines all processed predictor datasets (mental health, life expectancy, poverty)
    into one weighted composite index by state.

    You can edit the weights dictionary below to rebalance contributions.
    All weights automatically normalize to sum = 1.
    """

    # === Step 1: collect and process data ===
    mental = process_mental_health()
    life = process_life_expectancy()
    poverty = process_poverty()
    education = process_education()
    unemployment = process_unemployment()
    income = process_income()

    # === Step 2: merge all datasets on 'State_abbr' ===
    df = life.merge(poverty, on="State_abbr", how="outer")
    df = df.merge(mental, on="State_abbr", how="outer", suffixes=("", "_mh"))
    df = df.merge(education, on="State_abbr", how="outer", suffixes=("", "_edu"))
    df = df.merge(unemployment, on="State_abbr", how="outer", suffixes=("", "_unemp"))
    df = df.merge(income, on="State_abbr", how="outer", suffixes=("", "_inc"))

    # Fix state names if duplicated
    if "State_mh" in df.columns:
        df["State"] = df["State"].fillna(df["State_mh"])
        df.drop(columns=["State_mh"], inplace=True)

    # ✅ Keep only 50 states + D.C.
    df = filter_us_states(df)

    # === Step 3: handle missing mental health values ===
    train = df.dropna(subset=["mental_health_index"])
    X = train[["life_expectancy", "poverty_rate_2023"]].dropna()
    y = train.loc[X.index, "mental_health_index"]

    if len(X) > 0:
        model = LinearRegression().fit(X, y)
        missing = df["mental_health_index"].isna()
        if missing.sum() > 0:
            X_missing = df.loc[missing, ["life_expectancy", "poverty_rate_2023"]].dropna()
            if not X_missing.empty:
                df.loc[X_missing.index, "mental_health_index"] = model.predict(X_missing)
    else:
        print("⚠️ No valid rows for regression — skipping imputation.")

    # === Step 4: define weights (equal for now; editable later) ===
    weights = {
        "mental_health_norm": 0.5,  # can be lowered later (e.g., 0.5)
        "life_expectancy_norm": 1.0,
        "poverty_norm": 1.0,
        "education_index_norm": 1.0,
        "unemployment_norm": 1.0,
        "income_norm": 1.0,
    }

    # Normalize weights to sum = 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # === Step 5: compute weighted predictor index ===
    df["predictor_index"] = 0
    for col, w in weights.items():
        if col in df.columns:
            df["predictor_index"] += df[col].fillna(0) * w
        else:
            print(f"⚠️ Warning: column {col} not found in merged dataset")

    # Optional percentile classification (e.g., quintiles)
    df = classify_percentiles(df, column="predictor_index", n_classes=5)

    # === Step 6: save merged output ===
    os.makedirs("results/norm_predictors", exist_ok=True)
    out_csv = os.path.join("results/norm_predictors", "predictor_dataset_combined.csv")
    out_txt = os.path.join("results/norm_predictors", "predictor_dataset_combined.txt")
    df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(df[["State", "predictor_index", "Percentile_Class"]].to_string(index=False))

    print(f"\n✅ Predictor dataset saved to:\n  {out_csv}\n  {out_txt}\n")
    return df

# === TESTING ===
if __name__ == "__main__":
    build_predictor_dataset()
    # process_life_expectancy()
