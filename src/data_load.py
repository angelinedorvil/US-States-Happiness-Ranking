import pandas as pd
from sklearn.linear_model import LinearRegression
import us  # pip install us

# === Load CDC BRFSS mental health data (use all questions, avg across years) ===
cdc_path = "data/Behavioral_Risk_Factor_Surveillance_System_(BRFSS)_-_Mental_Health_Indicators_20251019.csv"
cdc = pd.read_csv(cdc_path)

# Keep only total demographic values (avoid gender/race splits)
cdc = cdc[cdc['Demographics_Value'] == 'Total'].copy()

# Convert Percent to numeric
cdc['Percent'] = pd.to_numeric(cdc['Percent'], errors='coerce')

# Group by state and year → average across all questions
cdc_yearly = (
    cdc.groupby(['Area', 'Area_abbr', 'Year'], as_index=False)['Percent']
    .mean()
    .rename(columns={'Area': 'state', 'Area_abbr': 'state_abbr', 'Percent': 'yearly_avg'})
)

# Then average across years (e.g. 2022 + 2023)
cdc = (
    cdc_yearly.groupby(['state', 'state_abbr'], as_index=False)['yearly_avg']
    .mean()
    .rename(columns={'yearly_avg': 'mental_health_index'})
)

# === Load Life Expectancy ===
lifeexp_path = "data/U.S._State_Life_Expectancy_by_Sex,_2019_20251019.csv"
lifeexp = pd.read_csv(lifeexp_path)
lifeexp = lifeexp[lifeexp['Sex'].str.lower() == 'total'][['State', 'LEB']]
lifeexp.rename(columns={'State': 'state', 'LEB': 'life_expectancy'}, inplace=True)
lifeexp['life_expectancy'] = pd.to_numeric(lifeexp['life_expectancy'], errors='coerce')

# ➕ Add state abbreviations to lifeexp for matching
lifeexp['state_abbr'] = lifeexp['state'].map(lambda s: us.states.lookup(s).abbr if us.states.lookup(s) else None)

# === Load poverty ===
ers = pd.read_csv("data/Poverty2023.csv")
ers = ers[ers['Stabr'] != 'US']
ers = ers[ers['Attribute'] == 'PCTPOVALL_2023'].copy()
ers['Value'] = pd.to_numeric(ers['Value'], errors='coerce')

state_pct = (
    ers.groupby('Stabr', as_index=False)['Value']
    .mean()
    .rename(columns={'Stabr': 'state_abbr', 'Value': 'poverty_rate_2023'})
)

# === Combine all three ===
# lifeexp + poverty (both have state_abbr now)
final_df = pd.merge(lifeexp, state_pct, on='state_abbr', how='inner')

# Add CDC mental health index
final_df = pd.merge(final_df, cdc, on='state_abbr', how='left')

# Fix any duplicate columns from merges
final_df.rename(columns={'state_x': 'state'}, inplace=True)
final_df.drop(columns=['state_y'], inplace=True, errors='ignore')

print("Before imputation — missing mental health index:",
      final_df['mental_health_index'].isna().sum())

# === Impute missing mental health index ===
train = final_df.dropna(subset=['mental_health_index'])
X = train[['life_expectancy', 'poverty_rate_2023']]
y = train['mental_health_index']

model = LinearRegression().fit(X, y)

missing_states = final_df[final_df['mental_health_index'].isna()]
if not missing_states.empty:
    final_df.loc[missing_states.index, 'mental_health_index'] = model.predict(
        missing_states[['life_expectancy', 'poverty_rate_2023']]
    )

print("After imputation — remaining missing values:",
      final_df['mental_health_index'].isna().sum())

print("Final dataset shape:", final_df.shape)
print(final_df[['state', 'state_abbr', 'mental_health_index', 'life_expectancy', 'poverty_rate_2023']].head())

# Save final dataset
final_df.to_csv("data/final_dataset.csv", index=False)
print("Saved final dataset to data/final_dataset.csv")
