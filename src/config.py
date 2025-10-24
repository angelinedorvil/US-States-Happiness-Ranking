# src/config.py
from pathlib import Path

RANDOM_STATE = 42
RESULTS_DIR = Path("results/models")
PLOTS_DIR = Path("results/plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VALID_STATE_ABBRS = { "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
                      "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
                      "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY" }
CHR_METRICS_NAMES = {
    # Health outcomes
    "v001_rawvalue": "Premature death per 100,000",
    "v127_rawvalue": "Premature age-adjusted mortality",
    "v128_rawvalue": "Child mortality",
    "v129_rawvalue": "Infant mortality",
    "v135_rawvalue": "Injury deaths",
    "v161_rawvalue": "Suicides",
    "v015_rawvalue": "Homicides",
    "v039_rawvalue": "Motor vehicle crash deaths",
    "v148_rawvalue": "Firearm fatalities",

    # Environmental / infrastructure factors
    "v125_rawvalue": "Air pollution (PM2.5)",
    "v124_rawvalue": "Drinking water violations",
    "v179_rawvalue": "Access to parks",
    "v182_rawvalue": "Adverse climate events",
    "v166_rawvalue": "Broadband access",
    "v181_rawvalue": "Library access",
    "v156_rawvalue": "Traffic volume",

    # Behavioral / social outcomes
    "v134_rawvalue": "Alcohol-impaired driving deaths",
    "v139_rawvalue": "Food insecurity",
    "v083_rawvalue": "Limited access to healthy foods",
    "v133_rawvalue": "Food environment index",
    "v155_rawvalue": "Flu vaccination",

    # --- Predictor variables ---
    # Health behaviors
    "v009_rawvalue": "Adult smoking",
    "v011_rawvalue": "Adult obesity",
    "v049_rawvalue": "Excessive drinking",
    "v070_rawvalue": "Physical inactivity",
    "v045_rawvalue": "Sexually transmitted infections",
    "v014_rawvalue": "Teen births",
    "v138_rawvalue": "Drug overdose deaths",
    "v060_rawvalue": "Diabetes prevalence",
    "v061_rawvalue": "HIV prevalence",
    "v132_rawvalue": "Access to exercise opportunities",
    "v183_rawvalue": "Feelings of loneliness",
    "v143_rawvalue": "Insufficient sleep",

    # Clinical care
    "v004_rawvalue": "Ratio of population to primary care physicians",
    "v062_rawvalue": "Ratio of population to mental health providers",
    "v088_rawvalue": "Ratio of population to dentists",
    "v005_rawvalue": "Preventable hospital stays",
    "v085_rawvalue": "Uninsured",
    "v131_rawvalue": "Ratio of population to other primary care providers",

    # Social & economic factors
    "v024_rawvalue": "Children in poverty",
    "v044_rawvalue": "Income inequality",
    "v069_rawvalue": "Some college education",
    "v168_rawvalue": "High school completion",
    "v023_rawvalue": "Unemployment",
    "v140_rawvalue": "Social associations",
    "v171_rawvalue": "Child care cost burden",
    "v151_rawvalue": "Gender pay gap",
    "v063_rawvalue": "Median household income",
    "v170_rawvalue": "Living wage",
    "v172_rawvalue": "Child care centers",
    "v141_rawvalue": "Residential segregation (Black/White)",
    "v149_rawvalue": "Disconnected youth",
    "v184_rawvalue": "Lack of social and emotional support",
    "v177_rawvalue": "Voter turnout",

    # Physical environment & housing
    "v136_rawvalue": "Severe housing problems",
    "v153_rawvalue": "Home ownership",
    "v154_rawvalue": "Severe housing cost burden",
    "v067_rawvalue": "Driving alone to work",
    "v137_rawvalue": "Long commute",

    # Education & community context
    "v167_rawvalue": "School segregation",
    "v169_rawvalue": "School funding adequacy",

    # Health outcomes (as predictive indicators)
    "v036_rawvalue": "Poor physical health days",
    "v042_rawvalue": "Poor mental health days",
    "v144_rawvalue": "Frequent physical distress",
    "v145_rawvalue": "Frequent mental distress",
    "v147_rawvalue": "Life expectancy"
}