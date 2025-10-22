# src/config.py
from pathlib import Path

RANDOM_STATE = 42
RESULTS_DIR = Path("results/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VALID_STATE_ABBRS = { "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
                      "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
                      "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY" }