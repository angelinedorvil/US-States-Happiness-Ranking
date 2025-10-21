# src/utils_composite.py
import numpy as np
import pandas as pd

def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

def minmax_0_100(s):
    return 100 * (s - s.min()) / (s.max() - s.min())

def build_composite(df, specs, weight_mode="equal", scale="z"):
    """
    df: state-level dataframe with indicator columns.
    specs: list of dicts: [{"col":"obesity", "higher_is_better": False, "weight":1.0}, ...]
    weight_mode: "equal" or "spec"
    scale: "z" or "minmax"
    returns (score, parts_df)
    """
    parts = []
    weights = []
    for item in specs:
        col = item["col"]
        hib = item.get("higher_is_better", True)
        w   = item.get("weight", 1.0)
        x = df[col].astype(float)

        # choose scaler
        s = zscore(x) if scale == "z" else minmax_0_100(x)

        # reverse if higher_is_better is False
        s = -s if scale == "z" and not hib else (100 - s if scale == "minmax" and not hib else s)

        parts.append(s)
        weights.append(w)

    parts_df = pd.concat(parts, axis=1)
    parts_df.columns = [item["col"] for item in specs]

    if weight_mode == "equal":
        score = parts_df.mean(axis=1)
    else:
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        score = (parts_df * w).sum(axis=1)

    return score, parts_df
