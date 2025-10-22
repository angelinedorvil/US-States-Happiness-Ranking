import pandas as pd
import numpy as np
import warnings


# Optional normalization helper
def normalize(series, reverse=False):
    norm = (series - series.min()) / (series.max() - series.min())
    if reverse:
        norm = 1 - norm
    return norm * 100

def process_chr_metric(df, var, reverse=False, year=None):
    """
    Compute population-weighted state average for a CHR metric.
    reverse=True if higher values are worse.
    Includes detailed warning capture for divide-by-zero or all-NaN issues.
    """

    df = df[df["state"].notna() & (df["state"] != "US")].copy()
    df[var] = pd.to_numeric(df[var], errors="coerce")

    denom_col = var.replace("_rawvalue", "_denominator")
    pop_col = "v051_rawvalue"

    has_denom = denom_col in df.columns and df[denom_col].notna().any() and df[denom_col].sum() > 0
    has_pop = pop_col in df.columns and df[pop_col].notna().any() and df[pop_col].sum() > 0

    if has_denom:
        weight_col = denom_col
        method = f"{denom_col}"
    elif has_pop:
        weight_col = pop_col
        method = f"{pop_col} (population)"
    else:
        weight_col = None
        method = "simple mean"

    bad_groups = []  # track any states that fail weighting

    def safe_weighted_mean(g):
        # Defensive checks inside each group
        weights = pd.to_numeric(g[weight_col], errors="coerce")
        values = pd.to_numeric(g[var], errors="coerce")

        total_w = np.nansum(weights)
        if total_w == 0 or np.isnan(total_w):
            bad_groups.append(g["state"].iloc[0])
            return np.nan

        valid_mask = (~np.isnan(values)) & (~np.isnan(weights))
        if not valid_mask.any():
            bad_groups.append(g["state"].iloc[0])
            return np.nan

        return np.nansum(values[valid_mask] * weights[valid_mask]) / total_w

    if weight_col:
        # Silence RuntimeWarnings, we’ll catch them ourselves
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            state_df = (
                df.groupby("state", as_index=False)
                  .apply(lambda g: pd.Series({var: safe_weighted_mean(g)}))
                  .reset_index(drop=True)
            )
    else:
        state_df = df.groupby("state", as_index=False)[var].mean(numeric_only=True)

    # Normalize result
    state_df[var + "_norm"] = normalize(state_df[var], reverse=reverse)

    # Print summary
    print(f"Processed {var} ({method}) — reverse={reverse}")
    if bad_groups:
        print(f"  Warning: {var} had invalid or zero weights for {len(bad_groups)} states")
        print(f"  States affected: {', '.join(sorted(set(bad_groups)))}")
        if year:
            print(f"   Year: {year}")

    return state_df
