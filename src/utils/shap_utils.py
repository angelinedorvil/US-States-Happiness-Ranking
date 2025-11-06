import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def _ensure_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    # Force everything to float—no strings, no objects.
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Drop columns that are entirely NaN (shouldn’t happen, but safe)
    X = X.dropna(axis=1, how="all")
    # If any rows have NaNs, fill with column means (stable for SHAP)
    if X.isna().any().any():
        X = X.fillna(X.mean(numeric_only=True))
    # Ensure dtype float
    return X.astype(float)

def compute_and_plot_shap(model, X, model_name, out_dir, max_display=20):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Ensure numeric & float dtypes
    X = _ensure_numeric_df(X)

    try:
        # 2) Choose explainer by model type
        if hasattr(model, "coef_"):  # linear family
            explainer = shap.Explainer(model, X)
            values = explainer(X)
        elif "xgboost" in model_name.lower():
            try:
                # Try fast TreeExplainer first (preferred)
                explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
                shap_values = explainer.shap_values(X)
                # wrap into a proper Explanation object if needed
                values = shap.Explanation(values=shap_values, data=X, feature_names=X.columns)
            except Exception as inner_e:
                print(f"TreeExplainer failed for XGBoost, fallback to KernelExplainer: {inner_e}")
                # KernelExplainer fallback
                bg = shap.sample(X, 20, random_state=0).values
                f = lambda data: model.predict(pd.DataFrame(data, columns=X.columns))
                explainer = shap.KernelExplainer(f, bg)
                shap_values = explainer.shap_values(X.values, nsamples="auto")
                values = shap.Explanation(values=shap_values, data=X, feature_names=X.columns)
        else:
            # RandomForest / trees
            explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
            values = explainer(X)

        # 3) Save CSV ranking
        mean_abs = np.abs(values.values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(Path(out_dir) / f"{model_name}_shap_summary.csv", index=False)

        # 4) Plots
        shap.summary_plot(values, X, max_display=max_display, show=False)
        plt.title(f"SHAP Summary Plot ({model_name})")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"{model_name}_shap_summary.png", dpi=150)
        plt.close()

        shap.summary_plot(values, X, plot_type="bar", max_display=max_display, show=False)
        plt.title(f"Mean |SHAP| Values ({model_name})")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"{model_name}_shap_bar.png", dpi=150)
        plt.close()

        print(f"SHAP analysis complete for {model_name}")
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")
