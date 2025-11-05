# utils/shap_utils.py
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def compute_and_plot_shap(model, X, model_name, out_dir, max_display=20):
    """
    Compute and visualize SHAP values for a trained model.
    Works for linear, tree-based, and ensemble models.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Detect model type
        if hasattr(model, "coef_"):  # linear models
            explainer = shap.Explainer(model, X)
        elif "xgboost" in model_name.lower():
            explainer = shap.TreeExplainer(model)
        else:  # tree models like RandomForest
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer(X)
        mean_abs = np.abs(shap_values.values).mean(axis=0)

        # Save SHAP summary (as CSV)
        shap_df = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(Path(out_dir) / f"{model_name}_shap_summary.csv", index=False)

        # Summary plot
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title(f"SHAP Summary Plot ({model_name})")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"{model_name}_shap_summary.png", dpi=150)
        plt.close()

        # Bar plot
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
        plt.title(f"Mean |SHAP| Values ({model_name})")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"{model_name}_shap_bar.png", dpi=150)
        plt.close()

        print(f"SHAP analysis complete for {model_name}")
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")
    