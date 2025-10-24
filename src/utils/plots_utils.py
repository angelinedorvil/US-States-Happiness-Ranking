from sklearn.metrics import precision_recall_curve, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel
from sklearn.model_selection import cross_val_score


def plot_roc_pr_curves(model, X_test, y_test, out_dir, model_name):
    classes = sorted(np.unique(y_test))
    y_bin = label_binarize(y_test, classes=classes)
    
    try:
        Y_score = model.predict_proba(X_test)
    except Exception:
        return  

    # ROC
    plt.figure()
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], Y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {c} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC: {model_name}')
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(Path(out_dir)/f"{model_name}_roc_curve.png", dpi=150)
    plt.close()

    # PR
    plt.figure()
    for i, c in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], Y_score[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Class {c} (AUC={pr_auc:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR: {model_name}')
    plt.legend(loc='lower left'); plt.tight_layout()
    plt.savefig(Path(out_dir)/f"{model_name}_pr_curve.png", dpi=150)
    plt.close()


def regression_mcc_from_bins(y_true_cont, y_pred_cont, n_bins=5):
    bins = pd.qcut(y_true_cont, q=n_bins, labels=False, duplicates='drop')
    bins_pred = pd.qcut(y_pred_cont, q=n_bins, labels=False, duplicates='drop')
    # Drop NaNs if any
    mask = (~pd.isna(bins)) & (~pd.isna(bins_pred))
    if mask.sum() == 0:
        return None
    return float(matthews_corrcoef(bins[mask], bins_pred[mask]))

def paired_ttest_models(model_a, model_b, X, y, scoring, cv):
    scores_a = cross_val_score(model_a, X, y, scoring=scoring, cv=cv)
    scores_b = cross_val_score(model_b, X, y, scoring=scoring, cv=cv)
    t, p = ttest_rel(scores_a, scores_b)
    return {"mean_a": float(scores_a.mean()), "mean_b": float(scores_b.mean()),
            "t_stat": float(t), "p_value": float(p)}