from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np, json, glob, pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel

def evaluate_model(model, X_test, y_test, model_name, out_dir):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, digits=3, output_dict=True)
    cm  = confusion_matrix(y_test, y_pred)
    auroc = None

    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=sorted(set(y_test)))
        auroc = roc_auc_score(y_bin, model.predict_proba(X_test), multi_class='ovr', average='macro')
    except Exception:
        pass
    out = {
        "accuracy": acc,
        "macro_f1": rep["macro avg"]["f1-score"],
        "macro_precision": rep["macro avg"]["precision"],
        "macro_recall": rep["macro avg"]["recall"],
        "auroc_macro": auroc
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{model_name}_test_report.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(cm).to_csv(Path(out_dir) / f"{model_name}_confusion_matrix.csv")

    reports = []
    for file in glob.glob(f"{out_dir}/*_test_report.json"):
        name = Path(file).stem.replace("_test_report", "")
        with open(file) as f:
            d = json.load(f)
            d["model"] = name
            reports.append(d)

    print(f"Evaulation finished for {model_name}. Results saved to {out_dir}")
    print(f"Report finished for {model_name}")
    return out
