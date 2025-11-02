import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)


def compute_basic_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    return {
        "accuracy": acc,
        "f1": f1,
        "mcc": mcc,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc,
    }


def latency_percentiles(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.array(latencies_ms)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99))}
