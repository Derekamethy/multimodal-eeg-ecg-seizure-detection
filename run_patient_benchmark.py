from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.multimodal_seizure.features import feature_columns_by_modality
from src.multimodal_seizure.provenance import artifact_spec, require_current, protect_existing, stamp_artifact

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("patient")
args = parser.parse_args()
patient = args.patient

feature_path = ROOT / "results" / "features" / f"{patient}_features.canonical.csv"
require_current(feature_path)
out_dir = ROOT / "results" / "benchmarks"
outputs = [out_dir / f"{patient}_canonical_{suffix}.csv" for suffix in ("fold_metrics", "oof_predictions", "pooled_metrics")]
for path in outputs:
    protect_existing(path)
spec = artifact_spec("benchmark", [Path(__file__).resolve(), feature_path])
df = pd.read_csv(feature_path)
eeg_cols, ecg_cols = feature_columns_by_modality(list(df.columns))

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=20260921,
            n_jobs=-1,
        )),
    ])
def score_row(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "sensitivity_at_0_5": recall_score(y, pred, zero_division=0),
        "precision_at_0_5": precision_score(y, pred, zero_division=0),
        "f1_at_0_5": f1_score(y, pred, zero_division=0),
        "balanced_accuracy_at_0_5": balanced_accuracy_score(y, pred),
    }

fold_rows = []
oof_rows = []
for test_file in sorted(df.file_name.unique()):
    train = df[df.file_name != test_file]
    test = df[df.file_name == test_file]
    y_train = train.label.to_numpy(int)
    y_test = test.label.to_numpy(int)

    eeg_model, ecg_model, early_model = make_model(), make_model(), make_model()
    eeg_model.fit(train[eeg_cols], y_train)
    ecg_model.fit(train[ecg_cols], y_train)
    early_model.fit(train[eeg_cols + ecg_cols], y_train)

    probs = {
        "EEG_only": eeg_model.predict_proba(test[eeg_cols])[:, 1],
        "ECG_only": ecg_model.predict_proba(test[ecg_cols])[:, 1],
        "Early_fusion": early_model.predict_proba(test[eeg_cols + ecg_cols])[:, 1],
    }
    probs["Late_fusion_alpha_0.5"] = 0.5 * probs["EEG_only"] + 0.5 * probs["ECG_only"]

    for model, p in probs.items():
        row = {
            "patient": patient,
            "test_file": test_file,
            "model": model,
            "n_test": len(test),
            "n_positive": int(y_test.sum()),
            "n_negative": int((y_test == 0).sum()),
        }
        row.update(score_row(y_test, p))
        fold_rows.append(row)
        for idx, prob in zip(test.index, p):
            oof_rows.append({
                "patient": patient, "file_name": test_file, "row_index": int(idx),
                "anchor_s": float(df.loc[idx, "anchor_s"]), "label": int(df.loc[idx, "label"]),
                "model": model, "probability": float(prob),
            })
folds = pd.DataFrame(fold_rows)
oof = pd.DataFrame(oof_rows)
out_dir = ROOT / "results" / "benchmarks"
out_dir.mkdir(parents=True, exist_ok=True)
folds.to_csv(outputs[0], index=False)
oof.to_csv(outputs[1], index=False)
stamp_artifact(outputs[0], spec)
stamp_artifact(outputs[1], spec)

pooled_rows = []
for model, g in oof.groupby("model"):
    row = {"patient": patient, "model": model, "n": len(g), "n_positive": int(g.label.sum())}
    row.update(score_row(g.label.to_numpy(int), g.probability.to_numpy(float)))
    pooled_rows.append(row)
pooled = pd.DataFrame(pooled_rows)
pooled.to_csv(outputs[2], index=False)
stamp_artifact(outputs[2], spec)

print("FOLD_METRICS")
print(folds.to_string(index=False))
print("\nPOOLED_OOF_METRICS")
print(pooled.to_string(index=False))
print("\nFULL_NEGATIVE_PARENT_DISJOINT_BASELINE")
