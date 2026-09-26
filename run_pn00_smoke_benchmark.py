from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.multimodal_seizure.features import feature_columns_by_modality
from src.multimodal_seizure.provenance import artifact_spec, require_current, protect_existing, stamp_artifact

ROOT = Path(__file__).resolve().parent
feature_path = ROOT / "pn00_smoke_features.canonical.csv"
require_current(feature_path)
output = ROOT / "pn00_smoke_benchmark.canonical.csv"
protect_existing(output)
spec = artifact_spec("smoke_benchmark", [Path(__file__).resolve(), feature_path])
df = pd.read_csv(feature_path)
eeg_cols, ecg_cols = feature_columns_by_modality(list(df.columns))

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=20260921,
            n_jobs=-1,
        )),
    ])

def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "f1_at_0_5": f1_score(y, pred, zero_division=0),
        "balanced_accuracy_at_0_5": balanced_accuracy_score(y, pred),
    }
rows = []
for test_file in sorted(df.file_name.unique()):
    train = df[df.file_name != test_file]
    test = df[df.file_name == test_file]
    y_train = train.label.to_numpy(int)
    y_test = test.label.to_numpy(int)

    eeg_model = make_model()
    ecg_model = make_model()
    early_model = make_model()

    eeg_model.fit(train[eeg_cols], y_train)
    ecg_model.fit(train[ecg_cols], y_train)
    early_model.fit(train[eeg_cols + ecg_cols], y_train)

    p_eeg = eeg_model.predict_proba(test[eeg_cols])[:, 1]
    p_ecg = ecg_model.predict_proba(test[ecg_cols])[:, 1]
    p_early = early_model.predict_proba(test[eeg_cols + ecg_cols])[:, 1]
    p_late = 0.5 * p_eeg + 0.5 * p_ecg

    for name, p in [
        ("EEG_only", p_eeg),
        ("ECG_only", p_ecg),
        ("Early_fusion", p_early),
        ("Late_fusion_alpha_0.5", p_late),
    ]:
        row = {
            "test_file": test_file,
            "model": name,
            "n_test": len(test),
            "n_positive": int(y_test.sum()),
            "n_negative": int((y_test == 0).sum()),
        }
        row.update(metrics(y_test, p))
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(output, index=False)
stamp_artifact(output, spec)
print(out.to_string(index=False))
print("\nMEAN_BY_MODEL")
print(out.groupby("model")[["roc_auc","average_precision","f1_at_0_5","balanced_accuracy_at_0_5"]].mean().to_string())
print("\nSMOKE_ONLY_NOT_FOR_CLAIMS")
