"""ECG-only fitting; the accepted Phase 2 alarm/matching code is unchanged."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import MODEL_CONFIG as EEG_CONFIG, THRESHOLDS, SEED
from .cardiac import FEATURES
from .metrics import evaluate_record, summarize, window_metrics

MODEL_CONFIG = {"C0_LR": dict(EEG_CONFIG["E0_LR"]), "C1_RF": dict(EEG_CONFIG["E1_RF"])}
COLUMNS = ["ecg__" + name for name in FEATURES]


def evaluate_ecg(frame, events, threshold):
    # Frozen evaluator's historic column name denotes the scored modality here.
    adapted = frame.assign(eeg_available=frame.ecg_available & frame.model_available)
    return evaluate_record(adapted, events, float(frame.recording_duration_s.iloc[0]), threshold, startup_s=60)


def ecg_window_metrics(frame):
    return window_metrics(frame.assign(eeg_available=frame.ecg_available & frame.model_available))


def fit_training(frames, names, forbidden, kind):
    if not names or set(names) & set(forbidden):
        raise ValueError("Parent overlap or empty training set")
    train = pd.concat([frames[name] for name in names], ignore_index=True)
    if set(train.file_name) != set(names):
        raise ValueError("Parent content mismatch")
    train = train.loc[train.training_eligible & train.ecg_available]
    if set(train.label.unique()) != {0, 1}:
        raise ValueError("Calibration-limited: training ECG lacks both classes")
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if kind == "C0_LR":
        steps.extend([("scaler", StandardScaler()), ("model", LogisticRegression(**MODEL_CONFIG[kind]))])
    else:
        steps.append(("model", RandomForestClassifier(**MODEL_CONFIG[kind])))
    model = Pipeline(steps)
    model.fit(train[COLUMNS], train.label.to_numpy(int))
    return model


def predict(model, frame):
    result = frame.copy()
    result["score"] = np.nan
    result["model_available"] = result.ecg_available & (model is not None)
    keep = result.model_available
    if keep.any():
        result.loc[keep, "score"] = model.predict_proba(result.loc[keep, COLUMNS])[:, 1]
    return result


def fit_outer(frames, events, held, kind, cache=None):
    names = sorted(n for n in frames if n != held)
    if len(names) < 2:
        raise ValueError("Calibration-limited: insufficient independent parents")
    cache = {} if cache is None else cache
    def fitted(train_names, forbidden):
        if set(train_names) & set(forbidden):
            raise ValueError("Parent overlap")
        key = (kind, tuple(train_names))
        if key not in cache:
            cache[key] = fit_training(frames, train_names, forbidden, kind)
        return cache[key]
    streams, support = {}, []
    for validation in names:
        training = [n for n in names if n != validation]
        streams[validation] = predict(fitted(training, [held, validation]), frames[validation])
        support.append(dict(validation_file=validation, train_files=training))
    rows = []
    for threshold in THRESHOLDS:
        metrics, matches = [], []
        for name, stream in streams.items():
            metric, _, match, _ = evaluate_ecg(stream, events[name], threshold)
            metrics.append(metric)
            matches.extend(match)
        row = summarize(metrics, matches)
        row.update(threshold=threshold, feasible=bool(np.isfinite(row["far_per_h"]) and row["far_per_h"] <= 1))
        rows.append(row)
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise ValueError("Calibration-limited: no inner ECG exposure")
    selected = min(feasible, key=lambda r: (-r["sensitivity"], r["far_per_h"], r["delay_median_s"] if np.isfinite(r["delay_median_s"]) else np.inf, -r["threshold"]))["threshold"]
    for row in rows:
        row.update(selected=row["threshold"] == selected, inner_support_json=json.dumps(support), train_files_json=json.dumps(names))
    return fitted(names, [held]), selected, pd.DataFrame(rows)
