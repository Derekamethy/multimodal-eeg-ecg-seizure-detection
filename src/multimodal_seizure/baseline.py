"""Fixed EEG models and parent-disjoint, training-only threshold selection."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import evaluate_record, summarize

SEED = 20260922
THRESHOLDS = tuple(round(i / 20, 2) for i in range(1, 20)) + (1.01,)
MODEL_CONFIG = {
    "E0_LR": dict(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=2000, tol=1e-4, class_weight="balanced", random_state=SEED),
    "E1_RF": dict(n_estimators=300, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=4),
}


def make_model(kind):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if kind == "E0_LR":
        steps += [("scaler", StandardScaler()), ("model", LogisticRegression(**MODEL_CONFIG[kind]))]
    elif kind == "E1_RF":
        steps += [("model", RandomForestClassifier(**MODEL_CONFIG[kind]))]
    else:
        raise ValueError(f"Unknown EEG baseline {kind}")
    return Pipeline(steps)


def fit_training(frames, train_files, forbidden_files, kind):
    if not train_files or set(train_files) & set(forbidden_files):
        raise ValueError("Empty or overlapping parent training split")
    train = pd.concat([frames[name] for name in train_files], ignore_index=True)
    if set(train.file_name) != set(train_files):
        raise ValueError("Parent content does not match split keys")
    train = train.loc[train.training_eligible & train.eeg_available]
    columns = sorted(c for c in train if c.startswith("eeg__"))
    if not columns or set(train.label.unique()) != {0, 1}:
        raise ValueError("Calibration-limited: training parents do not contain both classes")
    model = make_model(kind)
    model.fit(train[columns], train.label.to_numpy(int))
    return model, columns


def predict_frame(model, columns, frame):
    out = frame.copy()
    out["score"] = np.nan
    keep = out.eeg_available
    if keep.any():
        out.loc[keep, "score"] = model.predict_proba(out.loc[keep, columns])[:, list(model.classes_).index(1)]
    return out


def choose_threshold(streams, events):
    """Only inner held-parent streams and references are accepted here."""
    rows = []
    for threshold in THRESHOLDS:
        metric_rows, matches = [], []
        for filename, frame in streams.items():
            metric, _, match, _ = evaluate_record(frame, events[filename], float(frame.recording_duration_s.iloc[0]), threshold)
            metric_rows.append(metric)
            matches.extend(match)
        row = summarize(metric_rows, matches)
        row.update(threshold=threshold, feasible=bool(np.isfinite(row["far_per_h"]) and row["far_per_h"] <= 1.0))
        rows.append(row)
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise ValueError("Calibration-limited: no evaluable inner exposure")
    chosen = min(feasible, key=lambda r: (-r["sensitivity"], r["far_per_h"],
                  r["delay_median_s"] if np.isfinite(r["delay_median_s"]) else float("inf"), -r["threshold"]))
    for row in rows:
        row["selected"] = row["threshold"] == chosen["threshold"]
    return chosen["threshold"], pd.DataFrame(rows)


def fit_outer(frames, events, test_file, kind, model_cache=None):
    """The test frame and its labels are never read while fitting/selecting."""
    train_files = sorted(name for name in frames if name != test_file)
    if len(train_files) < 2:
        raise ValueError("Calibration-limited: fewer than two independent inner parents")
    cache = {} if model_cache is None else model_cache

    def fitted(names, forbidden):
        if set(names) & set(forbidden):
            raise ValueError("Outer/inner parent overlap")
        key = (kind, tuple(names))
        if key not in cache:
            cache[key] = fit_training(frames, names, forbidden, kind)
        return cache[key]

    inner, support = {}, []
    for validation_file in train_files:
        inner_train = [name for name in train_files if name != validation_file]
        model, columns = fitted(inner_train, [test_file, validation_file])
        inner[validation_file] = predict_frame(model, columns, frames[validation_file])
        support.append(dict(validation_file=validation_file, train_files=inner_train))
    threshold, table = choose_threshold(inner, {name: events[name] for name in train_files})
    model, columns = fitted(train_files, [test_file])
    table["inner_support_json"] = json.dumps(support)
    table["train_files_json"] = json.dumps(train_files)
    return model, columns, threshold, table, support
