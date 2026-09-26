"""Phase 2 only: continuous handcrafted EEG, nested parent-held baselines, frozen alarms."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pyedflib

from src.multimodal_seizure.baseline import MODEL_CONFIG, SEED, THRESHOLDS, fit_outer, predict_frame
from src.multimodal_seizure.data import PROJECT_ROOT, RAW_ROOT, get_official_channels, read_event_table
from src.multimodal_seizure.eeg import eeg_features
from src.multimodal_seizure.metrics import evaluate_record, summarize, window_metrics
from src.multimodal_seizure.provenance import artifact_spec, require_current, sha256, stamp_artifact
from src.multimodal_seizure.windows import build_window_index

CODE = [Path(__file__).resolve()] + [PROJECT_ROOT / f"src/multimodal_seizure/{name}.py"
    for name in ("windows", "eeg", "data", "alarms", "metrics", "baseline", "provenance")]
CONFIG = dict(phase=2, seed=SEED, models=MODEL_CONFIG, threshold_grid=THRESHOLDS,
    first_eeg_decision_s=10, stride_s=5, eeg_context_s=10, training_exclusion_s=300,
    inner_far_limit_per_h=1, threshold_ties=["max_sensitivity", "min_FAR", "min_median_delay", "max_threshold"],
    alarm_policy="first >= threshold opens; two consecutive lows close; no smoothing/refractory",
    matching="new declaration in [onset,offset); duplicates count as unmatched in primary FAR",
    uncertainty="exact canonical intervals excluded; barriers reset state, including between decisions",
    exposure="union of finite available decision cells [t,min(t+5,duration)), minus uncertainty/gaps",
    startup="[0,10); EEG unavailable; ECG history does not gate EEG",
    feature_units="microvolts; PSD microvolts^2/Hz; official EEG channels; fs=512Hz",
    feature_missingness="partial flat-channel spectral/correlation NaNs imputed from training; all-flat/nonfinite EEG unavailable",
    fallback="no external threshold fallback; unsupported inner calibration is explicitly ineligible",
    interpretation="retrospective offline EEG baseline; no runtime latency or clinical real-time claim")


def save_csv(path, frame, spec):
    if path.exists():
        require_current(path, expected_identity=spec["cache_identity"])
    frame.to_csv(path, index=False, float_format="%.17g")
    stamp_artifact(path, spec)


def extract_record(patient, filename, output, config_path):
    output, config_path = Path(output), Path(config_path)
    spec = artifact_spec("phase2_eeg_features", CODE + [config_path], dict(patient=patient, file_name=filename))
    if output.exists():
        require_current(output, expected_identity=spec["cache_identity"])
        return patient, filename, len(pd.read_csv(output)), "verified_cache"
    frame = build_window_index(patient, filename)
    indices = get_official_channels(patient, "EEG", filename).edf_index_0based.to_numpy(int)
    reader = pyedflib.EdfReader(str(RAW_ROOT / patient / filename))
    features, reasons = [], []
    try:
        if any(reader.getSampleFrequency(int(i)) != 512 for i in indices):
            raise ValueError("Unexpected EEG sampling rate")
        if any(reader.getPhysicalDimension(int(i)).strip().lower() not in {"uv", "µv"} for i in indices):
            raise ValueError("Unexpected EEG physical units; no implicit amplitude conversion")
        fs, previous_chunk, signals = 512, None, None
        for t in frame.anchor_s:
            chunk = int((t - 10) // 300) * 300
            if chunk != previous_chunk:
                end = min(chunk + 310, float(reader.file_duration))
                signals = np.vstack([reader.readSignal(int(i), start=chunk * fs, n=int((end - chunk) * fs)) for i in indices])
                previous_chunk = chunk
            start = int((t - 10 - chunk) * fs)
            segment = signals[:, start:start + 10 * fs]
            if segment.shape[1] != 10 * fs:
                raise ValueError("Short EEG support interval")
            if not np.isfinite(segment).all() or not np.any(np.std(segment, axis=1) > 0):
                features.append({})
                reasons.append("nonfinite_or_all_flat")
            else:
                features.append({f"eeg__{k}": v for k, v in eeg_features(segment, fs).items()})
                reasons.append("")
    finally:
        reader.close()
    frame["unavailable_reason"] = reasons
    frame["eeg_available"] = frame.unavailable_reason == ""
    frame["training_eligible"] &= frame.eeg_available
    frame = pd.concat([frame, pd.DataFrame(features)], axis=1)
    if len([c for c in frame if c.startswith("eeg__")]) != 46:
        raise ValueError("Unexpected EEG feature representation")
    save_csv(output, frame, spec)
    return patient, filename, len(frame), "extracted"


def experiment():
    resolved = dict(CONFIG, python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "pyedflib")})
    spec = artifact_spec("phase2_config", CODE, resolved)
    identity = spec["cache_identity"]
    output = PROJECT_ROOT / "results/phase2_eeg" / identity[:12]
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        require_current(config_path, expected_identity=identity)
    else:
        config_path.write_text(json.dumps(dict(resolved, experiment_identity=identity,
            metadata_identity=spec["metadata_identity"], code_sha256=spec["dependencies"]), indent=2) + "\n", encoding="utf-8")
        stamp_artifact(config_path, spec)
    return output, config_path, identity


def feature_stage(output, config_path, events, workers):
    destination = output / "features"
    destination.mkdir(exist_ok=True)
    parents = events[["patient", "canonical_file_name"]].drop_duplicates()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(extract_record, row.patient, row.canonical_file_name,
            str(destination / f"{row.patient}__{row.canonical_file_name}.csv"), str(config_path))
            for row in parents.itertuples()]
        for future in as_completed(jobs):
            print("FEATURES", *future.result(), flush=True)


def baseline_stage(output, config_path, events):
    scores_dir = output / "scores"
    scores_dir.mkdir(exist_ok=True)
    fold_rows, threshold_rows, alarm_rows, event_rows, exposure_rows, patient_rows = [], [], [], [], [], []
    feature_paths = sorted((output / "features").glob("*.csv"))
    score_paths = []
    for patient in sorted(events.patient.unique()):
        selected = events.loc[events.patient == patient]
        references = {name: group.copy() for name, group in selected.groupby("canonical_file_name")}
        frames = {}
        for name in references:
            path = output / "features" / f"{patient}__{name}.csv"
            require_current(path)
            frames[name] = pd.read_csv(path, float_precision="round_trip")
        model_cache = {}
        for kind in MODEL_CONFIG:
            streams = []
            for test_file in sorted(frames):
                print("FIT", patient, kind, "OUTER", test_file, flush=True)
                model, columns, threshold, table, support = fit_outer(frames, references, test_file, kind, model_cache)
                score = predict_frame(model, columns, frames[test_file])
                score = score.drop(columns=[c for c in score if c.startswith("eeg__")])
                score["threshold"] = threshold
                score["model"] = kind
                score["seed"] = SEED
                score["model_available"] = score.eeg_available & np.isfinite(score.score)
                score["calibration_status"] = "nested"
                score["train_files_json"] = json.dumps(sorted(name for name in frames if name != test_file))
                streams.append(score)
                threshold_rows.append(table.assign(patient=patient, model=kind, test_file=test_file, calibration_status="nested"))
            all_scores = pd.concat(streams, ignore_index=True)
            score_path = scores_dir / f"{patient}_{kind}_continuous_oof.csv"
            score_spec = artifact_spec("phase2_continuous_oof", CODE + [config_path] +
                [output / "features" / f"{patient}__{name}.csv" for name in frames], dict(patient=patient, model=kind))
            # Persist raw continuous scores BEFORE outer event evaluation.
            save_csv(score_path, all_scores, score_spec)
            score_paths.append(score_path)
            patient_metrics, patient_matches = [], []
            for test_file, stream in all_scores.groupby("file_name", sort=True):
                threshold = float(stream.threshold.iloc[0])
                metric, alarms, matches, exposure = evaluate_record(stream, references[test_file],
                    float(stream.recording_duration_s.iloc[0]), threshold)
                tags = dict(patient=patient, model=kind, test_file=test_file)
                metric.update(window_metrics(stream), **tags, threshold=threshold, calibration_status="nested",
                              train_files_json=stream.train_files_json.iloc[0], seed=SEED)
                fold_rows.append(metric)
                patient_metrics.append(metric)
                patient_matches.extend(matches)
                alarm_rows.extend(dict(**tags, **a) for a in alarms)
                event_rows.extend(dict(**tags, **m) for m in matches)
                exposure_rows.extend(dict(**tags, start_s=a, end_s=b) for a, b in exposure)
            combined = summarize(patient_metrics, patient_matches)
            combined.update(window_metrics(all_scores), patient=patient, model=kind)
            patient_rows.append(combined)
            print("EVALUATED", patient, kind, "ALL_FOLDS_SAVED", flush=True)
    thresholds = pd.concat(threshold_rows, ignore_index=True)
    spec = artifact_spec("phase2_evaluation", CODE + [config_path] + score_paths)
    pooled = []
    for kind in MODEL_CONFIG:
        row = summarize([r for r in fold_rows if r["model"] == kind], [r for r in event_rows if r["model"] == kind])
        scores = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in score_paths if kind in p.name], ignore_index=True)
        row.update(window_metrics(scores), model=kind)
        pooled.append(row)
    tables = {"threshold_selection.csv": thresholds, "fold_metrics.csv": pd.DataFrame(fold_rows),
              "patient_metrics.csv": pd.DataFrame(patient_rows), "pooled_metrics.csv": pd.DataFrame(pooled),
              "alarms.csv": pd.DataFrame(alarm_rows), "event_matches.csv": pd.DataFrame(event_rows),
              "evaluation_exposure.csv": pd.DataFrame(exposure_rows)}
    for name, frame in tables.items():
        save_csv(output / name, frame, spec)
    manifest = dict(experiment_identity=json.loads(config_path.read_text())["experiment_identity"],
        status="complete", expected_outer_folds=22, model_folds=len(fold_rows), calibration_limited_folds=[],
        failed_folds=[], raw_sources="31/31 verified before Phase 2", scope="EEG only",
        artifacts=[dict(path=p.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(p),
                        sidecar=p.with_suffix(p.suffix + ".provenance.json").relative_to(PROJECT_ROOT).as_posix())
                   for p in [config_path, *feature_paths, *score_paths, *(output / name for name in tables)]])
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    stamp_artifact(path, artifact_spec("phase2_manifest", CODE + [config_path] + [output / name for name in tables]))
    print("PHASE2_COMPLETE", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("features", "baseline", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output, config_path, identity = experiment()
    print("EXPERIMENT", identity, output, flush=True)
    events = read_event_table()
    try:
        if args.stage in {"features", "all"}:
            feature_stage(output, config_path, events, args.workers)
        if args.stage in {"baseline", "all"}:
            baseline_stage(output, config_path, events)
    except Exception as exc:
        failure = output / "failure.json"
        failure.write_text(json.dumps(dict(status="failed", error=repr(exc)), indent=2), encoding="utf-8")
        stamp_artifact(failure, artifact_spec("phase2_failure", CODE + [config_path]))
        raise


if __name__ == "__main__":
    main()
