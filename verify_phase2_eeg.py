"""Read-only replay of saved Phase 2 scores; never fits a model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal_seizure.data import PROJECT_ROOT, read_event_table
from src.multimodal_seizure.metrics import evaluate_record, window_metrics, summarize
from src.multimodal_seizure.provenance import metadata_identity, require_current, sha256


def verify(output):
    require_current(output / "manifest.json")
    manifest = json.loads((output / "manifest.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert config["metadata_identity"] == metadata_identity()
    for artifact in manifest["artifacts"]:
        path = PROJECT_ROOT / artifact["path"]
        assert sha256(path) == artifact["sha256"]
        require_current(path)
    historical = json.loads((PROJECT_ROOT / "metadata/canonical_manifest.json").read_text())
    assert all(sha256(PROJECT_ROOT / a["path"]) == a["sha256"] for a in historical["downstream_artifacts"])
    events = read_event_table()
    saved_folds = pd.read_csv(output / "fold_metrics.csv", float_precision="round_trip")
    thresholds = pd.read_csv(output / "threshold_selection.csv", float_precision="round_trip")
    alarm_rows, event_rows, exposure_rows, computed_folds = [], [], [], []
    streams = {}
    for path in sorted((output / "scores").glob("*.csv")):
        frame = pd.read_csv(path, float_precision="round_trip")
        for (patient, model, filename), stream in frame.groupby(["patient", "model", "file_name"]):
            assert stream.anchor_s.iloc[0] == 10
            np.testing.assert_array_equal(np.diff(stream.anchor_s), 5)
            assert filename not in json.loads(stream.train_files_json.iloc[0])
            refs = events.loc[(events.patient == patient) & (events.canonical_file_name == filename)]
            assert not refs.empty and refs.primary_eligible.all()
            choices = thresholds.loc[(thresholds.patient == patient) & (thresholds.model == model) & (thresholds.test_file == filename)]
            assert len(choices) == 20 and choices.selected.sum() == 1
            best = choices.loc[choices.feasible].sort_values(
                ["sensitivity", "far_per_h", "delay_median_s", "threshold"], ascending=[False, True, True, False], na_position="last").iloc[0]
            threshold = float(stream.threshold.iloc[0])
            assert threshold == best.threshold
            for inner in json.loads(best.inner_support_json):
                assert filename != inner["validation_file"] and filename not in inner["train_files"]
                assert inner["validation_file"] not in inner["train_files"]
            metric, alarms, matches, exposure = evaluate_record(stream, refs, float(stream.recording_duration_s.iloc[0]), threshold)
            metric.update(window_metrics(stream))
            saved = saved_folds.loc[(saved_folds.patient == patient) & (saved_folds.model == model) & (saved_folds.test_file == filename)].iloc[0]
            for key, value in metric.items():
                np.testing.assert_allclose(value, saved[key], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=key)
            tags = dict(patient=patient, model=model, test_file=filename)
            computed_folds.append(dict(**tags, **metric))
            alarm_rows.extend(dict(**tags, **a) for a in alarms)
            event_rows.extend(dict(**tags, **e) for e in matches)
            exposure_rows.extend(dict(**tags, start_s=a, end_s=b) for a, b in exposure)
            streams[(patient, model, filename)] = stream
    for name, rows in (("alarms.csv", alarm_rows), ("event_matches.csv", event_rows), ("evaluation_exposure.csv", exposure_rows)):
        actual = pd.DataFrame(rows).sort_values(["patient", "model", "test_file"], kind="stable").reset_index(drop=True)
        expected = pd.read_csv(output / name, float_precision="round_trip").sort_values(["patient", "model", "test_file"], kind="stable").reset_index(drop=True)
        actual = actual.replace("", np.nan)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    for name, keys in (("patient_metrics.csv", ["patient", "model"]), ("pooled_metrics.csv", ["model"])):
        saved = pd.read_csv(output / name, float_precision="round_trip")
        for _, row in saved.iterrows():
            relevant = lambda item: all(item[key] == row[key] for key in keys)
            metric = summarize([r for r in computed_folds if relevant(r)], [r for r in event_rows if relevant(r)])
            for key, value in metric.items():
                np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-12, equal_nan=True)
    assert len(computed_folds) == 44
    assert manifest["failed_folds"] == manifest["calibration_limited_folds"] == []
    print("REPLAY_PASS 44/44 model-folds; thresholds, alarms, matches, coverage, metrics, provenance and historical preservation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output.resolve())
