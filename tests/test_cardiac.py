import copy
import numpy as np
import pandas as pd
import pytest

from multimodal_seizure.cardiac import analyze_lead, analyze_pair, rr_series, rhythm_features, relative_features, RULES
from multimodal_seizure.ecg_baseline import COLUMNS, fit_training, fit_outer, predict, evaluate_ecg


def synthetic(times=None, duration=60, fs=512):
    t = np.arange(int(duration * fs)) / fs
    times = np.arange(1, duration - .5, 1.) if times is None else np.asarray(times)
    x = 20 * np.sin(2 * np.pi * .5 * t)
    for beat in times:
        x += 1000 * np.exp(-.5 * ((t - beat) / .015) ** 2)
        x -= 200 * np.exp(-.5 * ((t - beat - .05) / .012) ** 2)
    return x, times


def test_periodic_beat_timing_hr_and_deterministic_replay():
    x, expected = synthetic()
    first = analyze_lead(x)
    second = analyze_lead(x)
    assert first["evidence"]["good"]
    np.testing.assert_array_equal(first["peaks"], second["peaks"])
    detected = first["peaks"] / 512
    assert abs(len(detected) - len(expected)) <= 1
    assert max(min(abs(detected - t)) for t in expected[1:-1]) < .08
    features = rhythm_features(first["series"], 0)
    assert features["median_hr"] == pytest.approx(60, abs=1)
    assert features["median_rr"] == pytest.approx(1, abs=.02)
    assert features["rmssd_s"] < .02


def test_rr_gaps_invalid_values_and_actual_time_trend():
    series = rr_series([0, 100, 350, 470], 100)
    assert series["valid"].tolist() == [True, False, True]
    assert series["hr"].tolist() == pytest.approx([60, 24, 50])
    assert np.isnan(rhythm_features(series, 0)["rmssd_s"])
    assert not rr_series([0, 10], 100)["valid"].any()
    beats = np.cumsum(np.linspace(1.1, .6, 60))
    x, _ = synthetic(beats[beats < 59])
    result = analyze_lead(x)
    assert result["evidence"]["good"]
    features = rhythm_features(result["series"], 0)
    assert features["hr_slope_bpm_min"] > 0 and features["early_late_hr"] > 0


def test_bad_signals_too_few_peaks_noise_and_saturation():
    good, _ = synthetic()
    few, _ = synthetic([10, 20])
    for bad in (np.zeros_like(good), np.full_like(good, np.nan), few, np.random.default_rng(2).normal(0, 100, len(good))):
        assert not analyze_lead(bad)["evidence"]["good"]
    assert not analyze_lead(good, rails=(-1, 1))["evidence"]["good"]


def test_lead_selection_and_disagreement():
    x, _ = synthetic()
    bad = np.zeros_like(x)
    assert analyze_pair([x, bad])["selected"] == 0
    assert analyze_pair([bad, x])["selected"] == 1
    assert analyze_pair([x, x])["selected"] == 0
    assert analyze_pair([bad, bad])["selected"] is None
    fast, _ = synthetic(np.arange(1, 59, .6))
    result = analyze_pair([x, fast])
    assert result["selected"] is None and result["reason"] == "gross_interlead_disagreement"


def test_past_only_windows_and_relative_history():
    x, _ = synthetic(duration=400)
    t = 180
    def feature_at(data):
        history = {u: analyze_pair([data[(u-60)*512:u*512]] * 2, start_s=u-60) for u in (60, 120)}
        current = analyze_pair([data[(t-60)*512:t*512]] * 2, start_s=t-60)
        return current, relative_features(current, history, t)
    first = feature_at(x)
    x[t*512:] = 1e9
    second = feature_at(x)
    assert first[0]["features"] == second[0]["features"]
    assert first[1] == second[1]
    assert first[1][1]["baseline_ready"]
    relative, baseline = relative_features(first[0], {}, 60)
    assert not baseline["baseline_ready"] and all(np.isnan(v) for v in relative.values())


def frames():
    data, events = {}, {}
    for i, name in enumerate(("a.edf", "b.edf", "multi.4.5.6.edf")):
        times = np.arange(60, 140, 5)
        label = ((times >= 80) & (times < 100)).astype(int)
        row = dict(file_name=name, anchor_s=times, label=label, evaluation_label=label,
            ecg_available=True, training_eligible=True, recording_duration_s=140)
        row.update({col: label.astype(float) + i * .1 for col in COLUMNS})
        data[name] = pd.DataFrame(row)
        data[name].loc[0, COLUMNS[0]] = np.nan
        events[name] = pd.DataFrame([dict(event_id=name, onset_relative_s=80, offset_relative_s_canonical=100, uncertain_intervals_json="[]")])
    return data, events


def test_parent_leakage_preprocessing_and_fixed_quality_rules():
    data, events = frames()
    held = "multi.4.5.6.edf"
    frozen = copy.deepcopy(RULES)
    first = fit_outer(data, events, held, "C0_LR")
    data[held]["label"] = 1 - data[held].label
    data[held][COLUMNS] = 1e9
    data[held]["ecg_available"] = False
    events[held]["onset_relative_s"] = 0
    second = fit_outer(data, events, held, "C0_LR")
    assert first[1] == second[1]
    pd.testing.assert_frame_equal(first[2], second[2])
    np.testing.assert_array_equal(first[0]["model"].coef_, second[0]["model"].coef_)
    train = pd.concat([data["a.edf"], data["b.edf"]])[COLUMNS].to_numpy()
    median = np.nanmedian(train, axis=0)
    np.testing.assert_allclose(first[0]["imputer"].statistics_, median)
    np.testing.assert_allclose(first[0]["scaler"].mean_, np.where(np.isnan(train), median, train).mean(axis=0))
    assert RULES == frozen
    with pytest.raises(ValueError, match="overlap"):
        fit_training(data, [held], [held], "C0_LR")


def test_ecg_availability_breaks_frozen_alarms_and_excludes_exposure():
    data, events = frames()
    stream = data["a.edf"].assign(score=1., model_available=True)
    stream.loc[stream.anchor_s == 75, "ecg_available"] = False
    metric, alarms, _, _ = evaluate_ecg(stream, events["a.edf"], .5)
    assert [a["declaration_s"] for a in alarms] == [60, 80]
    assert metric["startup_s"] == 60 and metric["other_unavailable_s"] == 5
    assert metric["evaluable_s"] == 75


def test_real_official_leads_and_rr_reconstruction():
    from multimodal_seizure.data import get_official_channels, read_edf_segment, read_event_table
    events = read_event_table()
    for patient in sorted(events.patient.unique()):
        filename = sorted(events.loc[events.patient == patient, "canonical_file_name"].unique())[0]
        mapping = get_official_channels(patient, "ECG", filename)
        assert mapping.official_name.tolist() == ["EKG 1", "EKG 2"]
        assert mapping.edf_raw_label.tolist() == ["1", "2"]
        data, fs = read_edf_segment(patient, filename, mapping.edf_index_0based, 60, 60)
        first, replay = analyze_pair(data, fs, 60), analyze_pair(data, fs, 60)
        assert first["selected"] == replay["selected"]
        for a, b in zip(first["leads"], replay["leads"]):
            np.testing.assert_array_equal(a["peaks"], b["peaks"])
            np.testing.assert_allclose(a["series"]["rr"], np.diff(a["peaks"]) / fs)
            valid = a["series"]["valid"]
            np.testing.assert_allclose(a["series"]["hr"][valid], 60 / a["series"]["rr"][valid])
