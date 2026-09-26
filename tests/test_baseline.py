import json
import numpy as np
import pandas as pd
import pytest

from multimodal_seizure.baseline import choose_threshold, fit_outer, fit_training, predict_frame, THRESHOLDS


def parents():
    frames, events = {}, {}
    for index, name in enumerate(("one.edf", "two.edf", "PN10-4.5.6.edf")):
        times = np.arange(10, 80, 5)
        label = ((times >= 20) & (times < 30)).astype(int)
        x = label.astype(float) * 3 + index
        x[0] = np.nan
        frames[name] = pd.DataFrame(dict(file_name=name, anchor_s=times, eeg_available=True,
            training_eligible=True, label=label, evaluation_label=label, recording_duration_s=80,
            eeg__one=x, eeg__two=times.astype(float)))
        events[name] = pd.DataFrame([dict(event_id=name, onset_relative_s=20,
            offset_relative_s_canonical=30, uncertain_intervals_json="[]")])
    return frames, events


def test_held_parent_labels_and_features_cannot_change_fit_or_inner_selection():
    frames, events = parents()
    held = "PN10-4.5.6.edf"
    first = fit_outer(frames, events, held, "E0_LR")
    frames[held]["label"] = 1 - frames[held].label
    frames[held]["eeg__one"] = 1e9
    events[held]["onset_relative_s"] = 60
    second = fit_outer(frames, events, held, "E0_LR")
    assert first[2] == second[2]
    pd.testing.assert_frame_equal(first[3], second[3])
    for step, attr in (("imputer", "statistics_"), ("scaler", "mean_"), ("scaler", "scale_"), ("model", "coef_")):
        np.testing.assert_array_equal(getattr(first[0][step], attr), getattr(second[0][step], attr))
    for support in first[4]:
        assert held not in support["train_files"] and held != support["validation_file"]
        assert support["validation_file"] not in support["train_files"]
    assert len(first[3]) == 20 and set(first[3].threshold) == set(THRESHOLDS)


def test_imputation_and_scaling_fit_only_eligible_training_parents():
    frames, _ = parents()
    frames["one.edf"].loc[1, ["training_eligible", "eeg__two"]] = [False, 1e10]
    model, columns = fit_training(frames, ["one.edf"], ["two.edf", "PN10-4.5.6.edf"], "E0_LR")
    training = frames["one.edf"].loc[frames["one.edf"].training_eligible, columns].to_numpy(float)
    expected = np.nanmedian(training, axis=0)
    np.testing.assert_allclose(model["imputer"].statistics_, expected)
    filled = np.where(np.isnan(training), expected, training)
    np.testing.assert_allclose(model["scaler"].mean_, filled.mean(axis=0))
    with pytest.raises(ValueError, match="overlapping"):
        fit_training(frames, ["one.edf"], ["one.edf"], "E0_LR")


def test_same_seed_rf_replay_and_no_alarm_fallback():
    frames, events = parents()
    m1, columns = fit_training(frames, ["one.edf", "two.edf"], ["PN10-4.5.6.edf"], "E1_RF")
    m2, _ = fit_training(frames, ["one.edf", "two.edf"], ["PN10-4.5.6.edf"], "E1_RF")
    test = frames["PN10-4.5.6.edf"]
    np.testing.assert_allclose(predict_frame(m1, columns, test).score, predict_frame(m2, columns, test).score, rtol=0, atol=1e-15)
    stream = frames["one.edf"].assign(score=0.)
    threshold, table = choose_threshold({"one.edf": stream}, {"one.edf": events["one.edf"]})
    assert threshold == 1.01
    assert table.loc[table.selected, "detected"].iloc[0] == 0


def test_insufficient_independent_parents_is_explicit():
    frames, events = parents()
    frames.pop("PN10-4.5.6.edf")
    with pytest.raises(ValueError, match="fewer than two"):
        fit_outer(frames, events, "one.edf", "E0_LR")
