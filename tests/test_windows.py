import numpy as np

from multimodal_seizure.data import read_event_table
from multimodal_seizure.windows import build_patient_folds, build_window_index


def test_pn10_combined_recording_preserves_parent_unit_and_three_seizures():
    w = build_window_index("PN10", "PN10-4.5.6.edf")
    positive = w[w.label == 1]
    seen = set()
    for x in positive.seizure_numbers:
        seen.update(int(v) for v in x.split("|") if v)
    assert seen == {4, 5, 6}
    assert (w.file_name == "PN10-4.5.6.edf").all()


def test_negative_windows_are_outside_five_minute_exclusion_zone():
    events = read_event_table()
    w = build_window_index("PN12", "PN12-1.2.edf")
    neg = w[w.label == 0]
    ev = events[(events.patient == "PN12") & (events.canonical_file_name == "PN12-1.2.edf")]
    for t in neg.anchor_s:
        for _, row in ev.iterrows():
            assert not (
                row.onset_relative_s - 300.0
                <= t
                < row.offset_relative_s_canonical + 300.0
            )
def test_context_windows_are_causal_and_end_at_anchor():
    w = build_window_index("PN00", "PN00-1.edf")
    assert np.allclose(w.eeg_end_s, w.anchor_s)
    assert np.allclose(w.ecg_end_s, w.anchor_s)
    assert np.allclose(w.anchor_s - w.eeg_start_s, 10.0)
    available = w.ecg_history_available
    assert np.allclose(w.loc[available, "anchor_s"] - w.loc[available, "ecg_start_s"], 60.0)
    assert w.loc[~available, "ecg_start_s"].isna().all()
    assert (w.eeg_start_s >= 0).all()
    assert (w.loc[available, "ecg_start_s"] >= 0).all()


def test_patient_folds_hold_out_whole_edf():
    expected = {"PN00": 4, "PN06": 5, "PN10": 6, "PN12": 3, "PN14": 4}
    for patient, n in expected.items():
        folds = build_patient_folds(patient)
        assert len(folds) == n
        for fold in folds:
            assert fold["test_file"] not in fold["train_files"]
            assert len(fold["train_files"]) == n - 1
