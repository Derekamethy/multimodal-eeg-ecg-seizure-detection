from pathlib import Path

import numpy as np

from multimodal_seizure.data import (
    PILOT_PATIENTS,
    get_official_channels,
    read_edf_segment,
    read_event_table,
)


def test_all_pilot_patients_have_two_official_ecg_channels():
    for patient in PILOT_PATIENTS:
        ecg = get_official_channels(patient, "ECG")
        assert len(ecg) == 2
        assert list(ecg["official_name"]) == ["EKG 1", "EKG 2"]
        assert list(ecg["edf_raw_label"]) == ["1", "2"]


def test_canonical_event_table_preserves_28_events_with_27_primary_in_bounds():
    events = read_event_table()
    assert len(read_event_table(include_quarantined=True)) == 28
    assert len(events) == 27
    assert set(events["patient"]) == set(PILOT_PATIENTS)
    assert (events["onset_relative_s"] >= 0).all()
    assert (events["offset_relative_s_canonical"] > events["onset_relative_s"]).all()
    assert (events["offset_relative_s_canonical"] <= events["edf_duration_s"]).all()
def test_pn00_short_eeg_ecg_segment_is_synchronous_and_512_hz():
    eeg = get_official_channels("PN00", "EEG").head(2)
    ecg = get_official_channels("PN00", "ECG")
    indices = list(eeg["edf_index_0based"]) + list(ecg["edf_index_0based"])
    data, fs = read_edf_segment(
        "PN00",
        "PN00-1.edf",
        indices,
        start_s=10.0,
        duration_s=5.0,
    )
    assert fs == 512.0
    assert data.shape == (4, 5 * 512)
    assert np.isfinite(data).all()
    assert np.std(data[-1]) > 0
    assert np.std(data[-2]) > 0
