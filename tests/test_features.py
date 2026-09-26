import numpy as np

from multimodal_seizure.features import extract_feature_row, feature_columns_by_modality


def test_real_multimodal_feature_row_has_both_modalities():
    row = extract_feature_row("PN00", "PN00-1.edf", anchor_s=120.0)
    eeg, ecg = feature_columns_by_modality(list(row))
    assert len(eeg) >= 40
    assert len(ecg) >= 10
    assert row["sampling_rate_hz"] == 512.0
    assert row["n_ecg_channels"] == 2.0
    assert np.isfinite([row[k] for k in eeg]).all()
