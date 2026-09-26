import numpy as np

from multimodal_seizure.data import get_official_channels, read_edf_segment, read_event_table
from multimodal_seizure.eeg import channel_features, eeg_features


def test_10_hz_sine_is_alpha_dominant():
    fs = 512.0
    t = np.arange(int(10 * fs)) / fs
    x = np.sin(2 * np.pi * 10.0 * t)
    f = channel_features(x, fs)
    assert 9.5 <= f["dominant_frequency"] <= 10.5
    assert f["alpha_relative_power"] > 0.90
    assert f["spectral_entropy"] < 0.35


def test_multichannel_synchrony_detects_identical_signals():
    fs = 512.0
    t = np.arange(int(5 * fs)) / fs
    x = np.sin(2 * np.pi * 8.0 * t)
    data = np.vstack([x, x, x])
    f = eeg_features(data, fs)
    assert f["mean_abs_channel_corr"] > 0.999
    assert f["std_abs_channel_corr"] < 1e-6
def test_real_siena_eeg_segment_returns_finite_features():
    events = read_event_table()
    for patient in ["PN00", "PN06", "PN10", "PN12", "PN14"]:
        ev = events[events.patient == patient].iloc[0]
        eeg = get_official_channels(patient, "EEG")
        indices = list(eeg.edf_index_0based.astype(int))
        start = max(0.0, float(ev.onset_relative_s) - 30.0)
        data, fs = read_edf_segment(
            patient, ev.canonical_file_name, indices, start_s=start, duration_s=10.0
        )
        f = eeg_features(data, fs)
        assert fs == 512.0
        assert len(f) >= 40
        values = np.asarray(list(f.values()), dtype=float)
        assert np.isfinite(values).all()
        assert 0.0 <= f["mean_abs_channel_corr"] <= 1.0
        assert 0.0 <= f["spectral_entropy_mean"] <= 1.0
