import numpy as np

from multimodal_seizure.data import get_official_channels, read_edf_segment, read_event_table
from multimodal_seizure.ecg import detect_r_peaks, ecg_features, rr_intervals_seconds


def synthetic_ecg(fs=512.0, duration_s=60.0, hr_bpm=60.0):
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    x = 0.02 * np.sin(2 * np.pi * 0.5 * t)
    rr = 60.0 / hr_bpm
    r_times = np.arange(1.0, duration_s - 0.5, rr)
    sigma = 0.015
    for rt in r_times:
        x += np.exp(-0.5 * ((t - rt) / sigma) ** 2)
        x -= 0.15 * np.exp(-0.5 * ((t - (rt - 0.04)) / 0.012) ** 2)
        x -= 0.20 * np.exp(-0.5 * ((t - (rt + 0.05)) / 0.015) ** 2)
    return x, r_times


def test_r_peak_detector_recovers_regular_60_bpm_signal():
    fs = 512.0
    x, expected = synthetic_ecg(fs=fs, duration_s=60.0, hr_bpm=60.0)
    peaks = detect_r_peaks(x, fs)
    detected = peaks / fs
    assert abs(len(detected) - len(expected)) <= 1
    for rt in expected[1:-1]:
        assert np.min(np.abs(detected - rt)) < 0.08
def test_ecg_features_on_regular_signal_are_physically_correct():
    fs = 512.0
    x, _ = synthetic_ecg(fs=fs, duration_s=120.0, hr_bpm=75.0)
    f = ecg_features(x, fs)
    assert 73.0 <= f["mean_hr_bpm"] <= 77.0
    assert 0.78 <= f["mean_rr_s"] <= 0.82
    assert f["sdnn_s"] < 0.02
    assert f["rmssd_s"] < 0.02
    assert f["pnn50"] < 0.05


def test_rr_filter_rejects_implausible_intervals():
    fs = 100.0
    peaks = np.array([0, 20, 100, 200, 450])
    rr = rr_intervals_seconds(peaks, fs)
    assert np.allclose(rr, [0.8, 1.0])


def test_real_siena_ecg_segment_produces_plausible_features():
    events = read_event_table()
    for patient in ["PN00", "PN06", "PN10", "PN12", "PN14"]:
        ev = events[events.patient == patient].iloc[0]
        ecg = get_official_channels(patient, "ECG")
        idx = int(ecg.iloc[0].edf_index_0based)
        start = max(0.0, float(ev.onset_relative_s) - 120.0)
        data, fs = read_edf_segment(
            patient, ev.canonical_file_name, [idx], start_s=start, duration_s=60.0
        )
        f = ecg_features(data[0], fs)
        assert f["valid_rr_count"] >= 20
        assert 30.0 <= f["mean_hr_bpm"] <= 220.0
        assert 0.25 <= f["mean_rr_s"] <= 2.0



def test_multichannel_ecg_features_are_stable_on_two_identical_channels():
    from multimodal_seizure.ecg import multichannel_ecg_features

    fs = 512.0
    x, _ = synthetic_ecg(fs=fs, duration_s=60.0, hr_bpm=72.0)
    f = multichannel_ecg_features(np.vstack([x, x]), fs)
    assert 70.0 <= f["mean_hr_bpm_median"] <= 74.0
    assert f["mean_hr_bpm_range"] < 1e-6
    assert f["ecg_channel_agreement"] > 0.999
