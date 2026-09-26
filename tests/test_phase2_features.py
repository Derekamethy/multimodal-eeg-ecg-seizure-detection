import numpy as np
import pytest

from multimodal_seizure.eeg import BANDS, _band_integral, channel_features, eeg_features
from multimodal_seizure.windows import build_window_index


def test_timeline_uses_eeg_history_and_retains_training_ignored_anchors():
    frame = build_window_index("PN10", "PN10-2.edf")
    assert frame.anchor_s.iloc[0] == 10
    assert (np.diff(frame.anchor_s) == 5).all()
    early = frame.loc[frame.anchor_s < 60]
    assert early.eeg_available.all() and not early.ecg_history_available.any()
    assert ((frame.label == -1) & (frame.evaluation_label == 0)).any()
    uncertain = frame.loc[(frame.anchor_s >= 7828) & (frame.anchor_s < 7849)]
    assert uncertain.annotation_uncertain.all() and not uncertain.training_eligible.any()
    assert uncertain.eeg_available.all()  # annotation is not sensor loss


def test_spectral_integrals_cover_all_boundaries_and_preserve_units():
    f = np.arange(1, 45.25, .25)
    p = np.ones_like(f) * 2
    assert sum(_band_integral(f, p, a, b) for a, b in BANDS.values()) == 88
    assert _band_integral(f, p, 4.1, 8.1) == pytest.approx(8)
    fs = 512
    t = np.arange(10 * fs) / fs
    sine = channel_features(np.sin(2 * np.pi * 10 * t), fs)
    doubled = channel_features(2 * np.sin(2 * np.pi * 10 * t), fs)
    assert sine["total_power_1_45"] == pytest.approx(.5, abs=.001)
    assert doubled["total_power_1_45"] == pytest.approx(4 * sine["total_power_1_45"])
    assert sum(sine[f"{band}_relative_power"] for band in BANDS) == pytest.approx(1)
    assert 0 <= sine["spectral_entropy"] <= 1
    with pytest.raises(ValueError, match="sampling rate"):
        channel_features(np.ones(100), 50)


def test_flat_channels_and_correlation_are_defined_without_fake_frequency():
    flat = np.ones(5120)
    f = eeg_features(np.vstack([flat, flat]), 512)
    assert f["total_power_1_45_mean"] == 0
    assert np.isnan(f["dominant_frequency_mean"])
    assert np.isnan(f["mean_abs_channel_corr"])
    t = np.arange(5120) / 512
    wave = np.sin(2 * np.pi * 10 * t)
    f = eeg_features(np.vstack([flat, wave, -wave]), 512)
    assert f["mean_abs_channel_corr"] == pytest.approx(1)
