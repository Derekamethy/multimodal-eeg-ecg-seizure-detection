from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def bandpass_ecg(x: np.ndarray, fs: float, low_hz: float = 5.0, high_hz: float = 25.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("ECG input must be 1-D")
    if not np.isfinite(x).all():
        raise ValueError("ECG input contains non-finite values")
    nyq = fs / 2.0
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError("Invalid ECG bandpass")
    sos = butter(3, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    y = bandpass_ecg(x, fs)
    d = np.diff(y, prepend=y[0])
    energy = d * d
    win = max(1, int(round(0.12 * fs)))
    kernel = np.ones(win, dtype=float) / win
    integrated = np.convolve(energy, kernel, mode="same")

    med = np.median(integrated)
    mad = np.median(np.abs(integrated - med))
    scale = 1.4826 * mad
    threshold = med + 3.0 * scale
    distance = max(1, int(round(0.30 * fs)))
    candidates, _ = find_peaks(integrated, height=threshold, distance=distance)

    refined = []
    radius = max(1, int(round(0.10 * fs)))
    for p in candidates:
        lo = max(0, p - radius)
        hi = min(len(y), p + radius + 1)
        rp = lo + int(np.argmax(np.abs(y[lo:hi])))
        if not refined or rp - refined[-1] >= int(round(0.25 * fs)):
            refined.append(rp)
        elif abs(y[rp]) > abs(y[refined[-1]]):
            refined[-1] = rp
    return np.asarray(refined, dtype=int)
def rr_intervals_seconds(r_peaks: np.ndarray, fs: float, min_rr: float = 0.30, max_rr: float = 2.0) -> np.ndarray:
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 2:
        return np.asarray([], dtype=float)
    rr = np.diff(r_peaks) / float(fs)
    return rr[(rr >= min_rr) & (rr <= max_rr)]


def ecg_features(x: np.ndarray, fs: float) -> dict[str, float]:
    peaks = detect_r_peaks(x, fs)
    rr = rr_intervals_seconds(peaks, fs)
    if len(rr) < 2:
        return {
            "n_r_peaks": float(len(peaks)),
            "valid_rr_count": float(len(rr)),
            "mean_hr_bpm": np.nan,
            "mean_rr_s": np.nan,
            "sdnn_s": np.nan,
            "rmssd_s": np.nan,
            "pnn50": np.nan,
            "rr_cv": np.nan,
            "hr_trend_bpm_per_min": np.nan,
        }

    hr = 60.0 / rr
    drr = np.diff(rr)
    t_mid = np.cumsum(rr) - rr / 2.0
    if len(hr) >= 2 and np.ptp(t_mid) > 0:
        slope_bpm_per_s = np.polyfit(t_mid, hr, 1)[0]
        hr_trend = slope_bpm_per_s * 60.0
    else:
        hr_trend = np.nan

    return {
        "n_r_peaks": float(len(peaks)),
        "valid_rr_count": float(len(rr)),
        "mean_hr_bpm": float(np.mean(hr)),
        "mean_rr_s": float(np.mean(rr)),
        "sdnn_s": float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan,
        "rmssd_s": float(np.sqrt(np.mean(drr**2))) if len(drr) else np.nan,
        "pnn50": float(np.mean(np.abs(drr) > 0.050)) if len(drr) else np.nan,
        "rr_cv": float(np.std(rr, ddof=1) / np.mean(rr)) if len(rr) > 1 else np.nan,
        "hr_trend_bpm_per_min": float(hr_trend),
    }



def multichannel_ecg_features(data: np.ndarray, fs: float) -> dict[str, float]:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 1:
        raise ValueError("ECG data must have shape (channels, samples)")

    per_channel = [ecg_features(ch, fs) for ch in data]
    names = list(per_channel[0])
    out: dict[str, float] = {}
    for name in names:
        vals = np.asarray([row[name] for row in per_channel], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[f"{name}_median"] = float(np.median(finite)) if len(finite) else np.nan
        out[f"{name}_range"] = float(np.ptp(finite)) if len(finite) else np.nan

    if data.shape[0] >= 2:
        filtered = np.vstack([bandpass_ecg(ch, fs) for ch in data])
        corr = np.corrcoef(filtered)
        tri = np.abs(corr[np.triu_indices(data.shape[0], k=1)])
        out["ecg_channel_agreement"] = float(np.nanmean(tri))
    else:
        out["ecg_channel_agreement"] = np.nan
    return out
