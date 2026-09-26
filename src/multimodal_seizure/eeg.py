from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _band_integral(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    if lo < freqs[0] or hi > freqs[-1] or lo >= hi:
        return 0.0
    # Integrate the continuous piecewise-linear PSD, including both edges.
    # Excluding the upper sample lost one trapezoid at every band boundary.
    interior = (freqs > lo) & (freqs < hi)
    f = np.r_[lo, freqs[interior], hi]
    p = np.r_[np.interp(lo, freqs, psd), psd[interior], np.interp(hi, freqs, psd)]
    return float(np.trapezoid(p, f))


def channel_features(x: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(fs) or fs < 90:
        raise ValueError("EEG spectral features require a finite sampling rate >= 90 Hz")
    if x.ndim != 1 or len(x) < int(fs):
        raise ValueError("EEG channel must be 1-D and at least 1 second long")
    if not np.isfinite(x).all():
        raise ValueError("EEG contains non-finite values")

    centered = x - np.mean(x)
    nperseg = min(len(x), max(256, int(round(4 * fs))))
    freqs, psd = welch(centered, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= 1.0) & (freqs <= 45.0)
    f = freqs[mask]
    p = psd[mask]
    total = float(np.trapezoid(p, f))
    pnorm = p / np.sum(p) if np.sum(p) > 0 else np.zeros_like(p)

    mean_freq = float(np.sum(f * pnorm)) if np.sum(pnorm) else np.nan
    bandwidth = float(np.sqrt(np.sum(((f - mean_freq) ** 2) * pnorm))) if np.sum(pnorm) else np.nan
    dominant = float(f[np.argmax(p)]) if np.sum(p) > 0 else np.nan
    cdf = np.cumsum(pnorm)
    edge95 = float(f[np.searchsorted(cdf, 0.95, side="left")]) if np.sum(pnorm) else np.nan
    nz = pnorm[pnorm > 0]
    entropy = float(-np.sum(nz * np.log2(nz)) / np.log2(len(pnorm))) if len(nz) and len(pnorm) > 1 else np.nan
    dx = np.diff(centered)
    ddx = np.diff(dx)
    var0 = float(np.var(centered))
    var1 = float(np.var(dx))
    var2 = float(np.var(ddx))
    mobility = float(np.sqrt(var1 / var0)) if var0 > 0 else 0.0
    complexity = float(np.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0.0

    out = {
        "rms": float(np.sqrt(np.mean(centered ** 2))),
        "std": float(np.std(centered)),
        "line_length": float(np.mean(np.abs(dx))),
        "hjorth_activity": var0,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "total_power_1_45": total,
        "mean_frequency": mean_freq,
        "dominant_frequency": dominant,
        "spectral_bandwidth": bandwidth,
        "spectral_edge_95": edge95,
        "spectral_entropy": entropy,
    }
    for name, (lo, hi) in BANDS.items():
        bp = _band_integral(f, p, lo, hi)
        out[f"{name}_power"] = bp
        out[f"{name}_relative_power"] = bp / total if total > 0 else np.nan
    return out


def eeg_features(data: np.ndarray, fs: float) -> dict[str, float]:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 1:
        raise ValueError("EEG data must have shape (channels, samples)")
    per_channel = [channel_features(ch, fs) for ch in data]
    names = list(per_channel[0])
    out: dict[str, float] = {}
    for name in names:
        vals = np.asarray([row[name] for row in per_channel], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[f"{name}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
        out[f"{name}_std_across_channels"] = float(np.std(finite)) if len(finite) else np.nan

    nonflat = data[np.std(data, axis=1) > 0]
    if nonflat.shape[0] >= 2:
        corr = np.corrcoef(nonflat)
        tri = np.abs(corr[np.triu_indices(nonflat.shape[0], k=1)])
        out["mean_abs_channel_corr"] = float(np.nanmean(tri))
        out["std_abs_channel_corr"] = float(np.nanstd(tri))
    else:
        out["mean_abs_channel_corr"] = np.nan
        out["std_abs_channel_corr"] = np.nan
    return out
