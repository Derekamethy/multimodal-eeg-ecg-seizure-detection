"""Decision-causal rhythm features; no beat annotations or clinical SQI claims.

Each window is independently filtered forward and analyzed at its END. Window
thresholds can revise beat estimates across overlapping windows; this is not a
sample-by-sample online beat detector. No sample after decision t is accessed.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, lfilter, sosfilt

RULES = dict(context_s=60, baseline_start_s=360, baseline_end_s=60, baseline_good_s=120,
    band_hz=[5, 25], butter_order=3, energy_integrator_s=.12, threshold_mad=3,
    candidate_distance_s=.30, refinement_lookback_s=.16, refined_distance_s=.30, edge_start_s=.5, edge_end_s=.15,
    rr_min_s=.30, rr_max_s=2., min_valid_rr=20, min_valid_fraction=.8,
    min_rr_coverage=.75, min_raw_std_uv=1., min_filtered_std_uv=.5,
    max_rail_fraction=.01, min_peak_contrast=3., disagreement_hr_bpm=10.,
    disagreement_hr_fraction=.15, beat_tolerance_s=.10, min_beat_agreement=.70)
FEATURES = ("median_hr", "mean_hr", "median_rr", "mean_rr", "hr_slope_bpm_min",
    "early_late_hr", "sdrr_s", "rmssd_s", "rr_cv", "relative_hr_delta",
    "relative_hr_ratio", "relative_rr_delta")


def rr_series(peaks, fs, start_s=0.):
    peaks = np.asarray(peaks, dtype=int)
    if fs <= 0 or np.any(np.diff(peaks) <= 0):
        raise ValueError("Peaks must increase strictly and sampling rate must be positive")
    rr = np.diff(peaks) / fs
    valid = (rr >= RULES["rr_min_s"]) & (rr <= RULES["rr_max_s"])
    return dict(rr=rr, valid=valid, hr=np.divide(60., rr, out=np.full_like(rr, np.nan), where=rr > 0),
                time_s=start_s + (peaks[:-1] + peaks[1:]) / (2 * fs))


def rhythm_features(series, start_s, duration_s=60):
    rr, mask = series["rr"], series["valid"]
    hr, times = series["hr"][mask], series["time_s"][mask]
    good = rr[mask]
    out = {name: np.nan for name in FEATURES[:9]}
    if not len(good):
        return out
    adjacent = mask[:-1] & mask[1:]
    differences = np.diff(rr)[adjacent]
    early, late = hr[times < start_s + duration_s / 2], hr[times >= start_s + duration_s / 2]
    out.update(median_hr=float(np.median(hr)), mean_hr=float(np.mean(hr)),
        median_rr=float(np.median(good)), mean_rr=float(np.mean(good)),
        hr_slope_bpm_min=float(np.polyfit(times - start_s, hr, 1)[0] * 60) if len(hr) >= 2 and np.ptp(times) else np.nan,
        early_late_hr=float(np.median(late) - np.median(early)) if len(early) and len(late) else np.nan,
        sdrr_s=float(np.std(good, ddof=1)) if len(good) > 1 else np.nan,
        rmssd_s=float(np.sqrt(np.mean(differences ** 2))) if len(differences) else np.nan,
        rr_cv=float(np.std(good, ddof=1) / np.mean(good)) if len(good) > 1 else np.nan)
    return out


def analyze_lead(x, fs=512., start_s=0., rails=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or fs <= 50 or len(x) < fs:
        raise ValueError("Expected >=1 second of one ECG lead at fs>50 Hz")
    duration = len(x) / fs
    finite = float(np.isfinite(x).mean())
    evidence = dict(finite_fraction=finite, raw_std_uv=np.nan, filtered_std_uv=np.nan,
        rail_fraction=np.nan, n_peaks=0, valid_rr_count=0, valid_rr_fraction=0.,
        rr_coverage=0., peak_contrast=0., median_hr=np.nan, good=False, reason="nonfinite")
    peaks = np.array([], dtype=int)
    if finite != 1:
        return dict(evidence=evidence, peaks=peaks, series=rr_series(peaks, fs, start_s))
    evidence["raw_std_uv"] = float(np.std(x))
    evidence["rail_fraction"] = float(np.mean((x <= rails[0]) | (x >= rails[1]))) if rails is not None else 0.
    y = sosfilt(butter(3, RULES["band_hz"], fs=fs, btype="bandpass", output="sos"), x - np.median(x))
    evidence["filtered_std_uv"] = float(np.std(y))
    d = np.diff(y, prepend=y[0])
    width = max(1, round(RULES["energy_integrator_s"] * fs))
    energy = lfilter(np.ones(width) / width, [1.], d * d)
    median = np.median(energy)
    threshold = median + RULES["threshold_mad"] * 1.4826 * np.median(np.abs(energy - median))
    candidates, _ = find_peaks(energy, height=threshold, distance=round(RULES["candidate_distance_s"] * fs))
    refined = []
    for p in candidates:
        low = max(0, p - round(RULES["refinement_lookback_s"] * fs))
        peak = low + int(np.argmax(np.abs(y[low:p + 1])))
        if peak < RULES["edge_start_s"] * fs or peak >= len(x) - RULES["edge_end_s"] * fs:
            continue
        if not refined or peak - refined[-1] >= round(RULES["refined_distance_s"] * fs):
            refined.append(peak)
        elif peak > refined[-1] and abs(y[peak]) > abs(y[refined[-1]]):
            refined[-1] = peak
    peaks = np.asarray(refined, dtype=int)
    series = rr_series(peaks, fs, start_s)
    valid = series["valid"]
    background = 1.4826 * np.median(np.abs(y - np.median(y)))
    contrast = float(np.median(np.abs(y[peaks])) / max(background, 1e-9)) if len(peaks) else 0.
    evidence.update(n_peaks=len(peaks), valid_rr_count=int(valid.sum()),
        valid_rr_fraction=float(valid.mean()) if len(valid) else 0.,
        rr_coverage=float(series["rr"][valid].sum() / duration), peak_contrast=contrast,
        median_hr=float(np.median(series["hr"][valid])) if valid.any() else np.nan)
    failures = []
    checks = ((evidence["raw_std_uv"] >= RULES["min_raw_std_uv"], "flat_or_low_variation"),
        (evidence["filtered_std_uv"] >= RULES["min_filtered_std_uv"], "low_qrs_variation"),
        (evidence["rail_fraction"] <= RULES["max_rail_fraction"], "saturation"),
        (evidence["valid_rr_count"] >= RULES["min_valid_rr"], "few_beats"),
        (evidence["valid_rr_fraction"] >= RULES["min_valid_fraction"], "invalid_rr"),
        (evidence["rr_coverage"] >= RULES["min_rr_coverage"], "rr_gaps"),
        (contrast >= RULES["min_peak_contrast"], "low_peak_contrast"))
    failures = [reason for passed, reason in checks if not passed]
    evidence.update(good=not failures, reason="|".join(failures) or "good")
    return dict(evidence=evidence, peaks=peaks, series=series)


def beat_agreement(first, second, fs=512.):
    a, b = first["peaks"], second["peaks"]
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= RULES["beat_tolerance_s"] * fs:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return 2 * matched / (len(a) + len(b)) if len(a) + len(b) else np.nan


def choose_lead(leads, fs=512.):
    if len(leads) != 2:
        raise ValueError("Exactly the two official ECG leads are required")
    good = [i for i, lead in enumerate(leads) if lead["evidence"]["good"]]
    agreement = beat_agreement(*leads, fs)
    if not good:
        return None, "both_poor", agreement
    if len(good) == 1:
        return good[0], f"only_lead_{good[0] + 1}_good", agreement
    h1, h2 = [lead["evidence"]["median_hr"] for lead in leads]
    if abs(h1 - h2) > max(RULES["disagreement_hr_bpm"], RULES["disagreement_hr_fraction"] * min(h1, h2)) or agreement < RULES["min_beat_agreement"]:
        return None, "gross_interlead_disagreement", agreement
    selected = max(good, key=lambda i: (leads[i]["evidence"]["valid_rr_fraction"],
        leads[i]["evidence"]["rr_coverage"], leads[i]["evidence"]["peak_contrast"], -i))
    return selected, "both_good_agree_lexicographic_quality", agreement


def analyze_pair(data, fs=512., start_s=0., rails=(None, None)):
    if np.asarray(data).ndim != 2 or len(data) != 2 or len(rails) != 2:
        raise ValueError("Exactly two official ECG waveforms and rail limits are required")
    leads = [analyze_lead(x, fs, start_s, limit) for x, limit in zip(data, rails)]
    selected, reason, agreement = choose_lead(leads, fs)
    result = dict(leads=leads, selected=selected, reason=reason, agreement=agreement, features={})
    if selected is not None:
        result["features"] = rhythm_features(leads[selected]["series"], start_s)
    return result


def relative_features(current, history, t):
    # Five disjoint historical 60-s windows exactly tile [t-360,t-60).
    blocks = [history[u] for u in range(int(t) - 300, int(t), 60)
              if u >= 60 and u in history and history[u]["selected"] is not None]
    good_s = 60 * len(blocks)
    ready = good_s >= RULES["baseline_good_s"]
    out = dict(relative_hr_delta=np.nan, relative_hr_ratio=np.nan, relative_rr_delta=np.nan)
    baseline = dict(baseline_ready=ready, baseline_good_s=good_s, baseline_hr=np.nan, baseline_rr=np.nan)
    if ready:
        valid_rr = np.concatenate([b["leads"][b["selected"]]["series"]["rr"][b["leads"][b["selected"]]["series"]["valid"]] for b in blocks])
        baseline.update(baseline_hr=float(np.median(60 / valid_rr)), baseline_rr=float(np.median(valid_rr)))
        if current["selected"] is not None:
            f = current["features"]
            out.update(relative_hr_delta=f["median_hr"] - baseline["baseline_hr"],
                relative_hr_ratio=f["median_hr"] / baseline["baseline_hr"],
                relative_rr_delta=f["median_rr"] - baseline["baseline_rr"])
    return out, baseline
