"""Frozen declaration-time alarm policy on one parent recording at a time."""
from __future__ import annotations

import numpy as np


def union_intervals(intervals):
    result = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def subtract_intervals(intervals, excluded):
    result = union_intervals(intervals)
    for a, b in union_intervals(excluded):
        pieces = []
        for start, end in result:
            if b <= start or a >= end:
                pieces.append([start, end])
            else:
                if start < a:
                    pieces.append([start, a])
                if b < end:
                    pieces.append([b, end])
        result = pieces
    return result


def interval_seconds(intervals):
    return sum(b - a for a, b in union_intervals(intervals))


def alarm_stream(timeline, duration_s, threshold, uncertain=(), acquisition_gaps=(), stride_s=5.0):
    """Scores are held for at most one stride; missing cells are unavailable.

    Uncertainty masks reference knowledge, not EEG availability. Barriers end
    alarms at their exact onset, including barriers between two decisions.
    """
    if "file_name" in timeline and timeline.file_name.nunique() > 1:
        raise ValueError("Alarm state cannot span parent EDFs")
    times = timeline.anchor_s.to_numpy(float)
    if len(times) and (np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] >= duration_s):
        raise ValueError("Decisions must be strictly ordered inside [0, duration)")
    scores = timeline.score.to_numpy(float)
    available = timeline.eeg_available.to_numpy(bool) & np.isfinite(scores)
    if np.any((scores[available] < 0) | (scores[available] > 1)):
        raise ValueError("Scores must be probabilities in [0, 1]")
    known_barriers = union_intervals([*uncertain, *acquisition_gaps])
    cells = [[t, min(t + stride_s, duration_s)] for t, ok in zip(times, available) if ok]
    sensor_intervals = subtract_intervals(cells, acquisition_gaps)
    exposure = subtract_intervals(sensor_intervals, uncertain)
    alarms, active, below = [], None, 0
    last_time = None

    def close(end, reason):
        nonlocal active, below
        if active is not None:
            alarms.append(dict(declaration_s=active, end_s=float(end), termination_reason=reason))
        active, below = None, 0

    for t, score, ok in zip(times, scores, available):
        if last_time is not None:
            cuts = [(a, "annotation_or_acquisition_gap") for a, b in known_barriers if a <= t and b > last_time]
            if t > last_time + stride_s + 1e-8:
                cuts.append((last_time + stride_s, "missing_decision"))
            if cuts:
                cut, reason = min(cuts)
                close(max(last_time, cut), reason)
        excluded = any(a <= t < b for a, b in known_barriers)
        if not ok or excluded:
            close(t, "unavailable" if not ok else "annotation_or_acquisition_gap")
        elif score >= threshold:
            if active is None:
                active = float(t)
            below = 0
        elif active is not None:
            below += 1
            if below == 2:
                close(t, "two_below")
        last_time = float(t)
    if active is not None:
        end = min(last_time + stride_s, duration_s)
        cuts = [a for a, b in known_barriers if last_time < a < end]
        if cuts:
            close(min(cuts), "annotation_or_acquisition_gap")
        else:
            close(end, "recording_end" if end == duration_s else "missing_decision")
    return alarms, exposure, sensor_intervals
