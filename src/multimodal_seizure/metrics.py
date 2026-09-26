"""Strict new-declaration matching; overlap is a separately labeled diagnostic."""
from __future__ import annotations

import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .alarms import alarm_stream, interval_seconds, subtract_intervals


def uncertainty_intervals(events):
    return [interval for raw in events.uncertain_intervals_json for interval in json.loads(raw)] if "uncertain_intervals_json" in events else []


def summarize(metrics, matches):
    sum_fields = ("events", "detected", "overlap_detected", "false_declarations", "duplicate_declarations",
                  "total_declarations", "active_alarm_s", "raw_s", "startup_s", "uncertain_s",
                  "other_unavailable_s", "evaluable_s")
    out = {name: sum(m[name] for m in metrics) for name in sum_fields}
    delays = [m["delay_s"] for m in matches if m["detected"]]
    out.update(sensitivity=out["detected"] / out["events"] if out["events"] else np.nan,
               overlap_sensitivity=out["overlap_detected"] / out["events"] if out["events"] else np.nan,
               far_per_h=out["false_declarations"] * 3600 / out["evaluable_s"] if out["evaluable_s"] else np.nan,
               missed=out["events"] - out["detected"], matched_declarations=out["detected"],
               delay_n=len(delays), delay_median_s=float(np.median(delays)) if delays else np.nan,
               delay_q25_s=float(np.percentile(delays, 25)) if delays else np.nan,
               delay_q75_s=float(np.percentile(delays, 75)) if delays else np.nan,
               delay_min_s=min(delays) if delays else np.nan, delay_max_s=max(delays) if delays else np.nan,
               warning_fraction=out["active_alarm_s"] / out["evaluable_s"] if out["evaluable_s"] else np.nan,
               longest_alarm_s=max((m["longest_alarm_s"] for m in metrics), default=0))
    return out


def evaluate_record(timeline, events, duration_s, threshold, *, startup_s=10.0, stride_s=5.0, acquisition_gaps=()):
    uncertain = uncertainty_intervals(events)
    alarms, exposure, sensor = alarm_stream(timeline, duration_s, threshold, uncertain, acquisition_gaps, stride_s)
    refs = sorted(events.to_dict("records"), key=lambda e: (e["onset_relative_s"], e["offset_relative_s_canonical"]))
    matches = [dict(event_id=e["event_id"], onset_s=float(e["onset_relative_s"]),
                    offset_s=float(e["offset_relative_s_canonical"]), detected=False,
                    declaration_s=np.nan, delay_s=np.nan, overlap_detected=False) for e in refs]
    for alarm in alarms:
        candidates = [m for m in matches if m["onset_s"] <= alarm["declaration_s"] < m["offset_s"]]
        unmatched = [m for m in candidates if not m["detected"]]
        alarm.update(match_status="false", event_id="")
        if unmatched:
            match = unmatched[0]
            match.update(detected=True, declaration_s=alarm["declaration_s"], delay_s=alarm["declaration_s"] - match["onset_s"])
            alarm.update(match_status="matched", event_id=match["event_id"])
        elif candidates:
            alarm.update(match_status="duplicate", event_id=candidates[0]["event_id"])
        for match in matches:
            if alarm["declaration_s"] < match["offset_s"] and alarm["end_s"] > match["onset_s"]:
                match["overlap_detected"] = True
    raw = [[0, duration_s]]
    startup = min(startup_s, duration_s)
    uncertain_s = duration_s - interval_seconds(subtract_intervals(raw, uncertain))
    startup_only = interval_seconds(subtract_intervals([[0, startup]], uncertain))
    evaluable_s = interval_seconds(exposure)
    base = dict(events=len(matches), detected=sum(m["detected"] for m in matches),
                overlap_detected=sum(m["overlap_detected"] for m in matches),
                # Duplicates are unmatched declarations in the requested primary FAR.
                false_declarations=sum(a["match_status"] != "matched" for a in alarms),
                duplicate_declarations=sum(a["match_status"] == "duplicate" for a in alarms),
                total_declarations=len(alarms), active_alarm_s=sum(a["end_s"] - a["declaration_s"] for a in alarms),
                raw_s=float(duration_s), startup_s=startup_only, uncertain_s=uncertain_s,
                other_unavailable_s=duration_s - startup_only - uncertain_s - evaluable_s,
                evaluable_s=evaluable_s, longest_alarm_s=max((a["end_s"] - a["declaration_s"] for a in alarms), default=0))
    metrics = summarize([base], matches)
    return metrics, alarms, matches, exposure


def window_metrics(frame):
    keep = frame.eeg_available & np.isfinite(frame.score) & (frame.evaluation_label >= 0)
    y, p = frame.loc[keep, "evaluation_label"], frame.loc[keep, "score"]
    return dict(window_n=len(y), window_positive=int(y.sum()),
                average_precision=float(average_precision_score(y, p)) if y.nunique() == 2 else np.nan,
                roc_auc=float(roc_auc_score(y, p)) if y.nunique() == 2 else np.nan)
