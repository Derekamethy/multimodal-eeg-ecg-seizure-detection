"""Prespecified descriptive analyses. Never supplies model labels or thresholds."""
import json
import numpy as np
import pandas as pd

from .alarms import union_intervals, subtract_intervals, interval_seconds
from .metrics import uncertainty_intervals


def cells(frame, mask):
    duration = float(frame.recording_duration_s.iloc[0])
    return union_intervals([[t, min(t + 5, duration)] for t in frame.loc[mask, "anchor_s"]])


def intersect_seconds(intervals, start, end):
    return sum(max(0, min(b, end) - max(a, start)) for a, b in union_intervals(intervals))


def coverage(frame, events):
    duration = float(frame.recording_duration_s.iloc[0])
    uncertain = uncertainty_intervals(events)
    good = subtract_intervals(cells(frame, frame.ecg_available), uncertain)
    raw = [[0, duration]]
    uncertainty_s = duration - interval_seconds(subtract_intervals(raw, uncertain))
    startup_s = interval_seconds(subtract_intervals([[0, min(60, duration)]], uncertain))
    poor_s = interval_seconds(subtract_intervals(cells(frame, (frame.anchor_s >= 60) & ~frame.ecg_available), uncertain))
    exposure = interval_seconds(good)
    history_not_ready = interval_seconds(subtract_intervals(cells(frame, frame.ecg_available & ~frame.baseline_ready), uncertain))
    return dict(raw_s=duration, startup_s=startup_s, uncertain_s=uncertainty_s, poor_quality_s=poor_s,
        acquisition_missing_s=max(0., duration - startup_s - uncertainty_s - poor_s - exposure),
        baseline_not_ready_good_s=history_not_ready, evaluable_s=exposure, coverage_pct=100 * exposure / duration,
        good_window_fraction=float(frame.loc[frame.anchor_s >= 60, "ecg_available"].mean()),
        lead1_good_fraction=float(frame.loc[frame.anchor_s >= 60, "lead1_good"].mean()),
        lead2_good_fraction=float(frame.loc[frame.anchor_s >= 60, "lead2_good"].mean()),
        median_beat_agreement=float(frame.loc[frame.ecg_available, "beat_agreement"].median()),
        disagreement_windows=int((frame.selection_reason == "gross_interlead_disagreement").sum()),
        selected_lead1_windows=int((frame.selected_lead == 1).sum()), selected_lead2_windows=int((frame.selected_lead == 2).sum()))


def responses(frame, events):
    result = []
    good = frame.loc[frame.ecg_available & ~frame.annotation_uncertain]
    valid_intervals = subtract_intervals(cells(frame, frame.ecg_available), uncertainty_intervals(events))
    for e in events.itertuples():
        onset, offset = float(e.onset_relative_s), float(e.offset_relative_s_canonical)
        baseline = good.loc[(good.anchor_s >= onset - 300) & (good.anchor_s <= onset - 60)]
        # Require 120 s of quality-qualified support inside the fixed baseline interval.
        baseline_s = intersect_seconds([[t - 60, t] for t in baseline.anchor_s], max(0, onset - 360), onset - 60)
        ready = baseline_s >= 120
        pre = good.loc[(good.anchor_s >= onset - 5) & (good.anchor_s <= onset)]
        ictal = good.loc[(good.anchor_s >= onset) & (good.anchor_s < offset)]
        post = good.loc[(good.anchor_s >= offset) & (good.anchor_s < offset + 60)]
        def med(part, column):
            return float(part[column].median()) if len(part) else np.nan
        base_hr = med(baseline, "ecg__median_hr") if ready else np.nan
        ictal_hr = med(ictal, "ecg__median_hr")
        max_hr = float(ictal.ecg__median_hr.max()) if len(ictal) else np.nan
        base_rr = med(baseline, "ecg__median_rr") if ready else np.nan
        base_rmssd = med(baseline, "ecg__rmssd_s") if ready else np.nan
        base_cv = med(baseline, "ecg__rr_cv") if ready else np.nan
        result.append(dict(patient=e.patient, file_name=e.canonical_file_name, event_id=e.event_id,
            onset_s=onset, offset_s=offset, onset_type=e.onset_type, baseline_ready=ready,
            baseline_support_s=baseline_s, preictal_available=bool(len(pre)), ictal_available=bool(len(ictal)),
            analyzable=bool(ready and len(ictal)), ictal_valid_anchors=len(ictal),
            ictal_quality_coverage=intersect_seconds(valid_intervals, onset, offset) / (offset - onset),
            peri_event_quality_coverage=intersect_seconds(valid_intervals, max(0, onset - 60), min(frame.recording_duration_s.iloc[0], offset + 60)) /
                (min(frame.recording_duration_s.iloc[0], offset + 60) - max(0, onset - 60)),
            selected_leads_json=json.dumps(sorted(ictal.selected_lead.unique().astype(int).tolist())),
            baseline_hr=base_hr, preictal_hr=med(pre, "ecg__median_hr"), ictal_median_hr=ictal_hr,
            ictal_max_window_median_hr=max_hr, median_hr_change=ictal_hr-base_hr,
            max_hr_change=max_hr-base_hr, relative_hr_change=ictal_hr/base_hr-1 if base_hr > 0 else np.nan,
            baseline_rr=base_rr, ictal_rr=med(ictal, "ecg__median_rr"), rr_change=med(ictal, "ecg__median_rr")-base_rr,
            ictal_hr_slope_bpm_min=med(ictal, "ecg__hr_slope_bpm_min"), baseline_rmssd_s=base_rmssd,
            ictal_rmssd_s=med(ictal, "ecg__rmssd_s"), rmssd_change_s=med(ictal, "ecg__rmssd_s")-base_rmssd,
            baseline_rr_cv=base_cv, ictal_rr_cv=med(ictal, "ecg__rr_cv"), rr_cv_change=med(ictal, "ecg__rr_cv")-base_cv,
            postictal_hr=med(post, "ecg__median_hr")))
    return result


def patient_responses(table):
    rows = []
    for patient, group in table.groupby("patient"):
        a = group.loc[group.analyzable]
        rows.append(dict(patient=patient, events=len(group), analyzable_events=len(a),
            hr_increase_events=int((a.median_hr_change > 0).sum()),
            hr_increase_fraction=float((a.median_hr_change > 0).mean()) if len(a) else np.nan,
            baseline_hr_median=a.baseline_hr.median(), baseline_hr_min=a.baseline_hr.min(), baseline_hr_max=a.baseline_hr.max(),
            median_hr_change=a.median_hr_change.median(), min_hr_change=a.median_hr_change.min(), max_hr_change=a.median_hr_change.max(),
            median_rr_change=a.rr_change.median(), median_hr_slope=a.ictal_hr_slope_bpm_min.median(),
            median_rmssd_change=a.rmssd_change_s.median(), median_rr_cv_change=a.rr_cv_change.median()))
    return pd.DataFrame(rows)


def comparison(events_ecg, response_table, ecg_scores, eeg_dir):
    eeg_events = pd.read_csv(eeg_dir / "event_matches.csv")
    eeg_events = eeg_events.loc[eeg_events.model == "E1_RF", ["event_id", "detected", "delay_s"]].rename(columns={"detected":"eeg_detected", "delay_s":"eeg_delay_s"})
    ecg_events = events_ecg.loc[events_ecg.model == "C1_RF", ["event_id", "detected", "delay_s"]].rename(columns={"detected":"ecg_detected", "delay_s":"ecg_delay_s"})
    joined = response_table.merge(eeg_events, on="event_id", validate="one_to_one").merge(ecg_events, on="event_id", validate="one_to_one")
    joined["detection_group"] = np.select([joined.eeg_detected & joined.ecg_detected, joined.eeg_detected, joined.ecg_detected], ["both", "eeg_only", "ecg_only"], default="neither")
    return joined
