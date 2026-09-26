from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .data import read_event_table

DEFAULT_STRIDE_S = 5.0
DEFAULT_EEG_CONTEXT_S = 10.0
DEFAULT_ECG_CONTEXT_S = 60.0
DEFAULT_EXCLUSION_S = 300.0


def build_window_index(
    patient: str,
    file_name: str,
    *,
    stride_s: float = DEFAULT_STRIDE_S,
    eeg_context_s: float = DEFAULT_EEG_CONTEXT_S,
    ecg_context_s: float = DEFAULT_ECG_CONTEXT_S,
    exclusion_s: float = DEFAULT_EXCLUSION_S,
) -> pd.DataFrame:
    if stride_s <= 0 or min(eeg_context_s, ecg_context_s) <= 0 or exclusion_s < 0:
        raise ValueError("Invalid window context, stride or exclusion")
    events = read_event_table(include_quarantined=True)
    ev = events[(events.patient == patient) & (events.canonical_file_name == file_name)].copy()
    if ev.empty:
        raise ValueError(f"No seizure annotations for {patient}/{file_name}")
    if not ev.primary_eligible.all():
        raise ValueError(f"Quarantined parent EDF: {patient}/{file_name}; no primary windows/background allowed")

    duration_s = float(ev.edf_duration_s.iloc[0])
    # Decisions follow EEG availability. ECG history never delays EEG inference.
    anchors = np.arange(eeg_context_s, duration_s, stride_s)
    rows = []
    uncertain = [interval for raw in ev.uncertain_intervals_json for interval in json.loads(raw)]
    for t in anchors:
        positive_events = ev[
            (ev.onset_relative_s <= t)
            & (t < ev.offset_relative_s_canonical)
        ]
        if any(start <= t < end for start, end in uncertain):
            label, status, event_numbers = -1, "annotation_uncertain", ""
        elif not positive_events.empty:
            label = 1
            status = "positive"
            event_numbers = "|".join(str(int(x)) for x in positive_events.seizure_number)
        else:
            near = ev[
                ((ev.onset_relative_s - exclusion_s) <= t)
                & (t < (ev.offset_relative_s_canonical + exclusion_s))
            ]
            if not near.empty:
                label = -1
                status = "ignore"
                event_numbers = ""
            else:
                label = 0
                status = "negative"
                event_numbers = ""

        rows.append({
            "patient": patient,
            "file_name": file_name,
            "anchor_s": float(t),
            "eeg_start_s": float(t - eeg_context_s),
            "eeg_end_s": float(t),
            "ecg_start_s": float(t - ecg_context_s) if t >= ecg_context_s else np.nan,
            "ecg_end_s": float(t),
            "eeg_available": True,
            "ecg_history_available": bool(t >= ecg_context_s),
            "annotation_uncertain": status == "annotation_uncertain",
            "training_eligible": label >= 0,
            "evaluation_label": -1 if status == "annotation_uncertain" else int(not positive_events.empty),
            "recording_duration_s": duration_s,
            "label": int(label),
            "status": status,
            "seizure_numbers": event_numbers,
        })
    return pd.DataFrame(rows)


def build_patient_folds(patient: str) -> list[dict[str, object]]:
    events = read_event_table()
    files = sorted(events.loc[events.patient == patient, "canonical_file_name"].unique())
    return [
        {
            "patient": patient,
            "fold": i,
            "train_files": [x for x in files if x != test_file],
            "test_file": test_file,
        }
        for i, test_file in enumerate(files)
    ]
