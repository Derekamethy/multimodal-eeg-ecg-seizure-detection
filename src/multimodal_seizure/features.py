from __future__ import annotations

import numpy as np

from .data import get_official_channels, read_edf_segment
from .ecg import multichannel_ecg_features
from .eeg import eeg_features


def extract_feature_row(
    patient: str,
    file_name: str,
    anchor_s: float,
    *,
    eeg_context_s: float = 10.0,
    ecg_context_s: float = 60.0,
) -> dict[str, float]:
    eeg_channels = get_official_channels(patient, "EEG", file_name)
    ecg_channels = get_official_channels(patient, "ECG", file_name)

    eeg_indices = list(eeg_channels.edf_index_0based.astype(int))
    ecg_indices = list(ecg_channels.edf_index_0based.astype(int))

    eeg_data, eeg_fs = read_edf_segment(
        patient,
        file_name,
        eeg_indices,
        start_s=float(anchor_s - eeg_context_s),
        duration_s=float(eeg_context_s),
    )
    ecg_data, ecg_fs = read_edf_segment(
        patient,
        file_name,
        ecg_indices,
        start_s=float(anchor_s - ecg_context_s),
        duration_s=float(ecg_context_s),
    )
    if eeg_fs != ecg_fs:
        raise ValueError(f"EEG/ECG sample-rate mismatch: {eeg_fs} vs {ecg_fs}")
    eeg = eeg_features(eeg_data, eeg_fs)
    ecg = multichannel_ecg_features(ecg_data, ecg_fs)

    row: dict[str, float] = {
        "anchor_s": float(anchor_s),
        "sampling_rate_hz": float(eeg_fs),
        "n_eeg_channels": float(eeg_data.shape[0]),
        "n_ecg_channels": float(ecg_data.shape[0]),
    }
    row.update({f"eeg__{k}": float(v) for k, v in eeg.items()})
    row.update({f"ecg__{k}": float(v) for k, v in ecg.items()})
    return row


def feature_columns_by_modality(columns: list[str]) -> tuple[list[str], list[str]]:
    eeg = [c for c in columns if c.startswith("eeg__")]
    ecg = [c for c in columns if c.startswith("ecg__")]
    return eeg, ecg
