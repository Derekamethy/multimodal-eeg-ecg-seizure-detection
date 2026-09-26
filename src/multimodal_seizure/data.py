from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyedflib
from .edf import read_header
from .inventory import PROJECT_ROOT, RAW_ROOT, PILOT_PATIENTS
from .provenance import metadata_identity
from .siena import channel_rows

CHANNEL_MAP = PROJECT_ROOT / "metadata/canonical_channels.csv"
EVENT_TABLE = PROJECT_ROOT / "metadata/canonical_events.csv"


def read_event_table(*, include_quarantined: bool = False) -> pd.DataFrame:
    metadata_identity()
    events = pd.read_csv(EVENT_TABLE)
    return events if include_quarantined else events.loc[events.primary_eligible].copy()


def get_official_channels(patient: str, modality: str, file_name: str | None = None) -> pd.DataFrame:
    if patient not in PILOT_PATIENTS:
        raise ValueError(f"Unknown pilot patient: {patient}")
    modality = modality.upper()
    if modality not in {"EEG", "ECG"}:
        raise ValueError("modality must be EEG or ECG")
    paths = [_edf_path(patient, file_name)] if file_name else sorted((RAW_ROOT / patient).glob("*.edf"))
    mappings = [channel_rows(read_header(path), RAW_ROOT / patient / f"Seizures-list-{patient}.txt") for path in paths]
    if not mappings or any(rows != mappings[0] for rows in mappings[1:]):
        raise ValueError("Missing or recording-dependent channel mapping; supply an explicit parent EDF")
    df = pd.DataFrame(mappings[0])
    out = df.loc[df.modality == modality].copy()
    out = out.sort_values("channel_number_1based").reset_index(drop=True)
    return out
def _edf_path(patient: str, file_name: str) -> Path:
    path = RAW_ROOT / patient / file_name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_edf_segment(
    patient: str,
    file_name: str,
    channel_indices_0based: Iterable[int],
    start_s: float,
    duration_s: float,
) -> tuple[np.ndarray, float]:
    if start_s < 0 or duration_s <= 0:
        raise ValueError("start_s must be >= 0 and duration_s must be > 0")

    path = _edf_path(patient, file_name)
    reader = pyedflib.EdfReader(str(path))
    try:
        indices = list(channel_indices_0based)
        if not indices:
            raise ValueError("No channels requested")
        sample_rates = [float(reader.getSampleFrequency(i)) for i in indices]
        if len(set(sample_rates)) != 1:
            raise ValueError(f"Selected channels have mixed sample rates: {sample_rates}")
        fs = sample_rates[0]
        start_sample = int(round(start_s * fs))
        n_samples = int(round(duration_s * fs))
        signals = []
        for i in indices:
            total = int(reader.getNSamples()[i])
            if start_sample >= total:
                raise ValueError(f"start_s exceeds channel duration for index {i}")
            n = min(n_samples, total - start_sample)
            signals.append(reader.readSignal(i, start=start_sample, n=n))
        min_len = min(len(x) for x in signals)
        data = np.vstack([x[:min_len] for x in signals])
        return data, fs
    finally:
        reader.close()
