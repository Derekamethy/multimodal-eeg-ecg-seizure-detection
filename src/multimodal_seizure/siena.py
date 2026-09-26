from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .edf import EDFHeader


COMMON_EEG_19 = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
)
KNOWN_EEG = set(COMMON_EEG_19) | {"FC1", "FC5", "CP1", "CP5", "F9", "FC2", "FC6", "CP2", "CP6", "F10"}


def normalize_eeg_label(label: str) -> str:
    x = label.strip().upper()
    if x.startswith("EEG "):
        x = x[4:]
    return re.sub(r"\s+", "", x)


@dataclass(frozen=True)
class SienaChannelMap:
    eeg_indices: tuple[int, ...]
    eeg_names: tuple[str, ...]
    ecg_indices: tuple[int, int]
    ecg_names: tuple[str, str]
    extra_named_ecg_indices: tuple[int, ...]


def parse_declared_channels(seizure_list: str | Path) -> dict[int, str]:
    text = Path(seizure_list).read_text(encoding="utf-8", errors="replace")
    out: dict[int, str] = {}
    for m in re.finditer(r"Channel\s+(\d+)\s*:\s*([^\r\n]+)", text, re.I):
        number = int(m.group(1))
        if number in out:
            raise ValueError(f"Duplicate official channel number: {number}")
        out[number] = m.group(2).strip()
    if len(re.findall(r"^\s*Channel\s+\d", text, re.M | re.I)) != len(out):
        raise ValueError("Malformed or truncated channel declaration")
    if not out:
        raise ValueError("Missing official channel declarations")
    return out


def channel_rows(header: EDFHeader, seizure_list: str | Path) -> list[dict]:
    """The only modality/index reconciliation path; unlisted signals stay excluded."""
    rows = []
    identities = set()
    for number, name in sorted(parse_declared_channels(seizure_list).items()):
        index = number - 1
        if not 0 <= index < header.n_signals:
            raise ValueError(f"Official channel {number} outside EDF header range")
        raw = header.signals[index].label
        alias = ""
        if name.upper() in {"EKG 1", "EKG 2"}:
            modality, canonical = "ECG", name.upper().replace("EKG", "ECG")
            if raw != name[-1]:
                raise ValueError(f"ECG label mismatch at channel {number}: {raw!r}")
        else:
            modality, canonical = "EEG", normalize_eeg_label(name)
            if number == 5 and name == "1":
                canonical = "O1"
                alias = "official channel 5 '1' reconciled to EDF 'EEG O1'"
            if canonical not in KNOWN_EEG:
                raise ValueError(f"Unknown official channel identity: {name!r}")
            if normalize_eeg_label(raw) != canonical:
                raise ValueError(f"EEG label mismatch at channel {number}: {name!r} / {raw!r}")
        key = (modality, canonical)
        if key in identities:
            raise ValueError(f"Duplicate canonical channel: {key}")
        identities.add(key)
        rows.append(dict(channel_number_1based=number, edf_index_0based=index,
                         official_name=name, edf_raw_label=raw, modality=modality,
                         canonical_name=canonical, alias_note=alias))
    if {name for mod, name in identities if mod == "ECG"} != {"ECG 1", "ECG 2"}:
        raise ValueError("Missing official ECG lead")
    expected = KNOWN_EEG if header.path.parent.name in {"PN00", "PN06", "PN12", "PN14"} else set(COMMON_EEG_19)
    missing = expected - {name for mod, name in identities if mod == "EEG"}
    if missing:
        raise ValueError(f"Missing common EEG declarations: {sorted(missing)}")
    return rows


def build_channel_map(header: EDFHeader, seizure_list: str | Path) -> SienaChannelMap:
    rows = channel_rows(header, seizure_list)
    eeg_indices = tuple(r["edf_index_0based"] for r in rows if r["modality"] == "EEG")
    eeg_names = tuple(r["canonical_name"] for r in rows if r["modality"] == "EEG")
    ecg_indices = tuple(r["edf_index_0based"] for r in rows if r["modality"] == "ECG")

    extra_named = tuple(
        i for i, signal in enumerate(header.signals)
        if ("ECG" in signal.label.upper() or "EKG" in signal.label.upper())
        and i not in ecg_indices
    )
    ecg_names = tuple(header.signals[i].label for i in ecg_indices)

    return SienaChannelMap(
        eeg_indices=eeg_indices,
        eeg_names=eeg_names,
        ecg_indices=(ecg_indices[0], ecg_indices[1]),
        ecg_names=(ecg_names[0], ecg_names[1]),
        extra_named_ecg_indices=extra_named,
    )
