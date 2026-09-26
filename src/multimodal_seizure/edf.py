from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()


@dataclass(frozen=True)
class SignalHeader:
    label: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    samples_per_record: int

    def to_physical(self, values: np.ndarray) -> np.ndarray:
        span_d = self.digital_max - self.digital_min
        if span_d == 0:
            raise ValueError(f"Zero digital range for channel {self.label!r}")
        scale = (self.physical_max - self.physical_min) / span_d
        return (values.astype(np.float64) - self.digital_min) * scale + self.physical_min
@dataclass(frozen=True)
class EDFHeader:
    path: Path
    start_date: str
    start_time: str
    header_bytes: int
    n_records: int
    record_duration_s: float
    signals: tuple[SignalHeader, ...]

    @property
    def duration_s(self) -> float:
        return self.n_records * self.record_duration_s

    @property
    def n_signals(self) -> int:
        return len(self.signals)

    @property
    def record_samples(self) -> int:
        return sum(s.samples_per_record for s in self.signals)

    @property
    def record_bytes(self) -> int:
        return 2 * self.record_samples

    def sampling_rate(self, index: int) -> float:
        return self.signals[index].samples_per_record / self.record_duration_s

    def signal_offset_samples(self, index: int) -> int:
        return sum(s.samples_per_record for s in self.signals[:index])


def read_header(path: str | Path) -> EDFHeader:
    path = Path(path)
    with path.open("rb") as f:
        fixed = f.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Short EDF fixed header: {path}")
        start_date = _text(fixed[168:176])
        start_time = _text(fixed[176:184])
        header_bytes = int(_text(fixed[184:192]))
        n_records = int(_text(fixed[236:244]))
        record_duration_s = float(_text(fixed[244:252]))
        n_signals = int(_text(fixed[252:256]))
        if n_signals <= 0 or header_bytes != 256 * (n_signals + 1):
            raise ValueError(f"Invalid or truncated EDF header layout: {path}")
        if n_records <= 0 or not np.isfinite(record_duration_s) or record_duration_s <= 0:
            raise ValueError(f"Invalid EDF duration: {path}")
        signal_raw = f.read(header_bytes - 256)
    if len(signal_raw) != header_bytes - 256:
        raise ValueError(f"Short EDF signal header: {path}")

    pos = 0

    def column(width: int) -> list[str]:
        nonlocal pos
        out = [
            _text(signal_raw[pos + i * width : pos + (i + 1) * width])
            for i in range(n_signals)
        ]
        pos += n_signals * width
        return out

    labels = column(16)
    column(80)  # transducer
    column(8)   # physical dimension
    physical_min = [float(x) for x in column(8)]
    physical_max = [float(x) for x in column(8)]
    digital_min = [int(x) for x in column(8)]
    digital_max = [int(x) for x in column(8)]
    column(80)  # prefiltering
    samples_per_record = [int(x) for x in column(8)]
    if any(n <= 0 for n in samples_per_record):
        raise ValueError(f"Invalid EDF signal sample count: {path}")
    if path.stat().st_size < header_bytes + 2 * n_records * sum(samples_per_record):
        raise ValueError(f"Truncated EDF recording: {path}")
    column(32)  # reserved

    signals = tuple(
        SignalHeader(
            label=labels[i],
            physical_min=physical_min[i],
            physical_max=physical_max[i],
            digital_min=digital_min[i],
            digital_max=digital_max[i],
            samples_per_record=samples_per_record[i],
        )
        for i in range(n_signals)
    )
    return EDFHeader(
        path=path,
        start_date=start_date,
        start_time=start_time,
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration_s=record_duration_s,
        signals=signals,
    )


def read_channel_segment(
    path: str | Path,
    channel_indices: Iterable[int],
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> tuple[EDFHeader, dict[int, np.ndarray]]:
    header = read_header(path)
    indices = tuple(dict.fromkeys(int(i) for i in channel_indices))
    if not indices:
        return header, {}
    if any(i < 0 or i >= header.n_signals for i in indices):
        raise IndexError("Channel index outside EDF signal range.")
    if start_s < 0 or start_s >= header.duration_s:
        raise ValueError("start_s must lie inside the EDF recording.")

    end_s = header.duration_s if duration_s is None else min(
        header.duration_s, start_s + max(0.0, duration_s)
    )
    start_record = int(start_s // header.record_duration_s)
    end_record = int(np.ceil(end_s / header.record_duration_s))
    end_record = min(end_record, header.n_records)

    chunks: dict[int, list[np.ndarray]] = {i: [] for i in indices}
    offsets = {i: header.signal_offset_samples(i) for i in indices}
    with header.path.open("rb") as f:
        for record in range(start_record, end_record):
            record_base = header.header_bytes + record * header.record_bytes
            for i in indices:
                signal = header.signals[i]
                byte_offset = record_base + 2 * offsets[i]
                f.seek(byte_offset)
                raw = f.read(2 * signal.samples_per_record)
                if len(raw) != 2 * signal.samples_per_record:
                    raise ValueError(
                        f"Unexpected EOF in {header.path.name}, record {record}, channel {i}"
                    )
                chunks[i].append(np.frombuffer(raw, dtype="<i2").copy())

    result: dict[int, np.ndarray] = {}
    record_start_s = start_record * header.record_duration_s
    for i in indices:
        signal = header.signals[i]
        fs = header.sampling_rate(i)
        digital = np.concatenate(chunks[i]) if chunks[i] else np.empty(0, dtype=np.int16)
        first = int(round((start_s - record_start_s) * fs))
        count = int(round((end_s - start_s) * fs))
        digital = digital[first : first + count]
        result[i] = signal.to_physical(digital)

    return header, result
