from __future__ import annotations
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'data/raw/siena-scalp-eeg-1.0.0'
OUT = Path(__file__).resolve().parent / 'metadata'
PATIENTS = ["PN00", "PN06", "PN10", "PN12", "PN14"]

def field(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()

def parse_clock(text: str):
    if not text:
        return None
    clean = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    m = re.search(r"(\d{1,2})\s*[.:]\s*(\d{1,2})\s*[.:]\s*(\d{1,2})", clean)
    if not m:
        return None
    h, mnt, s = map(int, m.groups())
    if h > 23 or mnt > 59 or s > 59:
        return None
    return h * 3600 + mnt * 60 + s

def rel_time(clock: int | None, start: int | None):
    if clock is None or start is None:
        return None
    x = clock - start
    if x < -12 * 3600:
        x += 24 * 3600
    return x

def parse_edf_header(path: Path) -> dict:
    with path.open("rb") as f:
        fixed = f.read(256)
        start_date = field(fixed[168:176])
        start_time = field(fixed[176:184])
        header_bytes = int(field(fixed[184:192]))
        n_records = int(field(fixed[236:244]))
        record_duration = float(field(fixed[244:252]))
        n_signals = int(field(fixed[252:256]))
        sig = f.read(header_bytes - 256)
    pos = 0
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    cols = []
    for w in widths:
        vals = [field(sig[pos+i*w:pos+(i+1)*w]) for i in range(n_signals)]
        cols.append(vals)
        pos += n_signals * w
    labels = cols[0]
    samples = [int(x) for x in cols[8]]
    fs = [s / record_duration for s in samples]
    return {
        "file": path.name,
        "start_date": start_date,
        "start_time": start_time,
        "start_clock_s": parse_clock(start_time),
        "header_bytes": header_bytes,
        "n_records": n_records,
        "record_duration_s": record_duration,
        "duration_s": n_records * record_duration,
        "n_signals": n_signals,
        "labels": labels,
        "sampling_rates": fs,
    }

def parse_seizure_list(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    declared_channels = {}
    for m in re.finditer(r"Channel\s+(\d+)\s*:\s*([^\r\n]+)", text, re.I):
        declared_channels[int(m.group(1))] = m.group(2).strip()
    official_ecg = {
        idx: name for idx, name in declared_channels.items()
        if "ECG" in name.upper() or "EKG" in name.upper()
    }
    blocks = re.split(r"(?=Seizure\s+n\s*\d+)", text, flags=re.I)
    events = []
    for block in blocks:
        mnum = re.search(r"Seizure\s+n\s*(\d+)", block, re.I)
        if not mnum:
            continue
        def line(prefix):
            m = re.search(rf"{prefix}\s*:\s*([^\r\n]+)", block, re.I)
            return m.group(1).strip() if m else ""
        file_name = line("File name")
        file_name = re.sub(r"^PNO6", "PN06", file_name, flags=re.I)
        events.append({
            "seizure_number": int(mnum.group(1)),
            "file": file_name,
            "registration_start_raw": line("Registration start time"),
            "registration_end_raw": line("Registration end time"),
            "seizure_start_raw": line("Seizure start time"),
            "seizure_end_raw": line("Seizure end time"),
        })
    return text, declared_channels, official_ecg, events
edf_rows = []
edf_by_patient_file = {}
patient_meta = {}
for patient in PATIENTS:
    text, declared_channels, official_ecg, events = parse_seizure_list(ROOT / patient / f"Seizures-list-{patient}.txt")
    patient_meta[patient] = {
        "text": text,
        "declared_channels": declared_channels,
        "official_ecg": official_ecg,
        "events": events,
    }
    for path in sorted((ROOT / patient).glob("*.edf")):
        h = parse_edf_header(path)
        official_pairs = []
        for idx, name in sorted(official_ecg.items()):
            header_label = h["labels"][idx - 1] if 1 <= idx <= h["n_signals"] else "<out-of-range>"
            official_pairs.append(f"{idx}:{name}->{header_label}")
        extra_named = [
            f"{i+1}:{label}" for i, label in enumerate(h["labels"])
            if ("ECG" in label.upper() or "EKG" in label.upper()) and (i + 1) not in official_ecg
        ]
        row = {
            "patient": patient,
            "file": h["file"],
            "duration_s": h["duration_s"],
            "duration_h": round(h["duration_s"] / 3600, 4),
            "n_signals": h["n_signals"],
            "header_start_date": h["start_date"],
            "header_start_time": h["start_time"],
            "unique_sampling_rates_hz": " | ".join(f"{x:g}" for x in sorted(set(h["sampling_rates"]))),
            "official_ecg_channel_count": len(official_ecg),
            "official_ecg_mapping": " | ".join(official_pairs),
            "extra_ecg_named_channels": " | ".join(extra_named),
            "all_channels": " | ".join(f"{i+1}:{x}" for i, x in enumerate(h["labels"])),
        }
        edf_rows.append(row)
        edf_by_patient_file[(patient, h["file"])] = h

with (OUT / "edf_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(edf_rows[0]))
    w.writeheader()
    w.writerows(edf_rows)

event_rows = []
patient_summary = []
for patient in PATIENTS:
    meta = patient_meta[patient]
    events = meta["events"]
    file_counts = Counter(e["file"] for e in events)
    mfs = re.search(r"Data Sampling Rate:\s*([0-9.]+)\s*Hz", meta["text"], re.I)
    declared_fs = float(mfs.group(1)) if mfs else None
    for e in events:
        h = edf_by_patient_file.get((patient, e["file"]))
        flags = []
        if h is None:
            flags.append("file_not_found")
        canonical_start = h["start_clock_s"] if h else None
        list_reg_start = parse_clock(e["registration_start_raw"])
        if h and list_reg_start is not None:
            delta = abs(rel_time(list_reg_start, canonical_start) or 0)
            if delta > 5:
                flags.append("registration_start_text_disagrees_with_edf_header")
        sz_start_clock = parse_clock(e["seizure_start_raw"])
        sz_end_clock = parse_clock(e["seizure_end_raw"])
        sz_start = rel_time(sz_start_clock, canonical_start)
        sz_end = rel_time(sz_end_clock, canonical_start)
        if sz_start_clock is None:
            flags.append("seizure_start_unparsed")
        if sz_end_clock is None:
            flags.append("seizure_end_unparsed")
        if sz_start is not None and sz_start < 0:
            flags.append("negative_seizure_start_offset")
        if sz_start is not None and sz_end is not None and sz_end < sz_start:
            flags.append("seizure_end_before_start")
        if h is not None and sz_start is not None and sz_start > h["duration_s"] + 5:
            flags.append("seizure_start_after_edf_end")
        if h is not None and sz_end is not None and sz_end > h["duration_s"] + 5:
            flags.append("seizure_end_after_edf_end")
        event_rows.append({
            "patient": patient,
            "seizure_number": e["seizure_number"],
            "file": e["file"],
            "seizures_in_same_edf": file_counts[e["file"]],
            "edf_header_start_time": h["start_time"] if h else "",
            "registration_start_raw": e["registration_start_raw"],
            "edf_duration_s": h["duration_s"] if h else "",
            "seizure_start_raw": e["seizure_start_raw"],
            "seizure_end_raw": e["seizure_end_raw"],
            "seizure_start_offset_s": sz_start if sz_start is not None else "",
            "seizure_end_offset_s": sz_end if sz_end is not None else "",
            "seizure_duration_s": (sz_end - sz_start) if sz_start is not None and sz_end is not None else "",
            "flags": " | ".join(flags),
        })
    patient_edfs = [r for r in edf_rows if r["patient"] == patient]
    mappings = sorted(set(r["official_ecg_mapping"] for r in patient_edfs))
    extras = sorted(set(r["extra_ecg_named_channels"] for r in patient_edfs if r["extra_ecg_named_channels"]))
    patient_summary.append({
        "patient": patient,
        "edf_count": len(patient_edfs),
        "seizure_count": len(events),
        "declared_fs_hz": declared_fs,
        "official_ecg_count": len(meta["official_ecg"]),
        "official_ecg_mapping": " ; ".join(mappings),
        "extra_ecg_named_channels": " ; ".join(extras),
        "header_sampling_rates_hz": " ; ".join(sorted(set(r["unique_sampling_rates_hz"] for r in patient_edfs))),
        "total_duration_h": round(sum(r["duration_s"] for r in patient_edfs) / 3600, 3),
    })

with (OUT / "seizure_event_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(event_rows[0]))
    w.writeheader()
    w.writerows(event_rows)

with (OUT / "patient_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(patient_summary[0]))
    w.writeheader()
    w.writerows(patient_summary)

print(f"EDF_FILES={len(edf_rows)}")
print(f"SEIZURES={len(event_rows)}")
print(f"FLAGGED_EVENTS={sum(bool(x['flags']) for x in event_rows)}")
for r in patient_summary:
    print("PATIENT", r)
for e in event_rows:
    if e["flags"]:
        print("FLAG", e)
