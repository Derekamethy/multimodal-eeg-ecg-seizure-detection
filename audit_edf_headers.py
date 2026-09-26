from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent / 'data/raw/siena-scalp-eeg-1.0.0'
PATIENTS = ["PN00", "PN06", "PN10", "PN12", "PN14"]

def field(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip()

def parse_edf(path: Path) -> dict:
    with path.open("rb") as f:
        fixed = f.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Short EDF header: {path}")
        header_bytes = int(field(fixed[184:192]))
        n_records = int(field(fixed[236:244]))
        record_duration = float(field(fixed[244:252]))
        n_signals = int(field(fixed[252:256]))
        signal_header = f.read(header_bytes - 256)
    if len(signal_header) != n_signals * 256:
        raise ValueError(f"Unexpected signal header size: {path}")
    pos = 0
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    blocks = []
    for w in widths:
        blocks.append(signal_header[pos:pos + w*n_signals])
        pos += w*n_signals
    labels = [field(blocks[0][i*16:(i+1)*16]) for i in range(n_signals)]
    samples = [int(field(blocks[8][i*8:(i+1)*8])) for i in range(n_signals)]
    fs = [s / record_duration for s in samples]
    duration_s = n_records * record_duration if n_records >= 0 else None
    return {"file": str(path.relative_to(ROOT)).replace("\\","/"), "n_records": n_records,
            "record_duration_s": record_duration, "duration_s": duration_s,
            "n_signals": n_signals, "labels": labels, "samples_per_record": samples, "fs": fs}
rows = []
for patient in PATIENTS:
    for path in sorted((ROOT / patient).glob("*.edf")):
        info = parse_edf(path)
        eeg = [lab for lab in info["labels"] if "EEG" in lab.upper()]
        ekg = [lab for lab in info["labels"] if any(k in lab.upper() for k in ("ECG","EKG"))]
        unique_fs = sorted(set(info["fs"]))
        rows.append({
            "file": info["file"],
            "n_signals": info["n_signals"],
            "duration_s": info["duration_s"],
            "duration_h": None if info["duration_s"] is None else info["duration_s"]/3600,
            "unique_fs_hz": ",".join(f"{x:g}" for x in unique_fs),
            "eeg_channels": "|".join(eeg),
            "ekg_channels": "|".join(ekg),
            "all_labels": "|".join(info["labels"]),
        })
        print(info["file"])
        print(f"  signals={info['n_signals']} duration_h={rows[-1]['duration_h']:.3f} fs={unique_fs}")
        print(f"  EKG={ekg}")
        print(f"  labels={info['labels']}")

out = Path(__file__).resolve().parent / 'edf_header_audit.csv'
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"AUDIT_ROWS={len(rows)}")
print(f"OUTPUT={out}")
