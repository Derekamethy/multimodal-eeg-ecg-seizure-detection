from pathlib import Path
import argparse
import pandas as pd

from src.multimodal_seizure.features import extract_feature_row
from src.multimodal_seizure.provenance import artifact_spec, require_current, protect_existing, stamp_artifact, sidecar

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("patient")
args = parser.parse_args()
patient = args.patient

window_path = ROOT / "metadata/primary_window_index.csv"
require_current(window_path)
windows = pd.read_csv(window_path)
target = windows[(windows.patient == patient) & (windows.label >= 0)].copy()
target = target.sort_values(["file_name", "anchor_s"]).reset_index(drop=True)

out_dir = ROOT / "results" / "features"
out_dir.mkdir(parents=True, exist_ok=True)
partial = out_dir / f"{patient}_features.canonical.partial.csv"
final = out_dir / f"{patient}_features.canonical.csv"
spec = artifact_spec("features", [Path(__file__).resolve(), window_path] +
    [ROOT / f"src/multimodal_seizure/{name}.py" for name in ("features", "data", "eeg", "ecg")],
    {"patient": patient, "eeg_context_s": 10, "ecg_context_s": 60, "usable_only": True})
protect_existing(final)
if partial.exists():
    require_current(partial, expected_identity=spec["cache_identity"])

existing = pd.read_csv(partial) if partial.exists() else pd.DataFrame()
done = set()
if not existing.empty:
    done = set(zip(existing.file_name.astype(str), existing.anchor_s.astype(float)))
rows = existing.to_dict("records")
print(f"PATIENT={patient} TARGET={len(target)} RESUME={len(rows)}", flush=True)
for i, row in target.iterrows():
    key = (str(row.file_name), float(row.anchor_s))
    if key in done:
        continue
    feat = extract_feature_row(row.patient, row.file_name, float(row.anchor_s))
    feat.update({
        "patient": row.patient,
        "file_name": row.file_name,
        "label": int(row.label),
        "status": row.status,
        "seizure_numbers": row.seizure_numbers,
    })
    rows.append(feat)
    done.add(key)

    if len(rows) % 100 == 0:
        pd.DataFrame(rows).to_csv(partial, index=False)
        stamp_artifact(partial, spec)
        print(f"FEATURES {len(rows)}/{len(target)}", flush=True)

df = pd.DataFrame(rows)
df = df.sort_values(["file_name", "anchor_s"]).reset_index(drop=True)
if len(df) != len(target):
    raise RuntimeError(f"Feature count mismatch: {len(df)} != {len(target)}")
df.to_csv(final, index=False)
stamp_artifact(final, spec)
partial.unlink(missing_ok=True)
sidecar(partial).unlink(missing_ok=True)
print(f"COMPLETE {patient} ROWS={len(df)} OUTPUT={final}", flush=True)
