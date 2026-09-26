from pathlib import Path
import numpy as np
import pandas as pd

from src.multimodal_seizure.features import extract_feature_row
from src.multimodal_seizure.provenance import artifact_spec, require_current, protect_existing, stamp_artifact

ROOT = Path(__file__).resolve().parent
window_path = ROOT / "metadata/primary_window_index.csv"
require_current(window_path)
dest = ROOT / "pn00_smoke_features.canonical.csv"
protect_existing(dest)
spec = artifact_spec("smoke_features", [Path(__file__).resolve(), window_path] +
    [ROOT / f"src/multimodal_seizure/{name}.py" for name in ("features", "data", "eeg", "ecg")],
    {"seed": 20260921, "negative_ratio": 3})
windows = pd.read_csv(window_path)
pn00 = windows[(windows.patient == "PN00") & (windows.label >= 0)].copy()

selected = []
rng = np.random.default_rng(20260921)
for file_name, group in pn00.groupby("file_name"):
    pos = group[group.label == 1]
    neg = group[group.label == 0]
    n_neg = min(len(neg), max(1, 3 * len(pos)))
    neg_idx = rng.choice(neg.index.to_numpy(), size=n_neg, replace=False)
    selected.append(pos)
    selected.append(neg.loc[neg_idx])

sample = pd.concat(selected).sort_values(["file_name", "anchor_s"]).reset_index(drop=True)
print("SMOKE_ROWS", len(sample), "POS", int((sample.label == 1).sum()), "NEG", int((sample.label == 0).sum()), flush=True)

rows = []
for i, row in sample.iterrows():
    feat = extract_feature_row(row.patient, row.file_name, float(row.anchor_s))
    feat.update({
        "patient": row.patient,
        "file_name": row.file_name,
        "label": int(row.label),
        "status": row.status,
        "seizure_numbers": row.seizure_numbers,
    })
    rows.append(feat)
    if (i + 1) % 25 == 0 or i + 1 == len(sample):
        print(f"FEATURES {i+1}/{len(sample)}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(dest, index=False)
stamp_artifact(dest, spec)
print("OUTPUT", dest, flush=True)
