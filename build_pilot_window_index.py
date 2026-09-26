from pathlib import Path
import pandas as pd

from src.multimodal_seizure.data import PILOT_PATIENTS, read_event_table
from src.multimodal_seizure.windows import build_window_index
from src.multimodal_seizure.provenance import artifact_spec, protect_existing, stamp_artifact

ROOT = Path(__file__).resolve().parent
output = ROOT / "metadata/primary_window_index.csv"
summary_output = ROOT / "metadata/primary_window_summary.csv"
for path in (output, summary_output):
    protect_existing(path)
spec = artifact_spec("windows", [Path(__file__).resolve(), ROOT / "src/multimodal_seizure/windows.py",
                                ROOT / "src/multimodal_seizure/data.py"],
                     {"stride_s": 5, "eeg_context_s": 10, "ecg_context_s": 60, "exclusion_s": 300})
events = read_event_table()
frames = []

for patient in PILOT_PATIENTS:
    files = sorted(events.loc[events.patient == patient, "canonical_file_name"].unique())
    for file_name in files:
        frames.append(build_window_index(patient, file_name))

index = pd.concat(frames, ignore_index=True)
index.to_csv(output, index=False)
stamp_artifact(output, spec)

summary = (
    index.groupby(["patient", "status"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
summary["usable"] = summary.get("positive", 0) + summary.get("negative", 0)
summary.to_csv(summary_output, index=False)
stamp_artifact(summary_output, spec)

seen = set()
for _, row in index.loc[index.label == 1, ["patient", "seizure_numbers"]].iterrows():
    for token in str(row.seizure_numbers).split("|"):
        if token and token != "nan":
            seen.add((row.patient, int(token)))

expected = set((r.patient, int(r.seizure_number)) for _, r in events.iterrows())
missing = sorted(expected - seen)

print(summary.to_string(index=False))
print("TOTAL_WINDOWS", len(index))
print("USABLE_WINDOWS", int((index.label >= 0).sum()))
print("POSITIVE_WINDOWS", int((index.label == 1).sum()))
print("NEGATIVE_WINDOWS", int((index.label == 0).sum()))
print("IGNORE_WINDOWS", int((index.label == -1).sum()))
print("MISSING_SEIZURES", missing)
