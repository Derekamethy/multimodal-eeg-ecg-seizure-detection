# Siena Pilot Data Audit

> Historical pre-canonical audit, retained for traceability. Its proposed annotation corrections and inventories do not supersede the frozen canonical metadata. Current eligibility and results are documented in [FINAL_PROJECT_SUMMARY.md](FINAL_PROJECT_SUMMARY.md).

## Scope

Pilot cohort: PN00, PN06, PN10, PN12, PN14.

Downloaded raw files were verified against the PhysioNet `SHA256SUMS.txt` manifest before any signal analysis. The verified subset contains 23 EDF recordings and 28 annotated seizures.

## Integrity

- SHA-256 verification: 31/31 targeted files passed.
- All 23 EDF files report 512 Hz for every signal in the EDF header.
- Raw EDF files are excluded from Git by `.gitignore`.
- Audit outputs are stored in `metadata/edf_audit.csv`, `metadata/seizure_event_audit.csv`, and `metadata/patient_summary.csv`.

## ECG channel reconciliation

The seizure-list metadata, not the literal EDF label string, is authoritative for the two intended EKG channels.

| Patient | Official EKG 1 | Official EKG 2 | EDF header labels |
| --- | ---: | ---: | --- |
| PN00 | channel 33 | channel 34 | `1`, `2` |
| PN06 | channel 32 | channel 33 | `1`, `2` |
| PN10 | channel 32 | channel 33 | `1`, `2` |
| PN12 | channel 32 | channel 33 | `1`, `2` |
| PN14 | channel 29 | channel 30 | `1`, `2` |

PN00, PN06, PN10 and PN12 also contain an additional EDF signal literally labelled `EKG EKG`. Its physiological meaning is not sufficiently documented, so it is excluded from the primary ECG feature path and retained only for secondary diagnostics.
## Annotation anomalies

### PN00 seizure 3

The PhysioNet text file gives:

- EDF: `PN00-3.edf`
- recording start: 18:15:44
- seizure start: 18:28:29
- seizure end: 19:29:29

The EDF itself lasts only 2509 s, so the listed end time is outside the recording. An independent reproduction reports 18:29:29, while a later Meta-EEG derivative excluded PN00-3 because of incomplete annotations.

Primary decision: exclude `PN00-3.edf` entirely from primary model development and evaluation. Do not use its remaining samples as background.

Sensitivity-only decision: a separate explicitly labelled analysis may use 18:29:29 as a corrected end time. This result must never be merged silently into the primary estimate.

### PN14 seizure 3

The seizure-list gives registration start 16:17:45, but the EDF header gives 19:17:45. The EDF duration is 41995 s and its header start is consistent with the listed recording end time. Therefore the EDF header start time is used for time alignment. The seizure remains eligible.

### PN06 filenames

The seizure-list uses `PNO6-1.edf`, `PNO6-2.edf`, and `PNO6-4.edf` with letter O. These are normalized to the verified files `PN06-*.edf`. This is a filename typo only; seizure times remain unchanged.

## Primary cohort after quarantine

- Patients: 5
- EDF recordings: 22
- Seizures: 27
- Sampling rate: 512 Hz
- Official ECG leads: 2 per patient

The original 23 EDF / 28 seizure subset is retained on disk; only PN00-3 is quarantined from the primary analysis.
