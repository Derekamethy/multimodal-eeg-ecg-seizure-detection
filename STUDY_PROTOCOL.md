# Siena EEG–ECG Multimodal Seizure Detection Study Protocol

## Status
**Final closeout: Phase 1, Phase 2, Phase 3 and Phase 4 are complete and frozen. No Phase 5 is planned in this pilot.** The definitive implemented protocol and findings are in [docs/FINAL_PROJECT_SUMMARY.md](docs/FINAL_PROJECT_SUMMARY.md); immutable identities are in [PROJECT_STATUS.md](PROJECT_STATUS.md).

The sections below preserve the Phase 1 protocol and its original prospective planning language. Proposed feature families, causal replay and expansion work were not all implemented. They are historical context, not commitments or current capabilities. The final implementation addendum at the end records what was actually frozen. Pre-canonical pilot outputs remain historical/stale.

## Primary research question
When does ECG add useful incremental information to patient-specific EEG-based seizure detection?

## Pilot cohort
Initial cohort: PN00, PN06, PN10, PN12, PN14.
Official seizure counts: 5 + 5 + 10 + 4 + 4 = 28 seizures.
All downloaded source files must pass the PhysioNet SHA-256 checks before use.

## Source integrity and provenance
- Raw data: Siena Scalp EEG Database v1.0.0, PhysioNet.
- Sampling rate: 512 Hz for all pilot EDF files.
- Raw EDF files are immutable and excluded from Git.
- Channel identity is defined by the official per-patient Seizures-list channel number, then mapped to the corresponding EDF signal index.
- Raw EDF labels alone must not be used to infer modality.
- Example: the official EKG channels are stored under raw labels "1" and "2" in the pilot cohort.
- `metadata/canonical_channels.csv` is the canonical per-EDF channel mapping table, validated on every selected header by the single `siena.channel_rows` implementation. Official numbers, zero-based indices, raw labels, explicit modality and aliases are preserved. Both extraction/read paths use this implementation; the official 29-channel EEG set (19 for PN10) is retained.
- `metadata/canonical_events.csv` is the authoritative derived event table; `metadata/canonical_recordings.csv` governs parent eligibility. `build_seizure_event_table.py` builds all three together; `build_official_channel_map.py` delegates to the same builder.
- `metadata/canonical_manifest.json` records source annotation/header hashes, implementation and table hashes, inventory, and stale historical artifacts. Header hashes and file size do not replace full raw-file SHA-256 verification.
- The old root event/channel/window tables, metadata audits and results are preserved as historical evidence, not canonical inputs. New window/feature/result paths use `primary_` or `.canonical`/`_canonical_` names. Provenance sidecars reject changed annotations, mapping, producer code, configuration or parent artifacts; stale files must be preserved rather than overwritten.
## Annotation QA
PN00 seizure 3 is a documented annotation discrepancy:
- PhysioNet text currently lists seizure end as 19.29.29, outside PN00-3.edf.
- The claimed screenshot provenance is unverified. The proposal 18.29.29 / 825 s is stored separately and is not canonical ground truth.
- The raw value is preserved; canonical offset is unset. The entire PN00-3 parent is quarantined, including all possible negative background. Primary inventory: 22 EDFs / 27 events; all 23 EDFs / 28 source events remain stored.
- PN14-3 uses EDF-header alignment: [6740, 6781) s. Both registration clocks and their 10800-s discrepancy remain recorded. Header start plus sample duration agrees with the source next-day end.
- PN10-2 retains both offset candidates (7828, 7849 s); primary offset is 7828 s and [7828, 7849) is annotation-uncertain.
- PN10-3 retains clinical onset 7835 s and electrographic onset 7841 s. The documented clinical reference is primary, with [7835, 7841) explicitly uncertain for window labels. PN10-6 is clinical-only; no electrographic onset is invented.
- Canonical intervals are half-open [onset, offset). Annotation-uncertain intervals override positive/negative window labels and must not be treated as ordinary negative exposure. Event/FAR evaluation of these intervals remains Phase 2 work.
- Source registration end times are 20 s shorter than sample-derived EDF duration in ten recordings; both values and the discrepancy are retained. EDF sample duration governs bounds. No seizure eligibility changes result from these end-clock discrepancies.
- The explicit PN10-7 registration typo `1 6.49.25` is normalized to `16.49.25` with original text and decision retained. Other malformed clocks fail validation. Missing PN12-2 registration clocks are inherited only from the same parent EDF and marked as inherited.
No other silent timestamp corrections are allowed.

## Experimental units and leakage control
The evaluation unit is a complete EDF recording / seizure-bearing recording, never a randomly sampled window.
Adjacent windows from the same parent EDF must never be split across train and test.
All fitting operations are restricted to the training side of each split, including scaling, feature selection, threshold selection, fusion weights and quality thresholds.

## Patient-specific validation
Use leave-one-recording-out evaluation where the patient has multiple seizure-bearing recordings.
For EDFs containing multiple seizures, for example PN10-4.5.6 and PN10-7.8.9, the entire EDF is one indivisible parent unit.
A held-out EDF must remain untouched until all training-side choices for that fold are frozen.

## Primary model comparisons
1. EEG-only baseline.
2. ECG-only baseline.
3. EEG+ECG feature-level early fusion.
4. EEG+ECG late fusion.
5. Patient-adaptive / quality-aware late fusion.

Original planned late-fusion form (the `p` symbols denoted model scores, not calibrated probabilities; superseded by the final binary-gate formula below):
p_fused = (1 - alpha) * p_EEG + alpha * p_ECG,
where alpha is selected using training-side data only and may equal zero.
## Modality processing
EEG and ECG must be preprocessed independently before temporal alignment.
Initial EEG features should remain compact: band-power / spectral-shape, temporal statistics and selected synchrony features.
Initial ECG features should include RR/HR-derived features such as mean HR, HR change, mean RR and RR variability; SDNN/RMSSD/pNN50 may be added when the analysis window is physiologically appropriate.
LF/HF must not be used in short windows without an explicit validity justification.

## Channel policy
Patient-specific experiments may use each patient's official EEG channel set.
Cross-patient or pooled secondary experiments must use a predefined common channel policy rather than silently mixing montages.
The current official pilot lists contain fewer EEG channels for PN10 than for the other four patients; this must be handled explicitly.
Two official EKG channels are available for every pilot patient.

## Offline and causal tracks
Offline retrospective track may use zero-phase filtering, but must be labeled offline.
Deployment-oriented track must use causal filtering, causal/online QRS detection, trailing smoothing and confirmation-time alarms.
Offline and causal metrics must never be mixed in the same headline result.

## Thresholds, smoothing and alarms
Detection thresholds, smoothing widths, refractory periods and alarm-merging rules are hyperparameters.
They must be selected exclusively on training/validation data inside each fold.
Alarm latency is measured at the time at which a causal system can actually confirm the alarm, not at the center of a retrospective window.
## Primary reporting
Report event sensitivity, false alarms per hour, detection latency and an explicitly defined window-level metric.
Report per-patient results before any pooled average.
For fusion, report the patient/fold-specific alpha values and whether fusion improves, matches or degrades EEG-only.
Do not define success as "fusion must beat EEG for every patient."

## Secondary analyses
- ECG signal-quality stratification.
- Ictal HR / RR response versus fusion benefit.
- Early fusion versus fixed late fusion versus adaptive late fusion.
- Ablation of ECG feature groups.
- Failure analysis for false alarms and missed seizures.

## Freeze rule
Once a fold's held-out EDF has been evaluated, no model, threshold, feature, preprocessing or fusion change may be selected because it improves that held-out EDF.
Any post-hoc exploration must be labeled exploratory and evaluated in a new valid split or a future dataset.

## Historical planned implementation sequence
1. Complete source integrity audit.
2. Freeze channel map and event table.
3. Build modality-specific extraction tests.
4. Implement EEG-only baseline.
5. Implement ECG-only baseline.
6. Implement early fusion.
7. Implement late/adaptive fusion.
8. Add causal replay.
9. Run independent audit before public claims.

## Final implementation addendum

- Phase 1 froze the canonical annotation, eligibility and official channel mappings. PN00-3 remains quarantined; PN10 uncertainty and PN14 alignment policies above remain in force.
- Phase 2 completed continuous 10-second EEG windows on a 5-second decision grid, 46 engineered features, fixed LR/RF baselines and nested whole-parent thresholds. The event evaluator opens on the first high decision, bridges one low, closes on two consecutive lows and uses no refractory period. Strict new declarations must lie in `[onset, offset)`; duplicates are unmatched for FAR. Uncertainty/gaps interrupt state.
- Phase 3 completed decision-causal trailing 60-second ECG processing, deterministic dual-lead quality selection, 12 rhythm features and past-only references in `[t-360,t-60)`. This is not validated streaming QRS or clinical beat accuracy. No LF/HF, morphology bank or learned SQI was added.
- Phase 4 primary F2 used `sF=(1-alpha*q)*sE+alpha*q*sC`, frozen binary ECG availability and alpha in `{0,.1,.25,.5}`. The accepted pre-evaluation clarification also made the threshold fall back: `q=0` uses the inner EEG threshold; `q=1` uses the fusion threshold; alpha=0 always uses EEG. Gate changes retain alarm state. Consequently a fusion threshold of 1.01 disables alarms only on the fusion-active branch, not the EEG fallback branch.
- Nonzero alpha required training-side non-decreasing sensitivity, non-increasing FAR, at least one strict improvement, no more than 5 seconds median paired delay increase on jointly detected events, and preserved coverage. Otherwise alpha=0 retained EEG. With no jointly detected inner events, the delay comparison is undefined and imposes no additional constraint.
- F1 fixed .5 and F3 fixed-capacity early RF were secondary comparators. The threshold grid remained `.05,.10,...,.95,1.01`. No outer-driven model or rule redesign followed evaluation.
- The final study found no reliable incremental ECG value. EEG and F2 both detected 13/27, but F2 lost one EEG detection, rescued one, and changed false declarations only from 69 to 68. Proposed later causal, expansion and architecture work is not part of this release.
