# Independent EEG + ECG research audit and staged study plan

> Historical planning record, retained without rewriting its scientific chronology. Phase 1–4 are now complete and frozen; proposed later stages below are not release capabilities or active plans. The definitive account is [FINAL_PROJECT_SUMMARY.md](FINAL_PROJECT_SUMMARY.md). No Phase 5 is planned in this pilot.

**Review date: 22 September 2026. Status: research recommendation, not an implemented or frozen protocol.**

Scope: the current Siena project and the previous `eeg-seizure-detection` CHB-MIT/chb04 extension. This review inspected source code, raw annotation text, all pilot EDF headers, existing feature/prediction tables, tests, and the previous audit and its machine-readable evidence. It recomputed file hashes and saved-prediction metrics, but did not retrain models, regenerate datasets, modify raw files, replace results, or alter either existing protocol. Only this requested report was added.

Evidence labels used below: **verified locally**, **reported in literature**, **reviewer interpretation**, and **proposed protocol**. A proposal is not evidence that the corresponding mechanism works.

## 1. Executive summary

**Continue, but first repair the annotation and evaluation foundations. Do not run a fusion optimization campaign yet.** The defensible question is whether ECG adds information to a credible EEG detector on unseen recordings, and under which signal-quality and physiological conditions. A reproducible null result is a successful scientific outcome.

The recommended primary architecture is a compact EEG model plus a separate RR/HR model, combined by **training-only, quality-aware adaptive late fusion with an explicit EEG-only option**. Prefer relative HR response and a small number of short-window variability descriptors over a large HRV catalogue. A deterministic quality gate should disable ECG when it is unreliable. Learn at most one fusion weight per outer training set; do not learn a neural gate from this cohort. Treat model outputs as scores until calibration is independently demonstrated.

Use whole parent recordings for outer and inner splits. Evaluate continuous held-out recordings, including peri-ictal periods excluded from training. Freeze alarm semantics and report event sensitivity, false alarms per hour, latency, and recording coverage together. A separate chronological replay is required for claims about deployment to future recordings; causal feature extraction alone does not make leave-one-recording-out training prospective.

Three findings materially change the starting assumptions:

1. **PN14-3 is mislabeled in the event table used by the feature pipeline.** The root table uses onset/offset 17540/17581 s; the EDF-header-aligned audit uses 6740/6781 s. The difference is 10800 s. Both are inside this long EDF, so existing bounds tests pass.
2. **PN00-3 has conflicting eligibility policies.** `STUDY_PROTOCOL.md` includes a corrected end, while `docs/DATA_AUDIT.md` quarantines the entire EDF. The advertised correction lacks a verifiable citation in the derived row. Existing PN00 results include that EDF and cannot resolve this policy conflict.
3. **Siena EEG+ECG fusion is already published.** A 2024 detection paper and a 2023 prediction paper directly overlap the broad concept. Novelty must be accurate annotations, continuous event evaluation, training-only adaptation, honest quality fallback, and a reproducible causal/offline distinction—not first use of ECG on Siena. See R01–R03 below.

The raw-data foundation is sound: **31/31 selected files passed SHA-256; all 23 EDFs have 512 Hz signals; 18 existing tests passed.** Those checks establish integrity and limited implementation correctness, not validity of the scientific protocol.

## 2. Current repository audit

### 2.1 Verified data and engineering status

| Item | Independent observation | What it does and does not establish |
|---|---|---|
| File integrity | Ran `.venv/Scripts/python.exe verify_siena_subset.py`: `VERIFIED=31/31`, exit 0 | All 23 EDFs, five seizure lists, LICENSE, RECORDS and subject metadata match the local official manifest |
| Manifest provenance | Downloaded only the small official SHA256SUMS manifest into memory; byte-identical to local copy | Manifest SHA-256: `855450dd7c7b1ea40c042a195eff8385994e6718b1c4c17951498ada78d5e568` |
| Downloader status | Its `ALL_DOWNLOADS_VERIFIED` terminal condition is corroborated by the read-only checks above | Did not rerun the downloader: it can delete/redownload failed files. No assertion that an old console message was independently recovered |
| EDF headers | Read every pilot EDF header; 23 recordings, all signals 512 Hz | Header rate and mapped labels checked, not expert waveform adjudication of all hours |
| Raw recording time | 218187 s = **60.6075 h** | Use EDF sample-derived duration, not rounded/inconsistent subject metadata totals |
| Channel map | Every saved official index/raw-label pair matches every corresponding pilot EDF header | Builder itself checks only the first EDF per patient; future pipeline must enforce the all-record invariant |
| Tests | `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`: **18 passed in 10.27 s** | Existing tests miss annotation-source disagreement, RR-gap arithmetic, stream causality and event scoring |
| Git | `.git` absent in the new project directory at review time; `.gitignore` excludes `data/raw/`, `*.edf`, `.venv/` | Ignore rules are present; a versioned repository, tracked-file audit and reproducible commit are not yet established |
| Implemented | Readers, channel mapping, window indexing, engineered EEG/ECG features, PN00 feature cache, RF benchmarks | Nested adaptation, SQI fallback, continuous event evaluator, causal replay and immutable run manifests are absent |

The source of truth is [Siena v1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/), whose usage notes explicitly say to use the listed channels and ignore the others. Its database-wide counts are 14 patients and 47 seizures. This is a selected inpatient recording collection, not a representative free-living monitoring cohort.

| Patient | EDFs | Official events | Header hours | Official EEG count in active path | ECG channels, 1-based → 0-based |
|---|---:|---:|---:|---:|---|
| PN00 | 5 | 5 | 3.244722 | 29 | 33,34 → 32,33 |
| PN06 | 5 | 5 | 12.072778 | 29 | 32,33 → 31,32 |
| PN10 | 6 | 10 | 18.726944 | 19 | 32,33 → 31,32 |
| PN12 | 3 | 4 | 6.099167 | 29 | 32,33 → 31,32 |
| PN14 | 4 | 4 | 20.463889 | 29 | 29,30 → 28,29 |

### 2.2 Blocking annotation findings

**A. PN14-3: a three-hour alignment error in the active path.**

The raw seizure list says registration begins at 16:17:45. The EDF header says 19:17:45, with duration 41995 s; that header plus duration reaches the listed next-day end 06:57:40. The seizure is listed at 21:10:05–21:10:46. `build_seizure_event_table.py` subtracts the text registration start and never compares the header clock. `data.read_event_table()` and `windows.py` consume its root `seizure_events_audit.csv`, giving 17540–17581 s. In contrast, `audit_siena_data.py` and `metadata/seizure_event_audit.csv` use the EDF header and yield 6740–6781 s, with a discrepancy flag.

**Consequence:** current PN14 windows label an unrelated segment positive and can label the actual seizure negative. Representing all 28 event IDs is therefore not proof of representing all 28 physiological events. Correct the derived alignment with both raw clocks preserved; then invalidate downstream PN14 windows/features/results by provenance hash. Existing PN00 results are not affected by this particular error. The header-aligned choice is strongly supported by internal timing consistency, not a new clinical reannotation.

**B. PN00-3: preserve the error and quarantine primary analysis until adjudicated.**

Raw end 19:29:29 gives offset 4425 s in a 2509-s EDF. The root table retains this raw value and stores corrected 18:29:29 / 825 s with `CORRECTED_END_TIME`; that separation is good. However, the correction note has no URL, figure/page, retrieval date, source hash, or adjudicator. Targeted searches did not establish the claimed screenshot provenance. It must remain **unverified**, not promoted to ground truth because a plausible one-hour correction exists. The independent Meta-EEGs derivative explicitly excludes PN00-3 for incomplete annotations (R29).

Recommendation: follow the conservative existing `DATA_AUDIT.md` policy—quarantine the entire EDF in the primary analysis, leaving a provisional **22 EDF / 27-event pilot**. Do not reuse uncertain residual time as negative background. Keep a separately labeled corrected-end sensitivity analysis only after evidence is traceable. Do not select inclusion based on model performance. Resolve the conflicting documents before freezing anything; this report does not silently overwrite either.

**C. PN10 contains unmodeled annotation ambiguity.**

* Seizure 2 lists `11.41.04 opure 11.40.43`: two possible offsets, 7849 and 7828 s. The parser silently takes the first and emits no ambiguity flag.
* Seizure 3 lists clinical onset 15:43:53 and electric onset 15:43:59. The parser selects the first, offset 7835 s, while electrographic onset is 7841 s. These are different valid reference events, not interchangeable labels.
* Seizure 6 explicitly identifies a clinical onset; an electrographic onset is not separately supplied.
* Seizure 4 lasts only five seconds, 2309–2314 s. A two-consecutive-decision rule with 5-s stride can make it structurally undetectable within its annotated duration.

Recommendation: add `onset_kind`, separate clinical/electrographic onsets where available, candidate offsets, chosen policy, and uncertainty flags. Until expert resolution, use the shorter PN10-2 offset as a conservative primary boundary, label the intervening 21 s **annotation-uncertain**, and exclude that interval from training labels and strict specificity/FAR exposure while reporting its duration. Report both endpoints as a frozen sensitivity analysis. For the mixed-reference cohort, call latency **latency to the documented reference onset**, with a separately reported electrographic-onset subset. Do not call all latencies EEG-onset latencies. If uncertainty cannot be represented consistently, quarantine the affected parent EDF rather than treating ambiguous time as known negative; the final eligible count must then be recomputed.

### 2.3 Channel reconciliation and competing paths

The authoritative ECG sequence is correct: official channel number → EDF signal index (`number - 1`) → raw label (`1`, `2`) → canonical ECG identity. An additional signal literally named `EKG EKG` exists in PN00/06/10/12 but is excluded from the official ECG path. Do not automatically add it.

The active `data.py` path uses all non-EKG entries in `official_channel_map.csv`: 29 EEG channels except PN10, which has 19. It assumes every non-EKG declaration is EEG, so future metadata should classify modality explicitly. All pilot lists also call channel 5 `1`, whereas the header calls it `EEG O1`. Record this as a reconciled alias with evidence; neither source should be silently overwritten.

`siena.py` instead locates a hard-coded common 19-channel set by header label, while only ECG is mapped by official channel number. Its tests pass, but they are **not tests of the active benchmark's EEG selection path**. The active extractor already aggregates across channels, so its feature dimensionality is fixed at 46 EEG features even with 19 versus 29 inputs. The risk is changed spatial meaning/distribution, not currently a literal 19-versus-29 feature-matrix shape error. Use one versioned mapping implementation and test both identity and representation semantics.

### 2.4 Windows and current counts

The supplied counts are exact for the current root table:

| Patient | Positive | Negative | Ignore |
|---|---:|---:|---:|
| PN00 | 67 | 1612 | 600 |
| PN06 | 58 | 7979 | 600 |
| PN10 | 75 | 12140 | 1200 |
| PN12 | 59 | 3850 | 447 |
| PN14 | 34 | 14175 | 480 |
| Total | **293** | **39756** | **3327** |

There are 43376 decisions, 40049 usable rows, and all 28 event IDs have positive rows. These counts must change after adjudication and interval-boundary corrections; they are an engineering inventory, not an approved analysis cohort.

`windows.py` uses inclusive offsets (`offset >= t`). Four events currently have a positive anchor exactly at offset: PN00-3, PN00-4, PN12-1 and PN12-4. Adopt half-open labels `[onset, offset)` and test onset/offset/sample boundaries. Training context can overlap ictal time even when its anchor is outside; exclusion and label policies must specify both anchor labels and feature support.

`extract_patient_features.py` drops every `label=-1` row. Consequently, existing predictions omit peri-ictal time and cannot be used to reconstruct continuous alarm streams or claim all-time FAR. Deleting ignored rows and treating remaining neighbors as adjacent would also create artificial alarm merging across gaps. The initial 60 s of each EDF are absent; startup coverage must be explicitly accounted for.

### 2.5 Feature and model risks

| Code/evidence | Finding | Required correction before scientific use |
|---|---|---|
| `ecg.bandpass_ecg` | `sosfiltfilt` is zero-phase; peak energy uses centered convolution and refinement ±100 ms | Keep existing outputs OFFLINE; implement separate stateful causal path and beat confirmation timestamps |
| `features.extract_feature_row` | Reads only `[t-context,t)` | Important nuance: current per-window processing does **not** read samples after decision time t. Its within-window operations are noncausal in beat time, but this is not whole-EDF future leakage. Do not exaggerate the flaw |
| `rr_intervals_seconds` / `ecg_features` | Invalid RR intervals are deleted; `diff(rr)` then connects intervals across gaps and `cumsum(rr)` compresses elapsed time | Preserve RR endpoint times and original beat adjacency; compute RMSSD/pNN50 only from contiguous accepted pairs; fit slope against actual timestamps |
| Read-only RR probe | Peaks `[0,100,350,470]` at 100 Hz produce raw RR `[1,2.5,1.2]`, retained `[1,1.2]` and false adjacent difference `0.2` | There are zero valid original adjacent RR pairs; current implementation manufactures one |
| RR filtering | Accepted range 0.30–2.0 s; no normal-beat classification | Plausible RR is not NN. Call variability SDRR/cleaned-RR variability unless sinus beats are adjudicated; retain extreme-HR/arrhythmia flags |
| ECG lead pooling | With two finite leads, median equals arithmetic mean; one poor lead can bias HR | Choose a quality-qualified lead; preserve disagreement instead of averaging away failures |
| ECG agreement | Absolute raw filtered waveform correlation | Not a validated SQI; lead orientation and morphology differ, and shared movement can correlate strongly |
| EEG features | 46 aggregated spectral/time/correlation features; no explicit montage/reference harmonization or EEG quality policy | Fix channel/reference policy and units; retain compact interpretable features with flatline/clipping handling |
| EEG spectral integration | Band integrals exclude upper edges, while total includes 45 Hz | Document and test frequency-bin integration; do not assume relative bands sum exactly to one |
| Benchmark | RF: 300 trees, `min_samples_leaf=2`, balanced classes, fixed seed; train-fitted median imputation | No obvious direct test-fitted imputation; scores are not calibrated natural-prevalence probabilities |
| Feature selection | Prefix-only modality columns | Patient, file, anchor and label metadata are excluded from X in the current benchmark, a positive finding |
| Cache resume | Partial rows reused by `(file,anchor)` without source/config fingerprints | Same-size stale caches can survive preprocessing or annotation changes; hash-based cache identity needed |
| Portability | Hard-coded absolute paths, lower-bound dependency versions, raw-dependent tests | Lock environment and separate synthetic tests from optional data integration tests |

Existing QRS tests cover clean synthetic periodic beats and plausible real segments. They do not establish beat accuracy during seizures, movement, ectopy or signal loss. Existing window tests prove trailing **boundaries**, not end-to-end causal filtering.

### 2.6 PN00 results: arithmetic verified, inference limited

Recomputed ROC-AUC and average precision directly from the saved OOF predictions; all reported values match. Feature cache has 1679 rows, 67 positives, 46 EEG + 19 ECG features, no duplicate `(EDF,anchor)` keys and no nonfinite feature cells. Its keys exactly match PN00 usable windows. This is saved-output verification, not independent raw feature regeneration or model retraining.

| Model | ROC-AUC | Average precision |
|---|---:|---:|
| EEG | 0.953567 | 0.741019 |
| ECG | 0.773504 | 0.087266 |
| Early fusion | 0.948391 | 0.644548 |
| Late fusion, alpha=0.5 | 0.924466 | 0.318365 |

Average precision is the sklearn stepwise summary; it should be named explicitly rather than conflated with trapezoidal PR area. Positive prevalence is 67/1679 = 3.99%; ECG AP is above that descriptive reference, but the data are correlated and the reference is not a significance test. Across the complete pilot usable table, prevalence is only 293/40049 = 0.732%.

At threshold 0.5, **all 45 EEG false-positive windows are in PN00-3**; early fusion has 50 there, and fixed fusion 82. ECG has 149 there, 2 in PN00-2 and 8 in PN00-4. These are window counts, not alarms. This concentration makes recording-specific state drift and annotation policy central. It is not a reason to remove PN00-3 for performance improvement: quarantine must be justified independently by annotation uncertainty, and all models must be retrained when the training cohort changes.

The leave-one-EDF-out loop is parent-disjoint by file name and uses train-only imputation. However, adjacent windows remain strongly dependent within each test EDF; recordings may share session, medication withdrawal, sleep and seizure morphology. There is no event evaluator, nested thresholding, future-session test, or calibration analysis. Pooled OOF scores come from different fitted models and can also suffer between-record score shifts.

**Interpretation:** this particular OFFLINE development run favors EEG over the two naive fusion constructions. It motivates testing conditional ECG usefulness, but does not establish that adaptive fusion will help, that ECG adds independent autonomic information, or that the performance generalizes. PN00 is already a development patient. Neither its smoke experiment nor its full OOF results can be made pristine confirmatory evidence by changing the split now.

## 3. Lessons from the old chb04/6019 branch

The previous audit is preserved at `../eeg-seizure-detection/docs/MULTIMODAL_ECG_INDEPENDENT_AUDIT.md` with evidence JSON. This review inspected both plus the current multimodal source and ignored experimental runner. SHA-256 of those two source files still matches the evidence JSON. Saved evidence confirms the 58→51 arithmetic, five-seed deltas `[-7,-2,+14,+6,-16]`, and EEG-with-early-settings counterfactual 51. The expensive old training audit was **not** rerun.

| Previous failure mode | Why it mattered | New Siena prevention | Test that prevention works |
|---|---|---|---|
| One patient, three events, two seizure parents | Extremely weak independent information; fully nested positive calibration infeasible | Enumerate patient/parent/event units and gate eligibility by number of independent seizure parents | Fail nested-fold construction if inner training lacks positive/negative examples; report excluded/fallback cases |
| One-hour children from the same EDF crossed train/test | 138/140 background test children had training siblings | Whole parent EDF is indivisible at every nesting level; group sessions when verified | Assert zero parent and raw-sample-support intersection, including cached features and preprocessing state |
| EEG history and whole-record ECG processing crossed boundaries | Row disjointness did not mean information disjointness | Reset modality state at parent boundaries; compute test state from test past only | Perturb another fold's data and confirm unchanged test preprocessing |
| Gain concentrated in one background hour | Aggregate result masked unstable heterogeneity | Per-record event/FAR contributions and leave-one-patient-out influence display | Recompute paired summary after removing each cluster as a diagnostic, never as the chosen headline |
| Alarm settings explained apparent ECG gain | EEG alone also produced 51 with early-fusion settings | Freeze identical alarm state machine; primary fusion retains EEG threshold | Counterfactual fixed-score/fixed-rule audit and alpha=0 exact equivalence |
| Seed sensitivity | Early fusion won 3/5 schedules and lost 2/5 | Freeze folds and seeds separately; report all predeclared seeds | Deterministic replay and full seed table, without selecting best seed |
| Poor ECG changed the conclusion | Plausible beat counts did not guarantee physiological information | Quality-qualified lead, quality gating, all-time fallback | Missing/noisy/saturated/inverted/duplicated lead tests; retained exposure identical across models |
| Noncausal processing and backdated delays | Retrospective ≈11-s delays were not online latencies | Separate OFFLINE and CAUSAL tracks; timestamp availability/confirmation | Prefix-invariance and chunk-invariance tests; known-onset synthetic replay |
| Limited calibration and probability semantics | Sampling/weights changed score distributions | Treat outputs as scores; small training-only alpha search | No calibration on outer EDF; reliability plots descriptive; no entropy gate without validation |
| Alarm fragments and selective denominator | Background child hours differed from all nonictal time | Count declarations under one explicit event policy; log exact denominator | Hand-calculated fixtures with pre/ictal/postictal alarms and recording gaps |
| Repeated protocol exploration | Latest favorable result could replace committed negative result | Append-only experiment IDs and phase labels | Duplicate run identity cannot overwrite; every registered seed/config accounted for |
| Public runner hidden in ignored data path | Public checkout could not reproduce result | Public code/config plus manifest-driven private raw data lookup | Clean checkout synthetic smoke and a documented data-dependent command |

The old audit's parent-cluster FAR-difference interval included zero. Repeating seeds does not create new patients. Do not rescue that branch, combine it with Siena to increase sample size, or present its selected 12% reduction as prior evidence of effectiveness.

## 4. State of the literature

### 4.1 Search scope, access and limitations

This is a broad, targeted **engineering evidence review**, not a PRISMA systematic review or meta-analysis. Searches were performed on 22 September 2026, emphasizing 2022–2026 and specifically 2024–2026. Search-engine dates were not treated as publication dates: for example, Sigsgaard & Gu is a 2024 issue with a 2023 DOI; Miron et al. appeared online in 2024 and in a 2025 issue.

Queries covered the following requested themes; exact representative strings are recorded to make the search direction reproducible:

| Themes | Search combinations used | Evidence retained |
|---|---|---|
| EEG+ECG, ECG assistance, Siena prior art, early/late fusion | `EEG ECG fusion seizure detection Siena dataset 2024 2025 2026`; `multimodal EEG ECG seizure detection patient specific late fusion 2022 2023 2024`; exact-title follow-ups | R01–R05, R12–R15 |
| Tachycardia, peri-ictal HR/HRV, autonomic response, temporal/frontal origin and timing | `ictal tachycardia autonomic seizure heart rate onset review 2024 2025`; `ictal tachycardia precedes intracranial scalp 2022 2023 study`; exact cardiac review titles | R06–R11, R16 |
| ECG-only, individualized versus cross-patient detection | `seizure ECG false alarms 2025 detection prospective`; `Detection of seizures with ictal tachycardia Jeppesen 2024 sensitivity` | R01, R03, R07, R08, R13 |
| Adaptive, quality-, confidence- and uncertainty-aware fusion; missing modalities | `multimodal biosignal fusion uncertainty missing modality quality aware 2024 2025 survey`; `seizure detection quality-aware fusion`; `seizure detection uncertainty missing modalities` | R20–R22, R25; indirect transfer, not Siena efficacy proof |
| SQI, QRS motion/noise, two-lead reliability and contamination | `ECG signal quality index QRS detector agreement motion noise 2024 2025 benchmark`; `ECG seizure movement artifacts 2024` | R17–R20, R30 |
| Event sensitivity, FAR/h, causal/offline, online delay | `seizure detection evaluation event scoring false alarms per hour SzCORE 2024`; `EEG ECG seizure two-stage fusion`; `EEG ECG seizure false confirmation` | R07, R08, R14, R23, R26 |
| HRV duration and LF/HF | `ultra short term heart rate variability 30 60 120 seconds RMSSD SDNN pNN50 validity 2024 review`; `heart rate variability 2024 guidelines psychophysiology` | R10, R16, R24 |
| Wearables, physiological alignment, foundation representations | `seizure detection 2022 systematic review multimodal Westrhenen`; `EEG foundation model 2025 seizure detection LaBraM BENDR` | R04–R06, R12, R14, R15, R27, R28 |

Primary publisher pages, PubMed/PMC, author/institutional full texts and conference proceedings were prioritized. Core full texts were inspected through publisher HTML/PDF or Europe PMC's fullTextXML when the PMC browser endpoint failed. Reading an abstract or indexed passage is explicitly distinguished below. No unavailable metric was inferred from another paper's discussion. Reviews synthesize evidence; they are not independent replications. Preprints are labeled.

The following three linked tables form one literature extraction table keyed by **R ID**. `NR` means **not reported in the material inspected**; where full text was unavailable this is not a claim that the entire paper omits it. `N/A` means the work is not a seizure detector or has no applicable event metric. EEG used for ground truth is distinguished from EEG input.

### 4.2 Literature table A: citations, populations and modalities

| ID | Citation and link | Year | Dataset; patients; seizures | EEG / ECG inputs; other modalities | Access |
|---|---|---|---|---|---|
| R01 | [Sigsgaard & Gu, Comparison of patient non-specific seizure detection using multi-modal signals](https://doi.org/10.1016/j.neuri.2023.100152) | 2024 | Siena; 14; 47 | Yes / yes; none | Full publisher PDF via DTU |
| R02 | [Yang et al., Patient-specific approach using data fusion and adversarial training for epileptic seizure prediction](https://doi.org/10.3389/fncom.2023.1172987) | 2023 | Siena; 8 selected; evaluated episode count NR | Yes / yes; none | Full text/XML |
| R03 | [Vandecasteele et al., The power of ECG in multimodal patient-specific seizure monitoring](https://pmc.ncbi.nlm.nih.gov/articles/PMC8518059/) | 2021 | SeizeIT1 + Epilepsiae Freiburg/Paris; 135; 896 | Limited temporal EEG / single-lead ECG | Full text/XML |
| R04 | [Nielsen et al., Towards a wearable multi-modal seizure detection system](https://doi.org/10.1016/j.clinph.2022.01.005) | 2022 | 30 recruited, 3 classifier-eligible; 47+9+9 analyzed events | Behind-ear EEG / ECG; ACC | Abstract + indexed author PDF |
| R05 | [Nielsen et al., Out-of-hospital multimodal seizure detection](https://doi.org/10.1136/bmjno-2023-000442) | 2023 | 17 monitored; seizure analysis one patient, 15 electrographic seizures | Behind-ear EEG / ECG; ACC | Full text/XML |
| R06 | [Seth et al., Feasibility of cardiac-based seizure detection and prediction](https://doi.org/10.1002/epi4.12854) | 2024, online 2023 | Systematic review: 24 studies; no single cohort | Varies / ECG or PPG; wearable combinations | Full text |
| R07 | [Jeppesen et al., Detection of seizures with ictal tachycardia…hospital-based validation](https://doi.org/10.1002/epd2.20196) | 2024 | Brazilian external test: 34 analyzed; 107 usable seizures; responder subgroup 22/59 | EEG reference / ECG input | Full author-hosted PDF |
| R08 | [Jeppesen et al., Wearable ECG connected to a smartphone: phase 3 validation](https://doi.org/10.1016/j.ebiom.2025.105952) | 2025 | 101 enrolled; 17 eligible responders, 42 seizures | EEG reference / wearable ECG; behavioral confirmation | Full text/XML |
| R09 | [Miron et al., Autonomic biosignals, seizure detection, and forecasting](https://doi.org/10.1111/epi.18034) | 2025 issue, online 2024 | Narrative review; no single cohort | Varies / cardiac signals; EDA, ACC, temperature | Full sections |
| R10 | [Mason et al., Heart Rate Variability as a Tool for Seizure Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10856437/) | 2024 | Scoping review; 72 studies | Varies / HRV; heterogeneous sensors | Full indexed review sections |
| R11 | [Impaired brain-heart axis in focal epilepsy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11168720/) | 2024 | Siena: 14/47 source; **13/38 analyzed** | EEG / ECG-derived HRV | Full text/XML |
| R12 | [Karasmanoglou et al., Semi-supervised anomaly detection…fused EEG and ECG](https://doi.org/10.1016/j.bspc.2024.107083) | 2025 | Pediatric focal epilepsy; counts NR in inspected material | Focal EEG / HRV | Abstract/indexed passages; author PDF access failed |
| R13 | [Reintjes et al., ECG-Based Detection…SeizeIT2 Dataset](https://doi.org/10.3390/s25247687) | 2025 | SeizeIT2; 125; 886; 11640 h source corpus | EEG reference / wearable ECG | Full text/XML |
| R14 | [SeizeIT2 multicenter wearable validation](https://doi.org/10.1002/epi4.70260) | 2026 | 192 recruited; 616 focal seizures | Behind-ear EEG / ECG | Full text/XML |
| R15 | [Zhang et al., Multimodal wearable EEG, EMG and accelerometry](https://pubmed.ncbi.nlm.nih.gov/38772401/) | 2024 | Multicenter tonic-clonic study; counts NR in inspected final abstract | EEG / ECG recorded; best combination EEG+EMG+ACC | Final abstract; preprint screened separately |
| R16 | [Quigley et al., Publication guidelines for human HR and HRV](https://doi.org/10.1111/psyp.14604) | 2024 | Measurement guidelines; N/A cohort | N/A / ECG measurement | Full guideline sections |
| R17 | [Kristof et al., QRS detection in single-lead telehealth ECG](https://doi.org/10.1371/journal.pdig.0000538) | 2024 | Six ECG datasets; 995 recordings, not 995 patients | No / ECG | Full publisher text |
| R18 | [A reproducible benchmark of QRS algorithms across datasets/noise](https://pubmed.ncbi.nlm.nih.gov/42162090/) | 2026 | Five PhysioNet datasets; patients NR | No / ECG | PubMed abstract |
| R19 | [Assessment of ECG Signal Quality Index Algorithms Using Synthetic ECG](https://cinc.org/archives/2024/pdf/CinC2024-270.pdf) | 2024 | Synthetic ECG; clinical cohort N/A | No / ECG | Indexed conference paper |
| R20 | [Shahbakhti et al., Pulse-based PPG quality assessment improves seizure detection](https://doi.org/10.1002/epi4.70242) | 2026 | Pediatric nocturnal motor seizures; counts NR in inspected primary material | No / no ECG; PPG | Publisher abstract/indexed passages |
| R21 | [Multimodal Fusion on Low-quality Data: A Comprehensive Survey](https://arxiv.org/abs/2404.18947) | 2024 preprint; 2026 journal version located | General multimodal tasks; N/A cohort | Varies; not an epilepsy trial | Abstract and indexed sections |
| R22 | [Deep Multimodal Learning with Missing Modality: A Survey](https://arxiv.org/abs/2409.07825) | 2024 preprint | General multimodal tasks; N/A cohort | Varies | Abstract |
| R23 | [Dan et al., SzCORE evaluation framework](https://arxiv.org/html/2402.13005v2) | 2024 preprint; Epilepsia version also located | CHB-MIT, Siena, TUSZ, SeizeIT1 benchmark framework | EEG / not an ECG-fusion trial | Full framework text |
| R24 | [Burma et al., Validity and reliability of ultra-short HRV](https://doi.org/10.1152/japplphysiol.00955.2020) | 2021 | Physiological validation; counts NR in inspected material | No / cardiac intervals | Indexed primary study passages |
| R25 | [Su et al., CMEpiNet](https://doi.org/10.3390/s26134186) | 2026 | SeizeIT2; source 125; split 96 train/28 test, sub-097 excluded; analyzed seizures NR | EEG / ECG; EMG | Full text/XML methods |
| R26 | [Sang et al., A multi-feature method…pediatric intensive care unit](https://doi.org/10.1002/epi4.70171) | Online 2025 | 28; 1561 annotated seizures; 218.73 h | EEG / ECG; EMG | Full text/XML methods |
| R27 | [Jiang et al., Large Brain Model (LaBraM)](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf) | 2024, ICLR | ~2500 h pretraining across ~20 EEG datasets | EEG / no | Proceedings abstract/method overview |
| R28 | [Turgut et al., Are foundation models useful EEG feature extractors?](https://arxiv.org/abs/2502.21086) | 2025 preprint, revised 2026 | Multiple EEG tasks; seizure-specific counts NR | EEG / no | Versioned abstract |
| R29 | [Handa et al., Meta-EEGs](https://doi.org/10.1016/j.mex.2026.104005) | 2026 | CHB-MIT and Siena derivatives; Siena 46 retained events | EEG / no detector | Full text/XML |
| R30 | [Böttcher et al., Effects of seizures on wearable biosignal quality](https://doi.org/10.1111/epi.18138) | 2024 | Wearable seizure recordings; counts NR in inspected material | Reference EEG; wearable biosignals | Indexed primary methods/discussion |
| R31 | [Eggleston et al., Ictal tachycardia: the head-heart connection](https://doi.org/10.1016/j.seizure.2014.02.012) | 2014 | Review of 34 studies | EEG/ECG physiology | PubMed abstract + indexed full-text table |

### 4.3 Literature table B: representations, fusion and split

| ID | Patient-specific or cross-patient; split | ECG representation | EEG representation | Fusion; causal status |
|---|---|---|---|---|
| R01 | Cross-patient; leave-one-patient-out | 31 RR/HRV features, 60-s windows | Time/frequency/entropy, 6-s epochs; RF | OR of separate RF detectors; retrospective, no verified causal replay |
| R02 | Patient-specific; leave-one-episode-out, not proven parent-EDF-disjoint | ECG wavelet image + CNN | BiLSTM/CNN | Error-weighted decision fusion/adversarial training; offline prediction |
| R03 | Patient-specific EEG seizure-LOO; ECG patient-LOO and external-center testing | HRV/HR-change RF | Engineered SVM on limited channels | OR late integration; explicitly offline |
| R04 | Patient-specific SVM; full independent-parent nesting NR | Cardiac features; full detail NR | Engineered behind-ear features | Multimodal SVM; retrospective |
| R05 | Patient-specific; hospital-to-home and within-home LOO experiments | HR/ECG features | Wearable EEG features | SVM multimodal; retrospective |
| R06 | Review; heterogeneous validation | HR and/or HRV | Varies | Single/multimodal; predominantly retrospective |
| R07 | External Danish training to Brazilian test; patient-adaptive settings | HRV and adaptive logistic regression | Reference only | ECG-only; retrospective hospital validation |
| R08 | Prospective predefined algorithm; first-24-h personalization | 100-RR Lorenz/CSI-derived features | Reference only | ECG alarm then behavioral test; real-time study |
| R09 | Narrative synthesis | HR, HRV and other autonomic indices | Varies | Mechanisms and study-phase comparison |
| R10 | Scoping synthesis | Peri-ictal HRV | Varies | Prediction and detection kept distinct |
| R11 | Paired pre/post physiology, not predictive held-out evaluation | Kubios RR, interpolated series | Cortical oscillations | Brain-heart generative analysis; offline 10-min pre/post segments |
| R12 | Patient-specific anomaly framework; exact independent split NR | HRV | Temporal/spectral/nonlinear focal features | Multimodal anomaly detection; offline |
| R13 | Patients 001–096 validation, 097–125 test | ECG anomaly representations | Reference only | Matrix Profile/MADRID/TimeVQVAE-AD; retrospective |
| R14 | Multicenter validation; not Siena cross-validation | ECG plus EEG detector | Behind-ear EEG | Offline algorithm then blinded expert review |
| R15 | Final abstract: full split NR | ECG not in best reported combination | Wearable EEG features | EEG/EMG/ACC fusion; offline |
| R16 | Measurement guideline | HRV definitions, quality, stationarity | N/A | N/A |
| R17 | Manual-beat-reference benchmark | 18 QRS detectors | N/A | No fusion; tests signal-quality dependence |
| R18 | Five-dataset noise benchmark | 17 R-peak algorithms | N/A | No seizure fusion; end-to-end online status NR |
| R19 | Controlled synthetic signal/noise | SQI algorithms | N/A | No seizure fusion |
| R20 | Quality before HR estimation | PPG pulse quality, not ECG SQI | N/A | Wearable quality study; no Siena validation |
| R21 | General survey | Varies | Varies | Noisy/incomplete/imbalanced/quality-varying fusion |
| R22 | General survey | Varies | Varies | Missing-modality training/inference strategies |
| R23 | Chronological personalized and subject-independent evaluation | N/A | Framework-agnostic | Event/sample scoring conventions, not proof of online implementation |
| R24 | Segment-length validation against longer recordings | HR, RMSSD, SDNN, pNN50 | N/A | No fusion; validation context differs from ictal transitions |
| R25 | Subject-disjoint official holdout before segmentation | Complex-valued learned features | Complex-valued features | Intermediate semantic alignment + 3D attention; offline |
| R26 | Clinical validation; independently nested tuning split not established here | HR/HR bursts | Energy candidates and temporal correlation | Candidate screening then weighted confirmation; claimed real-time, not independently causality-audited |
| R27 | Multi-dataset pretraining/downstream adaptation | N/A | Tokenized masked EEG transformer | No ECG fusion; online behavior task-dependent |
| R28 | Frozen feature comparison across EEG tasks | N/A | General-purpose time-series embeddings | No ECG fusion; preprint |
| R29 | Data processing, no model split | N/A | Event segmentation/metadata | No fusion |
| R30 | Pre/ictal/post signal-quality comparison | Cardiac wearable signal reliability | Reference/context | No primary EEG+ECG detector comparison |
| R31 | Physiological review | Ictal HR change/tachycardia | Onset reference | No fusion algorithm |

### 4.4 Literature table C: reported outcomes and reviewer interpretation

FAR conversions below are explicitly arithmetic conversions of reported daily rates; they are not newly estimated pooled rates. Different event definitions and denominators prevent leaderboard comparison.

| ID | Reported event sensitivity | Reported FAR/h | Delay | Reported finding → reviewer interpretation / limitation |
|---|---|---|---|---|
| R01 | Mean EEG 81.43%, ECG 41.55%, fusion 87.62% | Means 3.61 / 3.25 / 6.82 | Means 20 / 36.5 / 19.3 s | Sensitivity gain with substantially more alarms; OR fusion is not evidence of suppression |
| R02 | Event sensitivity NR; 99.76% is sample sensitivity | **NR**: reported 0.001 is false-positive window fraction | NR | Prediction task; cannot relabel its FAR as alarms/hour or infer EDF independence |
| R03 | Gains of 11 and 8 percentage points at matched FAR in two datasets | Operating points 0.2/0.5/1 considered | NR here | Positive limited-montage evidence; not guaranteed for full scalp EEG or all centers |
| R04 | 84%; 100%; 100% in three patients | 8/24, 13/24, 5/24 ≈ 0.333, 0.542, 0.208 | NR | Feasible, highly selected small sample; many events from one person |
| R05 | 91% in one patient's transfer experiment | 18/24 = 0.75 | NR | Only 30.1% of EEG classified usable; home quality and one-patient inference limit transfer |
| R06 | HR-study range 56–100% | Range 0.02–8 | Heterogeneous/NR | 24 studies, search through Aug 2022; substantial heterogeneity and limited prospective evidence |
| R07 | Responders 84.8%; 52.6% if technically excluded responder seizures count as misses | 0.25/24 ≈ 0.0104 in responders | NR here | Quality exclusion can materially change sensitivity; external test is valuable but selected/fragmented data matter |
| R08 | 90.5% of 42 eligible seizures | Mean 2.5/24 ≈ 0.104; median patient 1.1/24 ≈ 0.046 | Median 28 s | Nonresponders: 16.4% of 73 seizures; first-seizure selection and 24-h adaptation restrict generalizability |
| R09 | N/A | N/A | N/A | Autonomic dynamics are heterogeneous and confounded; separate detection from forecasting |
| R10 | No single pooled detection estimate | Heterogeneous | Heterogeneous | Supports physiological study, not interchangeable HRV biomarkers or metrics |
| R11 | N/A | N/A | N/A | Brain-heart associations do not establish predictive incremental information; 10-min retrospective analysis |
| R12 | NR in inspected material | NR | NR | Relevant anomaly-fusion alternative; do not promote ROC-AUC claims to event performance |
| R13 | Extended-window examples: 98.16% Matrix Profile; 92.86% TimeVQVAE-AD | 13.9 and 15.25 respectively | NR here | Uses −5 to +3 min onset window in one track; distinguish strict results and separately optimized objectives |
| R14 | Algorithm 447/616 ≈ 73%; after review 193/616 ≈ 31% | Algorithm mean 6.22 | NR here | Algorithm precision 0.004; reviewed precision 0.83. Human review and subgroup selection are not automatic detector efficacy |
| R15 | Best reported EEG+EMG+ACC combination 90.9% | 0.1/24 ≈ 0.00417 | NR | Strong motor-seizure result; cannot attribute it to ECG or generalize to nonmotor focal seizures |
| R16 | N/A | N/A | N/A | LF/HF is not a selective sympathetic measure; disclose preprocessing and physiological assumptions |
| R17 | N/A—beat F1, not seizure sensitivity | N/A | N/A | Several detectors deteriorate on low-quality ECG; synthetic plausibility is insufficient |
| R18 | N/A | N/A | N/A | Noise/dataset benchmark supports detector stress testing, not selection by seizure performance |
| R19 | N/A | N/A | N/A | Useful for SQI test design; synthetic noise is not a clinical quality ground truth |
| R20 | NR in inspected primary material | NR | NR | Quality-before-HR is relevant; PPG results do not validate ECG thresholds |
| R21 | N/A | N/A | N/A | General design taxonomy; no evidence that learned weighting fits five Siena patients |
| R22 | N/A | N/A | N/A | Missing-modality robustness needs explicit testing; do not impute a confident normal ECG |
| R23 | Framework, not a single result | Framework | Framework | Standardization valuable; permissive event tolerances must not become negative online latency |
| R24 | N/A | N/A | N/A | Ultra-short validity is metric/context-specific; pNN50 particularly variable |
| R25 | NR extracted here | NR extracted here | NR | Subject-disjoint design is relevant, but complex architecture and EEG/EMG contributions do not isolate ECG value |
| R26 | Reports 94% | Reports 0.18 | NR | Abstract also states 1095 algorithm vs 1561 annotated events; aggregation/matching require reconciliation before comparison |
| R27 | N/A for this Siena question | NR | NR | Representation option only; pretraining overlap, reference and compute must be audited |
| R28 | NR event endpoint | NR | NR | Feature transfer promising; no evidence of benefit in this cohort |
| R29 | N/A | N/A | N/A | Independent evidence of PN00-3 annotation difficulty, not validation of an alternative end time |
| R30 | N/A | N/A | N/A | Quality can vary with seizures; deleting poor-quality ictal time causes selection bias |
| R31 | N/A | N/A | N/A | Review reports tachycardia in ~82% of patients; not 82% of every patient's seizures |

### 4.5 What prior Siena studies mean for novelty

R01 is the direct detection comparator. It uses patient-LOO rather than our planned patient-specific recording-LOO. Its ECG features include short-window frequency-domain HRV; RR outliers are interpolated. It changes overlap by seizure label and merges detections within 30 s. A fully nested threshold-selection rule and causal replay are not established by the inspected methods. These are reasons to define our protocol independently, not allegations of fabricated results.

R02 already uses adaptive decision fusion on Siena, but for preictal-versus-interictal prediction; its “episode” need not equal an EDF. Its FAR equation is a window fraction. R11 investigates physiological coupling rather than a detector. Thus the defensible distinction is **incremental event-detection value under audited recording isolation, missing-ECG fallback and availability-time scoring**, not a novel claim to combining the modalities.

Recent evidence also argues against equating complexity with credibility. R25 supplies a neural fusion comparison worth acknowledging; R14 supplies a much larger validation with difficult false-alarm tradeoffs. Neither justifies a small Siena neural gate. R26 provides direct precedent for EEG candidate screening followed by multimodal confirmation, but its PICU electroclinical population and evaluation details differ substantially. A two-stage design is a testable hypothesis here, not an established superior architecture.

## 5. Physiological rationale and representations

### 5.1 What ECG can measure

Seizures can engage cortical/subcortical autonomic networks, changing cardiac timing. Tachycardia is common but definitions, seizure types and measurement reference differ. R31 reports about 82% of patients, with average seizure-level proportions around 64% for generalized and 71% for partial-onset seizures; these older heterogeneous figures are not Siena priors. Temporal seizures often show autonomic changes, but frontal seizures can also do so. Siena contains essentially one frontal patient, so localization cannot be separated statistically from identity.

HR changes sometimes precede scalp-electrographic or clinical onset, but this can reflect earlier deep ictal activity rather than genuine prediction. Responses may instead emerge tens of seconds later, or be absent. Bradycardia/asystole are uncommon but real; low HR must not automatically mean “no seizure.” HRV can change before, during and after events, with respiration, sleep, arousal, medication and movement influencing both HR and variability. R09/R10 provide the synthesis; R08 demonstrates practical responder/nonresponder heterogeneity.

**Reviewer interpretation:** ECG is best framed as a patient-dependent autonomic trajectory and auxiliary evidence, not a necessary condition for a seizure. A rise in heart rate is not specific to epilepsy; a lack of rise is not reliable negative evidence. Relative HR may reduce baseline differences but cannot remove activity, sleep or medication confounding. Do not claim that a fusion benefit proves an autonomic mechanism unless QRS quality and artifact controls support that interpretation.

### 5.2 ECG representation decision

| Candidate | 30 s | 60 s | 120 s | Recommended status |
|---|---|---|---|---|
| Median RR and median instantaneous HR | Reasonable if enough clean beats | Reasonable | Reasonable, slower response | Primary; avoid including redundant mean HR + mean RR + median HR without purpose |
| HR change/slope | Responsive but noisy | Good compromise | Useful sustained-trend comparison, diluted onset | Primary at 60 s; 30/120-s alternatives as limited ablations |
| Relative HR change to past baseline | Useful | Useful | Useful but delayed | Primary with explicitly causal reference |
| RMSSD of consecutive accepted intervals | Exploratory/unstable in sparse or noisy windows | Secondary descriptor with sufficient adjacent pairs | More support but greater nonstationarity | Include one log-RMSSD60 term only after beat validation; not “vagal tone” ground truth |
| SDRR / RR CV | Descriptive; length-sensitive | Descriptive | Descriptive; slower trends increase variance | One RR CV60 term primary; avoid claiming clinical five-minute SDNN equivalence |
| pNN50 | Highly discrete, artifact-sensitive | Still unstable/redundant | Possible sensitivity-only descriptor | Omit primary |
| LF, VLF, LF/HF; entropy/fractal HRV catalogue | Unsupported primary choice | Unsupported primary choice | Still no automatic validity | Omit primary; longer windows do not cure confounding or stationarity assumptions |
| Raw QRS morphology/amplitude/baseline shifts | Signal/lead/artifact sensitive | Same | Same | QC diagnostics, not primary seizure predictors |
| Small CNN/temporal embedding | Many windows but few independent trajectories | Same | Same | Postpone; external pretraining plus frozen embedding only as later controlled experiment |

These are proposed engineering roles, not universal minimum-duration standards. R16 discourages interpreting LF or LF/HF as selective sympathetic activity; R24 shows ultra-short reliability varies by metric. Even where resting short-window RMSSD is acceptable, rapid ictal transitions, ectopic beats and motion need separate validation. With LF lower edge 0.04 Hz, a 60-s window contains only 2.4 cycles at that edge; numerical spectral output is not evidence of physiological reliability.

**Compact primary candidate feature set (maximum eight physiological variables):** median HR60, robust HR slope60 against real beat times, HR60 minus local baseline, log HR60/baseline ratio (retain only one of these two highly redundant relative terms after a *predeclared engineering* review), recent-30-s HR minus previous-30-s HR, RR CV60, log-RMSSD60, and an optional sustained-HR-change duration. Beat count, valid-pair count, lead choice, missingness and SQI are logged separately; they should not silently become seizure predictors. Use a maximum of seven if the redundant relative feature is dropped. Freeze the exact list before expansion.

**Baseline definition proposed for pilot:** robust median of quality-accepted RR-derived HR endpoints in `[t−360, t−60)`, using only the current recording's past. Require at least 120 s of accepted coverage, not merely a number of duplicated beats. Before availability, relative features are missing and the primary fusion gate is zero; EEG continues. Never use complete-EDF z-scores, test labels to remove seizures from this baseline, or a baseline centered around a known event. Log baseline age and coverage. A baseline may contain an unrecognized previous seizure: that is a real deployment limitation, not something to fix using test annotations.

For exploratory event-aligned physiology only, calculate a fixed pre-onset reference and HR changes in prespecified pre/onset/post bins, report coverage and avoid overlapping neighboring events. Such summaries can use annotations to explain outcomes after evaluation; they must not feed the detector or determine held-out alpha.

### 5.3 Window recommendation

Keep **EEG trailing 10 s, ECG trailing 60 s, stride 5 s** as the initial primary candidate. These values are reasonable compromises, not empirically established optima. The ECG window at early seizure time contains mostly preictal beats, so its mean dilutes rapid changes; recent-versus-earlier contrasts address this more economically than a large temporal encoder.

Prespecify only these sensitivity axes: EEG 5/10/20 s; ECG 30/60/120 s; stride 1 versus 5 s if resources allow; training exclusion 0 versus 300 s. Do not run their full Cartesian product. First compare one axis at a time on the development cohort and freeze one configuration. Different ECG contexts require matching evaluation exposure, startup policy and baseline availability.

Use `[onset,offset)` anchor labels for training state classification. Retain the ±300-s exclusion for **training negatives only**, subject to a no-exclusion ablation. At inference, score every evaluable anchor; peri-ictal predictions remain visible and contribute to strict false alarms outside annotated events. Annotation-uncertain intervals are a separate category from training exclusions and require explicit denominators.

### 5.4 EEG baseline hierarchy

| Level | Representation/model | Role and constraint |
|---|---|---|
| 1A | Compact log-power, spectral shape, line length, Hjorth and selected correlation features + L2 logistic regression | Interpretable reference; train-only imputation/scaling; fixed C=1 initially |
| 1B | Same features + controlled RF | **Primary credible comparator**; freeze a modest capacity, e.g. 300 trees, depth 6, minimum leaf 5, balanced weights; these are proposed defaults, not tuned results |
| 2 | One stronger engineered EEG model: bounded gradient boosting or RF with fixed spatial-region summaries | Secondary robustness: ECG contribution must not exist only against a weak baseline; match fusion comparison to this same EEG model |
| 3 | Compact CNN on raw or STFT EEG, or frozen externally pretrained encoder | Optional after primary analysis; separate hypothesis and compute budget; no end-to-end small-cohort architecture search |

The existing 46-feature path is a useful starting point, with redundant activity/RMS/variance and average-over-channels potentially hiding focal events. Prefer a small predefined set of regional summaries over hundreds of per-channel or pairwise inputs. Pairwise correlation is a synchrony proxy affected by common reference and volume conduction, not proof of neuronal coupling.

Raw flattened EEG plus classical ML is high-dimensional and sensitive to phase/position. STFT offers inspectable time-frequency patterns but introduces many correlated inputs; wavelet/scalogram choices add another search layer. A compact CNN is not statistically justified merely because 40049 windows exist. LaBraM and later representation work (R27/R28) make frozen embeddings conceivable, but require documented pretraining data, Siena-overlap checks, channel/reference/sample-rate compatibility, licensing and paired EEG/fusion evaluation. A strong EEG model is welcome when controlled; it must not be weakened to preserve a fusion story.

For the main within-patient analysis use the official declared EEG channels, with channel-number provenance and a frozen reference policy. Any rereferencing must use only simultaneous channels at time t, never whole-record fitting. Keep an explicit record of acquisition reference where known; do not invent it from labels. For secondary cross-patient analysis use the intersection of **officially eligible, reconciled canonical electrodes** across training/held-out datasets, fixed before outcomes. The tested 19-label helper is a candidate, not authorization to use undeclared channels. Use regional aggregation if the verified intersection is smaller, with one fixed output schema.

## 6. Recommended fusion architecture

### 6.1 Ranking for this cohort

| Rank/role | Architecture | Advantages | Main risks and decision |
|---|---|---|---|
| Primary | Quality-aware adaptive late score fusion, alpha may be zero | Interpretable, few parameters, independent modality QC, exact EEG fallback possible | SQI and alpha still need validation; quality is not evidence of usefulness |
| Mandatory ablation | Training-only adaptive late fusion without quality modulation | Isolates whether weighting alone helps | Can be damaged by bad ECG; tiny inner event counts make selected alpha unstable |
| Mandatory comparator | Early feature fusion | Simple interactions; compact LR/RF comparison | Scale LR inputs train-only; feature-count dominance in RF; missing ECG needs explicit policy; cannot automatically equal EEG when ECG disappears |
| Secondary hypothesis | EEG candidate + ECG confidence adjustment | Tests autonomic confirmation/suppression directly; efficient | Cannot recover events absent from candidates; risk of vetoing nonresponders and added delay |
| Sanity comparator | Fixed late alpha=0.5 | Transparent negative control | Equal weight is arbitrary when modality strength/calibration differ |
| Exploratory | Confidence-aware weighting | Could reflect uncertainty | RF vote entropy is not calibrated epistemic uncertainty; calibrator/gate data are insufficient |
| Postpone | Learned gates/intermediate neural fusion | Flexible nonlinear/time interactions | Too many degrees of freedom for 5–12 patients and few independent seizures |

Late fusion has the conceptual form `sF(t) = (1 − alpha*q(t))*sE(t) + alpha*q(t)*sC(t)`, with `q∈[0,1]` and alpha fixed per outer training fold. Do not describe `sF` as a calibrated probability by notation alone. A clean ECG may still be physiologically uninformative. Allow alpha=0, and report how often it is selected.

**Fallback must cover scores, thresholds and alarm state.** Setting q=0 while applying a separately optimized fusion threshold does not necessarily recover the EEG detector. Therefore the proposed primary comparison retains the inner-selected **EEG threshold** for quality-aware and adaptive late fusion. Early fusion and ECG-only receive their own training-selected thresholds. A secondary conventional blend may jointly select alpha and a fusion threshold, but it must route poor-ECG decisions through the EEG operating rule and report transition behavior separately. Entire-ECG-outage replay from the same initial state must reproduce EEG alarms exactly.

### 6.2 Two-stage hypothesis, precisely bounded

Use a training-side high-sensitivity EEG candidate threshold and preserve all candidate times. At each candidate, inspect ECG information already available up to that decision. Fit at most a regularized linear adjustment or choose a deterministic adjustment rule using **inner-OOF candidates**, including both false and true candidates. Training the second stage on in-sample EEG candidates would make them unrealistically easy.

A primary two-stage comparison should use no waiting period: it reranks/confirms candidates with past-only ECG. A distinct delayed-confirmation variant may wait up to 15 s, but must declare the alarm at confirmation and score short events missed before offset as misses under the strict metric. Never backdate to candidate onset. Report candidate-detector sensitivity as the stage-two ceiling.

Prefer soft adjustment to a universal veto. Missing/poor ECG must pass through the EEG candidate policy. Absence of tachycardia is not sufficient to reject an otherwise strong EEG seizure. The secondary model must be compared with an EEG-only candidate reranker of comparable capacity; otherwise a gain may arise from a second model or changed alarm rules rather than ECG.

### 6.3 Temporal alignment

Both modalities share the EDF timebase but need not respond simultaneously. Every decision uses EEG `[t−10,t)` and ECG `[t−60,t)` plus the explicitly older baseline; optional 30/120-s features are still trailing. Do not optimize per-test-seizure alignment, shift ECG using the known onset, or choose whichever delay gives the best correlation. A lag sensitivity analysis uses a small prespecified set of **past-only** offsets and remains exploratory. Event-aligned physiological plots may show future responses, but cannot be confused with online features.

## 7. Signal-quality framework and two ECG leads

### 7.1 Proposed transparent score

Maintain per-lead raw quality flags and beat reliability over trailing 60 s. The following formula is a **candidate engineering SQI requiring validation**, not a published validated medical score:

`q_lead = hard_valid * coverage * detector_agreement * morphology_consistency`

Each term is clipped to `[0,1]`. `hard_valid=0` for missing/nonfinite signal, sustained flatline, saturation/clipping, insufficient beat evidence or unusable QRS SNR. Coverage represents duration supported by credible RR observations. Detector agreement uses matched beat times within a frozen tolerance (initial candidate 80 ms) from two independently implemented detectors; matching must be one-to-one. Morphology consistency measures robust QRS-template similarity/prominence with past-only templates. Do not optimize these terms on held-out seizure detection outcomes.

Log physiological plausibility fraction, RR-outlier rate, baseline-noise ratio, high-frequency/powerline energy and inter-lead beat agreement as additional components. They may trigger warnings without being multiplied into the score twice. Thresholds should be set using blinded development signal-quality/beat labels and synthetic corruptions, then frozen; report the thresholds and their sensitivity analysis. A universal quality threshold cannot be justified from current repository tests.

HR outside a typical range and strong RR variability can be real arrhythmia. Distinguish **unreliable measurement** from **reliable unusual rhythm**: the latter can disable a seizure HR model while retaining a separate flag, not be silently edited into normal RR. Exclude LF/HF from this SQI; a seizure-related autonomic change is not automatically noise.

### 7.2 Lead policy

Choose the lead with the highest validated q using only past information. Add deterministic hysteresis (candidate: switch only if the alternative exceeds current q by 0.15 for three consecutive decisions) to prevent oscillation. A failed current lead can switch immediately to a qualified alternative. Beat buffers and previous lead identity must be retained; do not concatenate incompatible RR sequences across a lead switch.

If both leads are high quality, use their **matched beat timing agreement** to corroborate reliability. With one clearly good lead, retain that lead rather than punishing it solely because the other failed. With two supposedly good but discordant leads, flag uncertainty and lower/disable ECG contribution until resolved. If neither is acceptable, set q=0 and follow EEG. Avoid raw waveform averaging, feature median across just two leads, doubling features by concatenating both, or deriving a synthetic lead without electrode-placement justification.

### 7.3 Quality validation and selection bias

Before full experiments, manually annotate a fixed, small development panel across patients, leads, peri-ictal/interictal states and good/poor signal. A practical starting panel is two 60-s segments per lead per pilot patient plus every clear detector-disagreement case in that panel; this is a proposal, not work completed here. Keep a blinded holdout of those segments for beat/SQI validation. Document who labeled beats and uncertainty. Automated artifacts, inversion, clipping, flatline, dropped samples and missed/double beats supplement but do not replace real ictal checks (R17–R19).

Low-quality ECG must **not remove difficult seizures or recording time from the primary comparison**. Report q=0 fraction and ictal/interictal coverage separately. An ECG-only algorithm may abstain; its all-time seizure sensitivity must count relevant missed events, while conditional-on-available performance is secondary. Distinguish EEG unavailability from ECG fallback and report system coverage. Shared motion can create both EEG and ECG artifacts; include ECG-quality-only and nuisance-feature controls before calling a benefit autonomic.

## 8. Validation protocol and training-only tuning

### 8.1 Independent units and eligibility

The minimum grouping unit is `(patient, parent EDF)`, supplemented by a session identifier when known. All windows, beats, caches and seizures within PN10-4.5.6, PN10-7.8.9 or PN12-1.2 stay together. Leave-one-seizure-out is not equivalent to leave-one-recording-out: another seizure and shared background from the same EDF can remain in training. It is unsuitable as the primary scheme here.

Parent-disjointness does not guarantee independent sessions or future time. De-identified dates and clock inconsistencies require explicit session reconstruction; if chronology is uncertain, label analysis recording-held-out retrospective. Never infer future-session generalization from EDF separation alone.

For fully patient-specific nested LORO require at least **three eligible seizure-bearing parent recordings**, with both classes available in every inner fit. Three parents make the mathematics possible but inner fits have only one seizure-bearing parent: PN12 is particularly fragile. Freeze model/feature/SQI choices globally on development, restrict per-fold adaptation to a tiny alpha/threshold procedure, and report calibration support. Two parents support outer comparison only with externally frozen settings, not inner positive calibration. One parent cannot support independent-record patient-specific testing.

### 8.2 Proposed nested procedure

For each eligible patient and each held-out parent R:

1. Freeze outer training/test IDs and annotation eligibility. Never access R labels through model-selection functions.
2. Within outer training, leave one whole parent out at a time. Fit imputers, scalers, any selections and model on the inner-training parents only. Compute continuous predictions for the complete inner-validation parent, including near-seizure time.
3. Use the same training rows, fold manifest and seed schedule across EEG, ECG and early fusion. Negative sampling, if needed, happens only in each fit. Reset processing and alarm state per parent. Concatenate **event statistics**, not stateful streams across parents.
4. Pool inner-OOF event counts and exposure under the frozen alarm policy. Select the EEG threshold with the rule below. Obtain ECG/early thresholds independently by that same rule.
5. Select alpha from `{0, 0.1, 0.25, 0.5}` using inner-OOF predictions, the retained EEG threshold and the frozen quality score. A full `{0,0.1,…,1}` grid is a secondary sensitivity analysis, not another chance to choose a favorable primary method.
6. Fit each base model on all outer-training parents, with its training-only preprocessing. Freeze threshold, alpha, SQI, feature schema and alarm policy. Apply exactly once to R. Record all scores and quality states before evaluating labels.
7. Aggregate each recording exactly once for the declared seed/config. Retain all failed folds and fallback decisions with reasons; never silently drop them.

Prediction-only invariance tests must show that changing outer labels does not alter fitted models, selected alpha/threshold, features or alarms. Changing validation labels may change selection but not the base features. If a class is absent, mark the fold ineligible or use a prespecified external fallback; do not silently let `predict_proba[:,1]` or AUC fail and omit the result.

### 8.3 Exact proposed operating rule

Use a **research operating target of 1 false alarm per evaluable hour** as the primary training constraint. This is a development comparison point, not a claim of clinical acceptability. Prespecify 0.5/h as a secondary point; do not choose between them after test inspection. PN00 has little background time, so both targets can have coarse, unstable inner feasibility.

EEG threshold candidates: `{0.05,0.10,…,0.95,1.01}`, with `score >= threshold`. The final candidate represents no alarms. Among candidates satisfying pooled inner FAR≤1/h, maximize event sensitivity; break ties by lower FAR, shorter median delay among detected events, then higher threshold. Treat no detections as infinite delay. Save the entire inner selection table. The no-alarm candidate makes feasibility explicit but does not constitute detector success.

At the retained EEG threshold, an alpha>0 is eligible only if inner event sensitivity is at least EEG's and inner FAR is no greater, with a strict improvement in at least one. Require no more than 5 s deterioration in median delay among **events detected by both**. Rank eligible alphas by greater sensitivity, lower FAR, lower paired delay, then smaller alpha; otherwise choose zero. This conservative rule tests incremental contribution directly and limits threshold-related explanations. With very sparse inner events, alpha may be zero in most folds; that is informative.

This rule is a reviewer proposal to freeze after engineering validation. It is not established optimal, and training-side empirical FAR is not a confidence bound on future FAR. Keep threshold and alpha choices separate from statistical claims. An alternative jointly tuned blend is a secondary comparison, with its additional freedom disclosed.

| Alternative | Assessment |
|---|---|
| Fixed threshold 0.5 | Useful smoke/sanity check; not comparable at a clinically meaningful operating point when scores are uncalibrated |
| Maximize event F1 | Mixes seizure prevalence and alarm tradeoff; may favor selected seizure-rich recordings; secondary only |
| Maximize sensitivity subject to FAR | Recommended training objective; directly interpretable, but coarse in short recordings |
| Minimize FAR at a required sensitivity | Also defensible; with one or two inner seizures, sensitivity constraints jump between 0/50/100%; use only a frozen alternative |
| Threshold chosen on held-out EDF | Invalid, including choosing from its ROC curve after seeing outcomes |
| Many grids for alpha, smoothing, refractory, features and SQI | Not supportable; freeze most choices globally before nested evaluation |

Primary smoothing, merge gap, quality construction, model capacity and feature set are fixed, not per-fold searched. Optional supervised feature selection must be inside every inner fit and would require a new registered comparison. Probability calibration is postponed unless enough independent calibration parents exist; if later added, it must not fit on the same OOF labels used to present its unbiased evaluation.

### 8.4 Chronological deployment-oriented validation

Run a separately named forward-chaining analysis: train on earlier recordings, personalize only from earlier labeled events/background, then replay later recordings in acquisition order. Warm-up/label availability are part of the protocol. Do not use a later seizure to train the detector tested on an earlier EDF. If dates/session order cannot be established, do not make a prospective-training claim. Small eligible event counts may make this a feasibility analysis only.

## 9. Causal versus offline protocol

| Operation | Track A — OFFLINE | Track B — CAUSAL deployment-oriented |
|---|---|---|
| Filtering | Zero-phase allowed, explicitly labeled | Stateful one-pass filter; document group delay/transients; anti-alias causally if resampling |
| QRS | Batch detection allowed; future support disclosed | Adaptive thresholds from past; local peak confirmation occurs only when sufficient samples arrive |
| Beat timestamp | Physiological peak time and computational availability stored separately | RR becomes available no earlier than confirmation of its second beat |
| Normalization | Training-fitted model transforms; retrospective signal operations disclosed | Training-fitted transforms plus past-only baseline; no full-EDF statistics |
| SQI | Batch diagnostics allowed only in offline report | Uses received signal only; no future-quality lead selection |
| Windows | Explicit support intervals | Trailing; no post-t samples; reset per record, handle missing data |
| Smoothing | Retrospective smoothing may be compared separately | No centered score smoothing; proposed primary has no extra smoothing |
| Alarm duration/merge | May be retrospective for segmentation, never labeled online | State transitions using past/current decisions; no backdated confirmations |
| Latency | Processing horizon disclosed; not a deployment claim | Actual alarm availability minus reference onset, including buffering and confirmation; report measured compute overhead separately |
| Model training time | LORO can include chronologically later recordings | Forward-chaining required for future-record deployment claims |

A critical distinction: applying `filtfilt` to a window that ends at t can be recomputed when t arrives; it is not automatically reading the future beyond t. Nevertheless the present algorithm revises within-window peak estimates and is not the defined streaming Track B. Conversely, replacing `filtfilt` alone does not make normalization, QRS, smoothing or alarm timestamps causal.

Required replay tests: process a signal prefix, append a dramatically changed future, and verify already-emitted features/alarms remain unchanged; vary chunk size and confirm identical states/results within numerical tolerance; ensure previous beats are not retrospectively replaced without advancing availability time. A 60-s trailing context does not automatically impose 60-s seizure detection delay after startup; the update cadence, beat availability and needed response determine delay.

## 10. Event-level metrics and exact alarm semantics

### 10.1 Proposed primary alarm state machine

* Decision grid every 5 s. EEG first becomes available after its 10-s context; fusion initially falls back to EEG until ECG and baseline are valid. Include the startup interval in coverage accounting and report any seizure occurring before availability.
* At the first above-threshold decision while inactive, issue an alarm **at that decision's actual availability time**, and enter active state. No required two-window onset persistence; it would structurally disadvantage the five-second event.
* While active, above-threshold decisions keep the alarm active. Two consecutive below-threshold decisions close it at the second decision time. A single below-threshold gap is bridged. A missing/unavailable EEG decision is not normal background; record a coverage gap and reset according to the frozen missing-data policy.
* No additional primary refractory period. A 30-s refractory and a two-decision confirmation rule are secondary sensitivity variants, each reported with missed short events and latency costs. Never sweep merging/refractory and keep the best outcome.
* No bridging across EDF boundaries, excluded uncertain intervals, or acquisition gaps. Do not manufacture additional declarations from a long active alarm; report its duration and time-in-warning so a stuck alarm cannot look successful by having few declarations.

These rules are deliberately simple and are not claimed to be a clinical alarm policy. Record an alarm identifier, declaration/close times, score and modality/quality state. When a known acquisition gap interrupts state, record a distinct termination reason.

### 10.2 Reference matching

Use canonical reference intervals `[onset,offset)`. A seizure is detected under the **strict primary declaration metric** if a new alarm is declared inside that interval. Each declaration matches at most one reference event and each event earns at most one detection; use chronological matching. Repeated declarations inside an already detected event are logged as duplicate alarms, not additional sensitivity successes. Include their number in total alarm burden.

A pre-onset alarm that remains active through onset does not become a new timely detection retrospectively. It is a false early declaration under the strict metric, and the event may be missed. Report an additional active-state-overlap sensitivity so this conservative convention is transparent. A continuous alarm spanning several seizures cannot receive several primary true-positive declarations.

The strict primary metric measures alarm timing within the documented event, not every possible clinical warning use case. Prespecify a secondary late-detection metric with a +30-s post-offset tolerance; do not let it replace the strict result. Report pre-onset declarations separately as early warnings, without interpreting them as validated forecasting.

### 10.3 Required quantities

| Metric | Exact definition / reporting rule |
|---|---|
| Event sensitivity | Number of reference seizures with a matched declaration / all eligible reference seizures; report integer numerator/denominator per patient |
| False alarms/hour, primary | Unmatched declarations outside reference seizure intervals / evaluable recording hours; evaluable time includes ictal time, all ordinary peri-ictal time and ECG-outage fallback time |
| Nonictal FAR, companion | Same outside-event false declarations / evaluable nonictal hours; label denominator distinctly |
| Duplicate burden | Additional declarations within an already matched reference event, reported separately and in total declarations/hour |
| Event precision | Matched declarations / (matched + false + duplicate declarations), with this convention explicitly stated |
| Detection delay | First matched declaration availability minus the specified reference onset; never window-center/start backdating |
| Delay summaries | Median, IQR, range and optional mean for detected events; state number missed; no infinity hidden by dropping misses |
| Detected by 10/20/30 s | Number detected within both the event and the stated delay / **all** eligible seizures; misses remain in denominator |
| Warning burden | Total active-alarm time / evaluable time; longest alarm; total declarations and repeats |
| Coverage | Raw hours, startup hours, EEG-unavailable hours, annotation-uncertain hours, ECG-fallback hours; classify causes separately |
| Average precision | Continuous full-evaluation anchor labels at natural prevalence, with uncertainty exclusions disclosed; primary window summary |
| ROC-AUC | Secondary window diagnostic, not evidence of acceptable FAR |
| Window sensitivity/specificity | Supporting only at the frozen training-selected operating point; no random-window confidence interval |

Do not compute time exposure as `usable training row count × stride`. Derive it from EDF timeline intervals and the exact exclusion/coverage policy. The primary continuous comparison must use the same timeline for EEG and fusion. Give intention-to-monitor seizure sensitivity as well: events missed during algorithm startup/EEG unavailability count as misses unless their ground truth is itself unknowable. A conditional available-signal metric is secondary.

SzCORE offers standardized scoring, including a documented version with pre/post tolerances and event merging (R23). Use a pinned implementation as a **secondary compatibility score** and disclose its exact parameters. Its permissive overlap convention is not identical to the strict declaration metric here, and must not be used to justify negative causal latency. This study needs its own minimal deterministic fixtures even if it imports a scoring library.

## 11. Class imbalance, confounding and controls

Overlapping windows are repeated observations of a small number of seizures and recording states. The effective information is closer to five patients, at most 23 parents and 28 events before adjudication, with further dependence within patients. There is no single defensible “effective n” equal to a simple corrected window count.

Use class-weighted LR/RF as the primary simple approach. If training cost requires negative subsampling, select negatives within **training parents only**, stratify temporally across each parent, freeze seed and row IDs, and use the same rows across modalities. Evaluate every available held-out negative decision. Do not SMOTE neighboring windows or balance test sets. A balanced training bootstrap is a fitting mechanism, not uncertainty inference. Beware double compensation from aggressive subsampling plus class weights and do not interpret the resulting RF vote fraction as prevalence-calibrated probability.

| Threat | What can be checked | Limit of inference |
|---|---|---|
| Recording identity/state drift | Per-EDF score/feature distributions, timestamp-free X, held-out parents, chronological analysis | Shared admission and medication state can persist across EDFs |
| Medication, sleep, posture and arousal | Extract reliable metadata if available; report missing fields; look at background transitions | Current files do not establish medication-adjusted causal effects |
| ECG capturing movement or EEG contamination | Beat-level QC; waveform review; HR-only versus morphology/quality-only controls; shared-artifact strata | Without synchronized video/expert labels, mechanism cannot be proved |
| Cardiac contamination of EEG | Check QRS-locked EEG patterns in development samples; common-reference sensitivity | Correlation does not prove source or artifact direction |
| Seizure proximity | Continuous inference including the training exclusion zone | Better AP on easy distant negatives is not better alarms |
| Long-seizure dominance | Event sensitivity plus per-event training-weight sensitivity | Event weighting cannot create extra independent seizures |
| Feature count/morphology overfitting | Small frozen feature set and controlled model; modest secondary model | 65 features are already large relative to per-patient events |
| Patient-dependent benefits | Per-patient paired deltas, alpha and HR/SQI summaries | Five patients cannot support causal moderator claims |

Minimum negative controls: alpha=0; entire ECG missing (must reproduce EEG); ECG-quality-only auxiliary model; a label-blind training-side temporal shift of ECG for an exploratory alignment control. Shifting must preserve parent boundaries and avoid wraparound/future samples; discard or gate uncovered startup rather than using `np.roll` across time. This perturbation changes the task and is a falsification aid, not a source of independent significance. Do not select a shift because it destroys or creates the desired result.

## 12. Statistical analysis and scientific claims

### 12.1 Estimands

Primary comparison: paired **quality-aware adaptive fusion minus credible EEG-only** at the frozen training-side operating rule. Report the vector `(event sensitivity difference, FAR difference, latency difference, coverage difference)` rather than a single favorable metric. Per-patient differences come first; patient-macro sensitivity/FAR describe a typical patient, while pooled counts/time describe the observed event/hour-weighted cohort. Label both—PN14 contributes much more time than PN00 and PN10 more events than other patients.

The primary algorithm contrast and operating point must be selected before Phase 2. All early/fixed/adaptive/secondary variants remain named comparisons, not candidates from which to pick a new primary winner after testing.

### 12.2 Uncertainty

* For the pilot, show raw per-patient/per-parent/event outcomes and paired differences. Patient-cluster bootstrap intervals can be displayed descriptively, but five clusters yield unstable, discrete intervals. Do not claim precise population superiority.
* For an eligible expansion cohort, use a paired patient bootstrap (e.g. 10000 draws, fixed seed) retaining both algorithms' entire patient histories. Compute macro differences and separately ratio-of-sums pooled FAR differences. Keep the two estimands distinct.
* A hierarchical sensitivity bootstrap can resample patients, then parent EDFs within patient, retaining all seizures in each parent together. Avoid naïve seizure resampling that separates the three events in one EDF. For within-patient uncertainty, parent bootstrap is conditional and very weak at 3–6 parents.
* Seizure bootstrap may describe sensitivity/delay conditional on the observed recordings, but is not a robust cross-patient CI. Window bootstrap, ordinary independent-window binomial intervals and standard DeLong tests on correlated OOF windows are inappropriate for the principal claim.
* Conditional bootstrap of saved OOF predictions does not include refitting, tuning or earlier pilot design selection. State that limitation. Seed variability is reported separately, not multiplied into sample size or pooled as independent trials.
* Latency comparison uses events detected by both methods and reports the changed detection set alongside it. A detector that misses difficult seizures can have a deceptively short conditional median delay.
* With zero false alarms, report count and exposure; do not claim zero future risk. A simple Poisson upper bound is only a clearly labeled conditional illustration, because alarms cluster by recording/state.

Exploratory association of benefit with ictal HR change, baseline HR, quality, duration and phenotype should be plotted at patient/record/event levels with cluster identity visible. Do not fit a many-covariate regression on five patients or interpret a window-level p-value as a moderator effect. Do not derive a responder subgroup from held-out outcomes and then present that subgroup's performance as prospectively selected.

### 12.3 Meaningful outcomes

**Positive evidence:** predeclared paired improvement on previously uninspected evaluation data, with lower alarm burden at preserved event sensitivity and no material delay/coverage penalty, or greater sensitivity within the same alarm constraint. As a provisional engineering target, a ≥20% FAR reduction with no additional missed pilot events and ≤5-s paired median delay penalty is worth further study; it is not a powered clinical margin or a success rule to optimize retrospectively. Report absolute counts so a “20% reduction” from five to four alarms is not oversold. Confirmation requires uncertainty and replication, not merely meeting this point estimate.

**Meaningful negative evidence:** alpha frequently resolves to zero, nonzero fusion is unstable, or paired intervals allow no useful gain under the evaluated conditions; explain heterogeneity without retroactively changing the endpoint. An imprecise null does not prove equivalence or that ECG can never help.

**Invalidation:** unresolved misaligned labels, hidden parent/session leakage, test-selected settings, mislabeled offline latency, discarded difficult ECG periods changing exposure, untraceable annotation correction, or selection of the winning seed/metric after evaluation. Stop public performance claims until corrected; preserve invalidated artifacts with their provenance and status.

## 13. Minimum experiment matrix

Run only after the blocking corrections and evaluator tests. No full grid or new training was run for this report.

| ID | Experiment | Purpose | Tuning and reporting boundary |
|---|---|---|---|
| E0 | L2 logistic EEG / ECG / early fusion | Transparent baseline ladder | Same folds/rows; train-only scaling; fixed capacity |
| E1 | Controlled RF EEG-only | Primary EEG comparator | Inner threshold, frozen features/capacity |
| E2 | Compact ECG-only LR (RF secondary) | Quantify cardiac information and coverage | Same continuous exposure; separate operating threshold |
| E3 | RF early EEG+ECG | Conventional fusion comparator | Same EEG features, rows and RF capacity; separate threshold |
| E4 | Fixed late alpha=0.5 | Test arbitrary equal weighting | Clearly fixed; no rescue after seeing results |
| E5 | Adaptive late fusion | Isolate training-only alpha | Coarse grid, retain EEG threshold; alpha=0 allowed |
| E6 | Quality-aware adaptive late fusion | **Primary incremental comparison vs E1** | Frozen SQI/lead policy, same alpha rule and EEG threshold |
| A1 | E6 with all ECG missing / alpha=0 | Fallback equivalence | Must reproduce E1 scores/alarms from same state |
| A2 | Absolute versus relative HR; HR-only versus HR+RMSSD/CV | Physiological contribution | Small registered feature ablations, not best-of catalogue |
| A3 | Fixed lead versus quality-selected lead; q disabled | Quality/lead contribution | Same time exposure and all events |
| A4 | 30/60/120-s ECG and no-exclusion training variants | Timescale/proximity sensitivity | One axis at a time on pilot; report all planned values |
| A5 | Quality-only and temporal-misalignment controls | Artifact/alignment falsification | No test-selected shifts; label exploratory |
| A6 | PN00-3 corrected-end / PN10 alternative-boundary sensitivity | Annotation robustness | Separate cohorts, independent provenance; never choose by metric |
| E7 | EEG candidate + ECG soft confirmation | Secondary architecture hypothesis | Candidate OOF training; no-delay and fixed-15-s-delay tracks distinct |
| E8 | Stronger controlled EEG baseline plus matching E6 | Avoid weak-baseline argument | One prespecified alternative family; no leaderboard search |
| C1 | Causal raw-signal replay, E1 and E6 | Availability-time validity | Compare within CAUSAL track; report OFFLINE gap separately |
| C2 | Chronological earlier-to-later E1/E6 | Future-record feasibility | Only where chronology/support are defensible |
| X1 | Patient-LOO EEG / E6 with common montage | Secondary transfer | Inner patient grouping; separate from patient-specific primary |

Use a single primary stochastic seed initially, proposed `20260922`. A stability check on the **same fixed folds** uses two additional prespecified seeds `17` and `101` for E1/E3/E6 only. This perturbs model randomness without also changing fold assignment. Report all three; never replace the main seed with the best. Deterministic LR needs no gratuitous seed campaign.

## 14. Expansion plan: five patients to twelve is not twelve independent confirmations

Verified `subject_info.csv` gives 12 patients with ≥2 seizures, totaling 45 events; excluded single-seizure patients are PN07 and PN11. The five development patients already account for 28 events. Therefore the untouched extension is **seven patients / 17 events**, not 12 fresh patients / 45 fresh events. PN00-3 quarantine reduces the combined provisional count to 44, while other adjudications may change eligibility further.

The official `RECORDS` manifest exposes a critical calibration limitation before downloading anything:

| New patient | Official seizures | Parent EDFs | Consequence for proposed nested patient-specific evaluation |
|---|---:|---:|---|
| PN01 | 2 | **1** | No independent-record patient-specific test; cross-patient/external model only |
| PN03 | 2 | 2 | Outer LORO with externally frozen settings only; no inner positive calibration |
| PN05 | 3 | 3 | Nested scheme mathematically feasible, very sparse |
| PN09 | 3 | 3 | Same |
| PN13 | 3 | 3 | Same |
| PN16 | 2 | 2 | External settings fallback, separate stratum |
| PN17 | 2 | 2 | External settings fallback, separate stratum |

Only **three new patients / nine events** presently satisfy the ≥3-parent nested criterion, before signal/annotation QA. Do not split PN01's single EDF into seizure children to manufacture eligibility. A 12-patient patient-specific alpha claim is not supportable using the same nested protocol for everyone.

Recommended expansion design:

1. Freeze the complete algorithmic recipe using the five-patient development cohort. All five remain development data, including patients whose performance has not yet been run if they influence design decisions.
2. Apply the nested adaptive recipe unchanged to PN05/09/13, reporting this small confirmation stratum separately. Predeclared personalization on outer-training recordings is allowed; inspection-driven methodological changes are not.
3. For PN03/16/17, use a **separately frozen sparse-calibration recipe**: patient-specific base fitting on the other EDF, with operating/alpha settings learned exclusively from pilot cross-validation or a frozen external model. Report this as a different estimand, not pooled proof of the main adaptive recipe. If that recipe has not been validated on pilot simulations of the sparse setting, omit it and state the eligibility limitation.
4. PN01 can enter a secondary cross-patient test, using no labeled training data from its sole EDF. If all seven new patients form an untouched cross-patient test, train/tune on pilot patients only and keep that analysis separate from within-patient adaptation.
5. Report development, eligible confirmation, sparse-fallback and cross-patient results separately. A combined 12-patient descriptive table is allowed with stratum labels; it is not an independent confirmatory estimate.

Before Phase 2 freeze: raw-data/annotation versions and correction policies; parent/session definitions; feature schema and channel/reference policy; filters, QRS and beat confirmation; context/stride/baseline startup; SQI and lead switching; model families/capacity/weights; inner split and ineligibility rules; threshold and alpha objective/grid; alarm state machine; metrics/tolerances/denominators; primary contrast; missing-data policy; seeds; bootstrap estimands; analysis of ambiguous and short events; acceptance/negative-result language; and allowed sensitivity analyses.

No new model complexity is justified automatically at twelve patients. Postpone cross-patient transfer, candidate confirmation, calibrated uncertainty and frozen pretrained embeddings until the basic pipeline passes its audit. Even then the small new eligible stratum mainly supports feasibility, not broad population claims. An external dataset or prospectively acquired independent recordings would be required for stronger generalization evidence; do not expand scope to download/train on them without a separate task.

## 15. Reproducibility and anti-shopping design

The requested future structure `src/`, `tests/`, `configs/`, `results/`, `docs/` is appropriate. Avoid proliferating documentation before one authoritative protocol/config exists. Proposed implementation files below are future work; this review creates none of them.

Each run should have a content-derived experiment ID covering code revision or source-tree hash, environment lock, raw manifest hash, channel map hash, event-table hash, feature schema, parent folds, preprocessing track, model/threshold/SQI/alpha rules and seeds. Save the complete resolved configuration, train/validation/test IDs, sampled training-row IDs, inner selection results and outer predictions. A cache with a different dependency hash is stale even if its row count matches.

Results are append-only and include all planned seeds and failures. Re-running the same identity must verify equivalence or write a separately identified environment variant, not overwrite. Record when held-out outputs were first inspected. The pilot is openly exploratory; freeze a new protocol version before expansion and do not retroactively label existing experiments preregistered.

For each outer fold log: patient, test EDF/session, eligible training parents/events, seed, chosen alpha, all chosen thresholds, q coverage, selected ECG lead, EEG/ECG/early/fusion counts and exposure, delta versus EEG, event delays, reference onset type, annotation flags, baseline HR and ictal HR response (post-evaluation only), phenotype source and missingness, and recording characteristics. Never copy held-out physiology summaries into training configuration.

Raw EDFs stay outside Git. Before publication initialize/verify version control, inspect the actual tracked-file list and lock dependency versions. The existing `.gitignore` alone is not proof that raw data were never tracked. Do not publish source annotations or derivatives without retaining dataset attribution and license information.

## 16. Ordered implementation roadmap

Each stage is gated on its predecessor. Paths are proposed relative to the new project; they are not files created by this report.

| Stage | Goal and inputs | Proposed code/files | Required tests | Expected output | Go/no-go criterion |
|---|---|---|---|---|---|
| 0. Resolve source truth | Raw manifest, EDF headers, both event tables and conflicting documents | Unify annotation builder in `src/multimodal_seizure/annotations.py`; version `metadata/events.csv`; reconcile protocol/audit | PN14 clock discrepancy; PN00 raw/canonical provenance and quarantine; PN06 typo; PN12 inherited file/start; PN10 alternate end and onset types; midnight rollover; invalid clocks; exact bounds | One canonical event inventory and explicit eligibility flags | No silent ambiguity or three-hour shifts; all counts derived from approved inventory |
| 1. Unify readers/mapping | Official declarations + every EDF header | Consolidate `data.py` / `siena.py`; version mapping with units/reference/index/hash | All recordings, 1→0 indexing, unknown modality, missing/duplicate electrodes, channel-5 alias, no extra EKG, sample-unit conversion, exact-length reads | Stable EEG/ECG schema and per-file validation | Semantic identities agree; no out-of-range/truncated reads silently accepted |
| 2. Correct beats and SQI | Small fixed synthetic/real development panel | Revise `ecg.py`; add `quality.py`; past-only beat table with IDs/times/availability | Missed/double beats, inversion/noise/flatline/clipping, RR gap adjacency, known HR/slope, RMSSD formula, pNN50 units, lead switch, no future templates | Validated compact ECG representation + interpretable q | Beat accuracy/quality errors characterized; no fabricated adjacency; poor ECG produces fallback |
| 3. Freeze feature/window contract | Approved events, mapping, representation decisions | Revise `eeg.py`, `features.py`, `windows.py`; `configs/pilot_v1.json` | Half-open bounds, 5-s event, startup, no-seizure records, exclusion only for training, exact contexts, band integrals, entropy, analytic Hjorth, flat channels/correlation | Hashed feature contract and complete decision timeline | Same timeline across models; no annotation-derived inference feature |
| 4. Implement evaluator first | Handcrafted timelines and reference fixtures | `alarms.py`, `metrics.py`, evaluator fixtures in `tests/` | Pre-onset active alarm; late and duplicate alarms; two events/one alarm; gaps/EDF reset; denominator arithmetic; available-time latency; all-missed/no-alarm cases | Trusted deterministic event accounting | All hand-calculated fixtures agree before model fitting |
| 5. Implement nested runner | Approved folds/features + E0/E1/E2 | `validation.py`, `models.py`, `run_experiment.py`; immutable fold/config manifests | Parent/sample-support disjointness; test-label perturbation; train-only scaling/imputation; missing class; seed/cache reproducibility | Credible EEG and ECG baselines with complete OOF streams | No leakage; stable deterministic execution; inference failure surfaced rather than excluded |
| 6. Add minimal fusion | Saved inner OOF streams, validated SQI | `fusion.py`; E3–E6 configs | alpha bounds/zero/one; q bounds; exact EEG outage fallback including threshold/state; deterministic tie-breaking; quality transition | Paired main comparisons and prescribed ablations | No extra test fitting; negative/zero-alpha result retained |
| 7. Causal replay | Validated algorithms + small signal fixtures, then selected recordings | `streaming.py` and separate CAUSAL config | Future perturbation, prefix/chunk invariance, beat availability, filter delay, record reset, actual alarm timestamp | End-to-end causal replay and separate OFFLINE gap | No real-time claim unless all components and chronology are appropriately scoped |
| 8. Pilot audit and freeze | Full development outputs, source hashes and selection logs | Freeze versioned protocol/config; compact machine-readable result manifest | Recompute metrics from predictions; account for all folds/seeds; no hidden time deletion; inspect per-parent failures | Explicit main hypothesis, frozen recipe and expansion eligibility | Independent review can reconstruct every count; stop if labels/quality/evaluation remain unresolved |
| 9. Untouched expansion | Only after frozen stage 8 | Manifest-driven new cohort run using existing code | Same hashes/config; new-source QA without outcome tuning; sparse-setting eligibility | Separate eligible confirmation and secondary strata | No methodological changes after held-out inspection; amendments create a new exploratory version |
| 10. Portfolio release | Audited code/config/results and honest limitations | Public runner, pinned environment, dataset acquisition instructions in existing project documentation | Clean synthetic CI; selected data-dependent checks; verify no raw tracked files | Reproducible scientific project, including negative findings | Claims match scope; no implied clinical validation or first-ever Siena fusion claim |

Stage 2 real-data beat labeling and annotation adjudication may require clinical expertise the current repository does not provide. Document that dependency; do not replace expert uncertainty with algorithmic confidence. Stages 5 onward involve training and should be explicitly scheduled as implementation work after this planning task.

## 17. Testing strategy in detail

Use small fixtures and targeted tests; reserve data integration checks for configured local raw data. Do not make CI require gigabytes of downloads.

| Area | Required regression/property |
|---|---|
| EDF mapping | Permuted labels cannot change official index mapping; official order checked against each EDF; two physical readers agree on a short segment |
| Annotation parsing | Both PN10 time candidates retained; clinical/electric onset disambiguated; PN14 uses documented header policy; PN00 correction cannot be accepted without provenance |
| Time conversion | 23:59:59→00:00:01; malformed clocks; real same-day negative offsets must not automatically become next-day events; inherited PN12 context |
| Window labeling | Exact onset/offset, shortest event, 1/5-s grid, startup and terminal samples; no artificial adjacency after ignored rows |
| Leakage | Intersections of patient/parent/raw-support sets; mutation of outer labels leaves all selection unchanged; no train-fitted transform sees outer X |
| Filtering/QRS | Prefix invariance; chunk-size invariance; bounded availability delay; inverted/irregular beats; no future interpolation |
| RR/HRV | Preserve absolute timestamps and original neighbors; calculate RMSSD only on valid adjacent pairs; distinguish fraction versus percentage pNN50; finite/empty behavior |
| EEG | Known sinusoid powers and integrated total; white noise versus tone entropy; analytic Hjorth scaling; identical/independent/flat-channel correlation; units |
| Fusion/SQI | alpha=0 equality; missing ECG equality; one clean/one noisy lead; both conflicting leads; quality changes do not change evaluated exposure |
| Thresholds | Deterministic inner-only objective/ties; all-negative/no-positive validation; no-feasible-positive-detector state retained |
| Alarms | First declaration not backdated; two-low reset; no cross-EDF merge; repeats; pre/late alarms; long stuck warning; exact FAR denominator |
| Reproducibility | Same config+seed gives same predictions; changed raw/event/feature hash prevents cache reuse; no result overwrite |

The 18 current tests are useful engineering smoke checks; they do not cover this matrix. A small number of meaningful boundary and invariance tests is preferable to many tests that merely duplicate implementation formulas.

## 18. Risks, prohibited actions and final answers

### 18.1 What must not be done

Do not modify raw Siena annotations or EDFs; erase old results/audits; silently reconcile conflicting event tables; include PN00-3 because its removal helps performance; use random window splits; split multi-seizure EDFs to make nested tuning feasible; normalize over complete held-out records; interpolate RR using future beats in the causal track; choose SQI, alpha, threshold, seed, merge gap or reporting metric from held-out outcomes; remove poor ECG time only from the fusion denominator; describe arbitrary RF scores as calibrated confidence; interpret patient-level phenotype metadata as exact per-event semiology; turn predictive association into autonomic causation; describe a retrospective recording-held-out analysis as real-time clinical validation; or add deep learning for portfolio appearance.

Do not assume 45 seizures are 45 independent observations, twelve patients are twelve fresh confirmations, or repeated seeds increase clinical sample size. Do not claim clinical equivalence from a nonsignificant result. Do not equate a good cardiac-response subgroup chosen retrospectively with a deployable selection rule. Do not compare headline metrics across papers with different event windows, time exposure or sensor montages as if they share a benchmark.

### 18.2 Answers to the twelve pre-implementation questions

| Question | Concrete answer |
|---|---|
| 1. Worth continuing? | Yes, as an incremental-information/robustness study after annotation/evaluator corrections; no promise of benefit |
| 2. Most defensible ECG role? | Patient-specific autonomic auxiliary evidence and confidence adjustment; optional candidate confirmation, never universal veto |
| 3. Strongest features? | Reliable RR timing, median HR, actual-time slope, change from past baseline, limited cleaned-RR variability with coverage |
| 4. Is 60 s appropriate? | Reasonable primary candidate; not a validated optimum or universal HRV duration |
| 5. Multiple timescales? | A small recent-versus-earlier contrast is justified; 30/120 s as predeclared ablations, not a large feature bank |
| 6. Individual baseline? | Yes, past-only, quality-qualified and explicit startup; no whole-record z-score |
| 7. Primary fusion? | Quality-aware adaptive late score fusion with a fixed EEG operating threshold and alpha=0 fallback; early/fixed/adaptive comparators required |
| 8. Keep simple? | Features, base model families, SQI formula, one alpha, alarm rules; no learned gate or deep embedding from scratch |
| 9. Postpone? | Cross-patient transfer, delayed confirmation, calibration/uncertainty and pretrained embeddings; twelve patients still do not justify unrestricted complexity |
| 10. Meaningful positive? | Paired improvement on frozen unseen evaluation with sensitivity, FAR, delay and coverage jointly acceptable; counts/uncertainty transparent |
| 11. Meaningful negative? | Stable zero-alpha/no-benefit or harmful ECG in defined conditions; distinguish lack of information from lack of statistical precision |
| 12. What invalidates it? | Wrong labels/timebase, parent/future leakage, test shopping, selective time/event exclusion, or claims exceeding actual validation |

**Final recommendation:** first reconcile annotations and channel paths; then build a continuous event evaluator and causal beat/SQI subsystem; then establish compact credible EEG/ECG baselines; finally test whether a training-only, quality-gated ECG contribution improves the same detector under the same alarm policy. Freeze that recipe before opening expansion outcomes. A technically strong portfolio result is an inspectable answer about when ECG helps or fails, with all failure cases retained—not a manufactured fusion win.

## 19. Review verification boundary

Completed during this review: read both relevant projects; checked old evidence against the preserved source hashes; independently hashed all 31 Siena files; authenticated the checksum manifest against PhysioNet; checked all 23 raw headers and saved index mappings; recomputed window inventories and saved PN00 OOF metrics; ran the existing 18-test suite; performed small read-only RR-gap and interval-boundary probes; inspected primary literature and current 2026 evidence. All proposed corrections and experiments remain **unimplemented**.

Not established: expert validity of the PN00 replacement end, clinical reannotation of PN10 ambiguities, beat-level accuracy throughout every recording, exact electrode derivations beyond available metadata, prospective session independence, calibration, event-level performance of existing PN00 outputs, or a positive ECG fusion effect. No model training, dataset regeneration, benchmark campaign, original-data change, commit or publication was performed.
