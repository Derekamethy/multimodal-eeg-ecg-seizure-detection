# Final public-release repository audit

Public-release audit: **2026-09-24**. Scientific closeout: 2026-09-23. Scope: actual Git candidate content, release hygiene and read-only verification. No new scientific experiment was performed.

## Git state and audit boundary

The project was initialized as a real Git repository, finalized on branch `main`, committed, and pushed to the public GitHub repository `Derekamethy/multimodal-eeg-ecg-seizure-detection`. Release baseline commit: `5b1b3cdec968b7eb4b59cac21ff3869d130cbe54`.

Before any staging, native `git status --short`, `git status --ignored`, `git ls-files`, branch/remote/history checks and `git add --dry-run .` were saved outside the repository with a SHA-256 baseline of 337 protected files. The initial index was empty. Git identified **396 public candidate files** and **14,450 ignored files**; ignored counts enumerate individual paths, including the local environment and caches.

The earlier scientific closeout changed 10 existing files and added 18, without deleting or moving anything. This release pass modifies only README conclusion wording, the final summary Git-status reference and this audit, and initializes Git. These are filesystem changes; with no parent commit, Git presents all release files as additions.

## Immutable scientific state

| Evidence | Identity |
|---|---|
| Canonical metadata | `91b7c18e669fd92ef785c9692b15635c766db42d584bb1a24300dd34d64160b3` |
| Phase 2 EEG | `3ba65f49586b8c13e4e497ba0aa2e160bfed9d1cdd64434bb72a131f72a47b21` |
| Phase 3 ECG | `6813d261b922f82721d27bccdfa452135a56c4d47c3afe5f7bc72506a0c4737e` |
| Phase 4 fusion | `ae451eb4142626cd35daa0e4309a64ece2f56fd43b242c0a6b5b1e55cfb9ebea` |

The protected set comprises 306 frozen result/sidecar files, seven metadata files, 18 package source files and six Phase 2–4 runner/verifier scripts: **337 files**. All retain their pre-initialization SHA-256 values. Of these, 336 belong in the public index; `metadata/edf_audit.csv` is intentionally local and ignored, but remains protected against modification. The 13 historical preservation dependencies also retain their original hashes, checked by Phase 2 replay.

The scientific cohort remains **five patients, 22 eligible EDFs, 27 seizures and 59.91 h**, in a patient-specific retrospective design. EEG RF detects 13/27 with 69 false declarations; ECG RF 4/27 with seven; F1 11/27 with 89; F2 13/27 with 68; F3 11/27 with 95. F2 preserves 12 EEG detections, rescues one, loses one and still misses 13.

Current compact rhythm-based ECG features did not demonstrate reliable incremental value over the patient-specific EEG detector under strict whole-record continuous evaluation. No Phase 5, feature/threshold changes, new patients, result-driven redesign or production model retraining occurred.

## Real Git candidate classification

| Category | Files | Disposition |
|---|---:|---|
| Source code | 39 | Package, runners, verifiers and utilities retained |
| Tests | 12 | Existing tests retained |
| Canonical metadata | 4 | Authoritative tables and manifest |
| Frozen scientific evidence | 306 | All accepted configs, scores, features, tables, manifests and sidecars |
| Current documentation | 5 | README, status, protocol, final summary and this audit |
| Generated figures/manifest | 9 | Four PNG/SVG pairs and provenance manifest |
| CI/release support | 6 | Workflow, attributes, ignore rules, package metadata and requirements |
| Historical evidence/documents | 15 | Thirteen required preservation files and two superseded documents |
| Private/local material in candidate | 0 | Excluded by actual Git behavior |

All candidate paths were inventoried and classified outside the repository. Legacy audit, extraction and smoke entrypoints remain traceability utilities; README identifies the current verification path. Old PN00-only outputs are not current model results.

The 13 historical files required by the canonical manifest and frozen Phase 2 verifier remain at their original paths:

```text
seizure_events_audit.csv
official_channel_map.csv
pilot_window_index.csv
pilot_window_summary.csv
pn00_smoke_features.csv
pn00_smoke_benchmark.csv
ecg_signal_qc.csv
metadata/seizure_event_audit.csv
metadata/patient_summary.csv
results/benchmarks/PN00_full_fold_metrics.csv
results/benchmarks/PN00_full_oof_predictions.csv
results/benchmarks/PN00_full_pooled_metrics.csv
results/features/PN00_features.csv
```

`docs/DATA_AUDIT.md` and `docs/MULTIMODAL_EEG_ECG_RESEARCH_PLAN.md` explicitly mark their contents historical/superseded. The study protocol distinguishes historical plans from its final implemented addendum. No stale proposal is presented as current implementation or future authorization.

## Raw data, ignores and byte preservation

Actual-project `git check-ignore -v` checks exclude **32/32 raw files**, including representative EDFs from PN00, PN06, PN10, PN12 and PN14. This includes 31 verified source files and the checksum manifest. None was tracked before staging; none is in the candidate list.

Native Git checks also exclude `.venv`, Python/pytest caches, IDE metadata, OS junk, logs, and the local-only `edf_header_audit.csv` and `metadata/edf_audit.csv`. README, code, tests, canonical metadata, all frozen evidence, docs, figures, workflow and dependency files remain Git-visible. Four explicit negative ignore patterns correctly re-include the required historical result CSVs.

`.gitattributes` remains `* -text`. Native `git check-attr` reports `text: unset` for all 337 protected paths, including CSV, JSON, sidecars, manifests and source. No mass normalization, destructive reset or history rewrite occurred. Staging must preserve the protected working-tree hashes and the hashes of all 336 public protected blobs.

Nothing was deleted or moved in this pass. Project-cache deletion had been rejected by execution policy during the prior scientific closeout; caches remain local and ignored. No deletion retry or alternative bypass was attempted. A whole-directory archive bypassing Git exclusions is not the release candidate.

## Public-content and licensing audit

Conservative scans covered all Git-visible text for likely credentials, private absolute paths, device/session identifiers and internal conversation notes. No credible secret, private-path leakage or internal prompt/AI note was found. Broad path-pattern hits were reviewed: 1,287 were public URLs or numerical substrings, and two were regular-expression literals for sampling rate and content length. No frozen evidence required sanitization. This is a bounded content/pattern audit, not a guarantee against every possible hidden secret.

All public utilities resolve project-relative paths; none requires one developer's absolute home directory. The six utility path fixes from the earlier closeout remain unchanged, and no frozen producer was edited.

Source data must be obtained from [PhysioNet Siena v1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/), whose dataset files use [CC BY 4.0](https://physionet.org/content/siena-scalp-eeg/view-license/1.0.0/). Raw EDFs, annotations and the downloaded dataset license remain excluded from this repository. Dataset licensing does not license project software. The repository now uses the **MIT License** for project source code; Siena/PhysioNet data remain governed by their original dataset license and are not redistributed here.

## Size and GitHub suitability

The actual Git candidate set is approximately **334.24 MB (318.75 MiB)** before Git object compression. Frozen results dominate its size. Seven files exceed 10 MB; no candidate reaches the [GitHub 100 MiB hard file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Raw data is approximately 9.88 GB and excluded. No LFS was introduced and no reproducibility evidence was removed.

All seven large files are under `results/phase3_ecg/6813d261b922/`:

| Path within the frozen directory | Bytes |
|---|---:|
| `scores/PN14_C0_LR_continuous_oof.csv` | 22,611,215 |
| `scores/PN14_C1_RF_continuous_oof.csv` | 22,538,060 |
| `scores/PN10_C0_LR_continuous_oof.csv` | 18,886,897 |
| `scores/PN10_C1_RF_continuous_oof.csv` | 18,868,594 |
| `scores/PN06_C1_RF_continuous_oof.csv` | 15,041,410 |
| `scores/PN06_C0_LR_continuous_oof.csv` | 14,981,360 |
| `features/PN14__PN14-3.edf.csv` | 11,624,915 |

## Scientific verification before staging

All six requested commands were run successfully from the actual project root:

```powershell
.\.venv\Scripts\python.exe verify_siena_subset.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe verify_phase2_eeg.py results/phase2_eeg/3ba65f49586b
.\.venv\Scripts\python.exe verify_phase3_ecg.py results/phase3_ecg/6813d261b922
.\.venv\Scripts\python.exe verify_phase4_fusion.py results/phase4_fusion/ae451eb41426
.\.venv\Scripts\python.exe generate_final_figures.py --check-docs
```

| Check | Observed result |
|---|---|
| Raw source verification | 31/31 |
| Full tests | 74 passed in 43.03 s |
| Phase 2 | 44/44 model-folds, including historical preservation |
| Phase 3 | 44/44 model-folds and 27/27 response rows |
| Phase 4 | 88/88 model-folds, 22/22 inner selections, 27/27 events, 69/69 false fates, 66/66 fallback controls; 180 Phase 2/3 files unchanged |
| Displayed result tables | 2/2, including model counts, event denominator, event fates and alpha distribution |

Tests fit small synthetic fixtures; no production model was retrained or experiment regenerated. All four immutable identities pass the existing verification paths.

## Documentation and figures

README answers the research question, dataset boundary, implementation, leakage safeguards, negative finding, reproduction and limitations. README, summary, status, protocol and this audit agree on the frozen study. The displayed five-system result tables match frozen CSVs. The F3 18-second median has an explicit footnote: detected-event sets differ, so it does not establish faster detection.

Four static figures were visually reviewed. The figure manifest verifies **11 source CSV hashes, eight PNG/SVG hashes and the generator hash**. The files match the previously verified deterministic render; regeneration was unnecessary. The generator reads frozen CSVs without importing model code, raw-signal access or training.

PN06-1 correctly displays EEG, ECG and F2 scores, effective thresholds, the canonical seizure interval and the EEG declaration at +42 s that F2 suppresses. Titles, axes and legends are readable. Event counts are descriptive and imply no unsupported statistical significance. Figure 1 remains the README Mermaid diagram.

## CI and release simulation

`.github/workflows/tests.yml` uses read-only repository permissions, Python 3.11.9 and `requirements-frozen.txt`, with no raw-data download, secrets or local-machine path. The Python 3.11 series remains in its [security-support period](https://devguide.python.org/versions/); 3.11.9 is the frozen reproduction pin, not the latest security patch. Core dependencies and plotting dependencies were not changed in this pass.

The exact test command was extracted from the workflow and rerun:

```text
python -m pytest -q -p no:cacheprovider
  tests/test_alarms_metrics.py tests/test_baseline.py tests/test_fusion.py
  tests/test_cardiac.py tests/test_eeg.py tests/test_ecg.py
  tests/test_phase2_features.py
  tests/test_inventory.py::test_clock_parser_rejects_malformed_and_impossible_values
  tests/test_inventory.py::test_half_open_labels_and_uncertainty_override
  tests/test_inventory.py::test_event_identity_and_transitive_dependencies_invalidate_cache
  -k "not real_ and not timeline_uses_eeg_history"
```

Lines above form one shell command, as in the workflow's folded YAML scalar.

- Source-only copy without `data/`, `metadata/` or `results/`: **53 passed, 4 deselected in 23.75 s**.
- Complete 396-file Git candidate copy, without raw data or the local environment: **53 passed, 4 deselected in 23.78 s**.
- Candidate Python syntax: **51/51 files passed**. Package imports from the copy: **17/17 passed**.

Both test runs use the accepted local interpreter with bytecode writes disabled. This verifies clean content, not a fresh dependency installation. README links/layout and required source/dependency inclusion were checked in the release candidate. Final documentation-only edits are synchronized into the candidate before staging; the tested code is unchanged.

**Hosted GitHub Actions passed.** A fresh Ubuntu 24.04 runner installed the frozen Python 3.11.9 dependency set successfully and completed the data-independent suite with **53 passed, 4 deselected in 13.55 s**. This provides independent hosted Linux validation of installation and the CI-selected test subset; full real-data replay still requires the local Siena source subset.

## Release gate and remaining actions

**Verdict: PUBLIC RELEASE COMPLETE WITH DOCUMENTED LIMITATIONS.** The audited 396-file release was committed and pushed to `main`. GitHub Actions subsequently completed successfully in a fresh hosted Ubuntu environment. All previously verified scientific identities remain frozen; this documentation-only release update does not alter Phase 1–4 scientific artifacts.

Scientific limitations remain five patients, 27 seizures, seizure-bearing source recordings, patient-specific retrospective analysis, recording/ECG heterogeneity, no manual beat ground truth, no clinical/streaming validation and no external untouched cohort. The finding is not a universal claim that ECG lacks utility.

Release actions are complete: software licensing is MIT, the first commit is on `main`, the GitHub remote is connected, the release is public, and hosted CI passed **53 tests with 4 deselected**. No further pilot research is planned; future changes should be documentation/packaging maintenance unless a separately scoped study is created.
