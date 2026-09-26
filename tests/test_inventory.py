import hashlib
import json
from dataclasses import replace

import pandas as pd
import pytest

from multimodal_seizure.data import read_event_table
from multimodal_seizure.edf import read_header
from multimodal_seizure.inventory import (
    RAW_ROOT, TABLES, clock_seconds, parse_candidates, parse_events, sha256, validate_interval,
)
from multimodal_seizure.provenance import (
    artifact_spec, artifact_status, metadata_identity, require_current, stamp_artifact,
)
from multimodal_seizure.siena import build_channel_map, channel_rows
from multimodal_seizure import windows


def test_pn14_clock_discrepancy_and_header_alignment():
    event = read_event_table().set_index("event_id").loc["PN14-3"]
    assert event.registration_start_raw == "16.17.45"
    assert event.edf_start_time == "19.17.45"
    assert event.registration_discrepancy_s == 10800
    assert event.source_clock_onset_relative_s == 17540
    assert (event.onset_relative_s, event.offset_relative_s_canonical) == (6740, 6781)
    assert event.primary_eligible


def test_pn00_proposal_is_not_ground_truth_or_negative_background():
    event = read_event_table(include_quarantined=True).set_index("event_id").loc["PN00-3"]
    assert event.seizure_end_raw == "19.29.29"
    assert event.offset_relative_s_raw == 4425
    assert event.proposed_offset_s == 825
    assert not event.primary_eligible
    assert pd.isna(event.offset_relative_s_canonical)
    assert json.loads(event.uncertain_intervals_json) == [[0, 2509]]
    with pytest.raises(ValueError, match="Quarantined"):
        windows.build_window_index("PN00", "PN00-3.edf")
    assert "PN00-3.edf" not in {f["test_file"] for f in windows.build_patient_folds("PN00")}


def test_pn10_candidates_and_onset_kinds_survive():
    events = read_event_table().set_index("event_id")
    end = events.loc["PN10-2"]
    assert [c["relative_s"] for c in json.loads(end.offset_candidates_json)] == [7849, 7828]
    assert end.offset_relative_s_canonical == 7828
    assert json.loads(end.uncertain_intervals_json) == [[7828, 7849]]
    onset = events.loc["PN10-3"]
    assert (onset.clinical_onset_s, onset.electrographic_onset_s) == (7835, 7841)
    assert onset.onset_relative_s == 7835 and onset.onset_type == "clinical"
    assert json.loads(onset.uncertain_intervals_json) == [[7835, 7841]]
    clinical_only = events.loc["PN10-6"]
    assert clinical_only.onset_type == "clinical" and pd.isna(clinical_only.electrographic_onset_s)
    assert events.loc["PN10-7"].registration_start_raw == "1 6.49.25"
    inherited = events.loc["PN12-2"]
    assert inherited.registration_inherited and inherited.registration_start_effective_raw == "15.51.31"


def test_midnight_rollover_and_impossible_intervals():
    header = replace(read_header(RAW_ROOT / "PN00/PN00-1.edf"), start_time="23.59.50", n_records=100, record_duration_s=1)
    text = """Seizure n 1
File name: midnight.edf
Registration start time: 23.59.50
Registration end time: 00.01.30
Seizure start time: 00.00.05
Seizure end time: 00.00.15
"""
    row = parse_events("TEST", text, {"midnight.edf": header}, "synthetic", "hash")[0]
    assert (row["onset_relative_s"], row["offset_relative_s_canonical"]) == (15, 25)
    for changed in (text.replace("00.00.15", "00.00.01"), text.replace("00.00.15", "00.02.00")):
        with pytest.raises(ValueError, match="outside EDF"):
            parse_events("TEST", changed, {"midnight.edf": header}, "synthetic", "hash")
    for interval in ((-1, 5, 10), (5, 5, 10), (5, 11, 10), (float("nan"), 5, 10)):
        with pytest.raises(ValueError):
            validate_interval(*interval)


def test_clock_parser_rejects_malformed_and_impossible_values():
    assert clock_seconds("16:13.23") == 58403
    for value in ("24.00.00", "12.60.00", "12.00.60", "-1.00.00", "12.00", "x12.00.00", "12.00.00 trailing", "1 6.49.25"):
        with pytest.raises(ValueError):
            clock_seconds(value)
    for value in ("12.00.00 opure", "12.00.00 (UNKNOWN ONSET)", "25.00.00"):
        with pytest.raises(ValueError):
            parse_candidates(value)


def test_half_open_labels_and_uncertainty_override(monkeypatch):
    events = pd.DataFrame([dict(patient="P", canonical_file_name="f", edf_duration_s=80,
        primary_eligible=True, onset_relative_s=60, offset_relative_s_canonical=65,
        seizure_number=1, uncertain_intervals_json="[[70, 75]]")])
    monkeypatch.setattr(windows, "read_event_table", lambda **kwargs: events)
    frame = windows.build_window_index("P", "f", exclusion_s=0).set_index("anchor_s")
    assert frame.loc[60, "label"] == 1
    assert frame.loc[65, "label"] == 0
    assert frame.loc[70, "status"] == "annotation_uncertain"
    assert frame.loc[75, "label"] == 0


def test_channel_conversion_alias_and_extra_lead_exclusion():
    header = read_header(RAW_ROOT / "PN00/PN00-1.edf")
    source = RAW_ROOT / "PN00/Seizures-list-PN00.txt"
    rows = channel_rows(header, source)
    alias = next(r for r in rows if r["channel_number_1based"] == 5)
    assert (alias["edf_index_0based"], alias["official_name"], alias["edf_raw_label"], alias["canonical_name"]) == (4, "1", "EEG O1", "O1")
    assert alias["alias_note"]
    mapping = build_channel_map(header, source)
    assert mapping.ecg_indices == (32, 33)
    assert all(header.signals[i].label == "EKG EKG" for i in mapping.extra_named_ecg_indices)
    assert not set(mapping.extra_named_ecg_indices) & set(mapping.ecg_indices)


def test_channel_failures_include_later_recording_and_missing_noncommon(tmp_path):
    source = RAW_ROOT / "PN00/Seizures-list-PN00.txt"
    text = source.read_text()
    # A mismatch on the second recording must fail just as the first would.
    header = read_header(RAW_ROOT / "PN00/PN00-2.edf")
    bad = replace(header, signals=(replace(header.signals[0], label="EEG UNKNOWN"), *header.signals[1:]))
    with pytest.raises(ValueError, match="mismatch"):
        channel_rows(bad, source)
    duplicate = replace(header, signals=(header.signals[0], replace(header.signals[1], label="EEG Fp1"), *header.signals[2:]))
    path = tmp_path / "duplicate.txt"
    path.write_text(text.replace("Channel 2: F3", "Channel 2: Fp1"))
    with pytest.raises(ValueError, match="Duplicate canonical"):
        channel_rows(duplicate, path)
    for changed in (
        text.replace("Channel 1: Fp1", "Channel 1: UNKNOWN"),
        text + "\nChannel 1: Fp1\n",
        text.replace("Channel 2: F3", "Channel 2: Fp1"),
        text.replace("Channel 33: EKG 1", "Channel 333: EKG 1"),
        "\n".join(line for line in text.splitlines() if not line.startswith("Channel 9:")),
        text.replace("Channel 34: EKG 2", "Channel 34:"),
    ):
        path = tmp_path / "list.txt"
        path.write_text(changed)
        with pytest.raises(ValueError):
            channel_rows(header, path)


def test_truncated_edf_header_fails(tmp_path):
    source = RAW_ROOT / "PN00/PN00-1.edf"
    path = tmp_path / "truncated.edf"
    with source.open("rb") as f:
        path.write_bytes(f.read(270))
    with pytest.raises(ValueError, match="Short EDF signal header"):
        read_header(path)


def test_event_identity_and_transitive_dependencies_invalidate_cache(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for name in TABLES:
        (metadata / name).write_text("initial\n")

    def rebuild():
        payload = dict(tables={name: sha256(metadata / name) for name in TABLES}, sources={}, implementation={})
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        (metadata / "canonical_manifest.json").write_text(json.dumps(dict(**payload, metadata_identity=identity)))
        return identity

    old_identity = rebuild()
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("old result\n")
    with pytest.raises(ValueError, match="STALE artifact"):
        require_current(legacy, root=tmp_path)
    producer = tmp_path / "producer.py"
    producer.write_text("policy = 1\n")
    cache = tmp_path / "cache.csv"
    cache.write_text("anchor,label\n60,1\n")
    spec = artifact_spec("features", [producer], {"window": 60}, root=tmp_path)
    stamp_artifact(cache, spec)
    require_current(cache, root=tmp_path, expected_identity=spec["cache_identity"])
    result = tmp_path / "result.csv"
    result.write_text("result\n")
    stamp_artifact(result, artifact_spec("result", [cache], root=tmp_path))
    producer.write_text("policy = 2\n")
    assert artifact_status(result, old_identity, tmp_path) == "stale"
    producer.write_text("policy = 1\n")
    (metadata / TABLES[0]).write_text("changed event\n")
    with pytest.raises(ValueError, match="Canonical table changed"):
        metadata_identity(tmp_path)
    new_identity = rebuild()
    assert old_identity != new_identity
    assert artifact_spec("features", [producer], {"window": 60}, root=tmp_path)["cache_identity"] != spec["cache_identity"]
    with pytest.raises(ValueError, match="STALE artifact"):
        require_current(cache, root=tmp_path)
