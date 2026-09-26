"""Canonical Siena annotation and recording inventory. Raw evidence is never edited."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .edf import read_header
from .siena import channel_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data/raw/siena-scalp-eeg-1.0.0"
PILOT_PATIENTS = ("PN00", "PN06", "PN10", "PN12", "PN14")
POLICY = "siena-phase1-v1-header-clock-half-open-quarantine-PN00-3"
TABLES = ("canonical_events.csv", "canonical_recordings.csv", "canonical_channels.csv")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def header_sha256(path: Path) -> str:
    h = read_header(path)
    with path.open("rb") as f:
        return hashlib.sha256(f.read(h.header_bytes)).hexdigest()


def clock_seconds(raw: str) -> int:
    match = re.fullmatch(r"\s*(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})\s*", raw)
    if not match:
        raise ValueError(f"Malformed clock: {raw!r}")
    h, m, s = map(int, match.groups())
    if h > 23 or m > 59 or s > 59:
        raise ValueError(f"Impossible clock: {raw!r}")
    return h * 3600 + m * 60 + s


def relative_seconds(clock: int, start: int) -> int:
    return (clock - start) % 86400


def validate_interval(onset: float, offset: float, duration: float) -> None:
    if not 0 <= onset < offset <= duration:
        raise ValueError(f"Event outside EDF or invalid interval: [{onset}, {offset}) / {duration}")


def parse_candidates(raw: str) -> list[dict]:
    out = []
    for part in re.split(r";|\bopure\b", raw, flags=re.I):
        match = re.fullmatch(r"\s*(\d{1,2}[.:]\d{1,2}[.:]\d{1,2})\s*(?:\((CLINICAL|ELECTRIC|ELECTROGRAPHIC) ONSET\))?\s*", part, re.I)
        if not match:
            raise ValueError(f"Malformed event clock/candidate: {part!r}")
        kind = (match.group(2) or "unspecified").lower()
        if kind in {"electric", "electrographic"}:
            kind = "electrographic"
        out.append(dict(raw_clock=match.group(1), clock_s=clock_seconds(match.group(1)), kind=kind))
    return out


def parse_events(patient: str, text: str, headers: dict, source: str, source_hash: str) -> list[dict]:
    rows, registrations, seen = [], {}, set()
    for block in re.split(r"(?=Seizure\s+n\s+\d+)", text, flags=re.I):
        number = re.search(r"Seizure\s+n\s+(\d+)", block, re.I)
        if not number:
            continue
        number = int(number.group(1))
        if number in seen:
            raise ValueError(f"Duplicate event: {patient}-{number}")
        seen.add(number)

        def field(name: str, required: bool = True) -> str:
            match = re.search(r"^" + re.escape(name) + r":\s*([^\r\n]+)", block, re.M | re.I)
            if not match and required:
                raise ValueError(f"Missing {name} in {patient}-{number}")
            return match.group(1).strip() if match else ""

        raw_file = field("File name")
        filename = raw_file.replace("PNO6", "PN06") if patient == "PN06" else raw_file
        if filename not in headers:
            raise ValueError(f"Unknown parent EDF: {filename}")
        header = headers[filename]
        if not 0 < header.duration_s < 86400:
            raise ValueError("Date-free Siena clocks require an EDF shorter than one day")
        start_raw = field("Registration start time", False)
        end_raw = field("Registration end time", False)
        inherited = not bool(start_raw)
        if inherited:
            if filename not in registrations:
                raise ValueError(f"No registration clock to inherit for {filename}")
            effective_start, effective_end = registrations[filename]
        else:
            effective_start, effective_end = start_raw, end_raw
            if filename in registrations and registrations[filename] != (start_raw, end_raw):
                # PN10-7's explicit spacing typo is the only lexical normalization.
                prior = registrations[filename]
                if tuple(x.replace("1 6.49.25", "16.49.25") for x in prior) != (start_raw, end_raw):
                    raise ValueError(f"Conflicting registration clocks for {filename}")
            registrations[filename] = (start_raw, end_raw)
        notes = []
        normalized_start = effective_start
        if patient == "PN10" and filename == "PN10-7.8.9.edf" and effective_start == "1 6.49.25":
            normalized_start = "16.49.25"
            notes.append("Source registration typo '1 6.49.25' normalized to '16.49.25'")
        source_start = clock_seconds(normalized_start)
        header_start = clock_seconds(header.start_time)
        source_end = clock_seconds(effective_end)
        source_end_relative = relative_seconds(source_end, header_start)
        end_discrepancy = source_end_relative - header.duration_s
        if end_discrepancy:
            notes.append(f"Source registration end differs from sample-derived EDF end by {end_discrepancy:g} s; EDF duration retained")
        discrepancy = (header_start - source_start + 43200) % 86400 - 43200
        if discrepancy:
            if (patient, number, discrepancy) != ("PN14", 3, 10800):
                raise ValueError(f"Unadjudicated registration/header discrepancy: {patient}-{number}")
            if relative_seconds(source_end, header_start) != header.duration_s:
                raise ValueError("PN14-3 header alignment is not supported by recording end/duration")
            notes.append("EDF header + sample duration agrees with next-day source end; source start differs by 10800 s")
        onset_raw, offset_raw = field("Seizure start time"), field("Seizure end time")
        starts, ends = parse_candidates(onset_raw), parse_candidates(offset_raw)
        for candidate in starts + ends:
            candidate["relative_s"] = relative_seconds(candidate["clock_s"], header_start)
            candidate["day_offset_from_edf_start"] = int(candidate["clock_s"] < header_start)
        # Preserve documented clinical onset for mixed-reference primary labels.
        chosen = next((c for c in starts if c["kind"] == "clinical"), starts[0])
        onset = chosen["relative_s"]
        offsets = sorted({c["relative_s"] for c in ends})
        offset = offsets[0]
        quarantine = patient == "PN00" and number == 3
        reason = "Unverified out-of-EDF source end; entire parent quarantined" if quarantine else ""
        uncertainty = []
        if len(offsets) > 1:
            uncertainty.append([offsets[0], offsets[-1]])
            notes.append("Conservative shorter end; candidate tail is annotation-uncertain, never ordinary negative")
        if len(starts) > 1:
            uncertainty.append([min(c["relative_s"] for c in starts), max(c["relative_s"] for c in starts)])
            notes.append("Clinical onset is primary documented reference; interval before electrographic onset remains uncertain")
        if chosen["kind"] == "clinical" and not any(c["kind"] == "electrographic" for c in starts):
            notes.append("Clinical-only reference; electrographic onset not supplied")
        if quarantine:
            if onset != 765 or offsets != [4425] or header.duration_s != 2509:
                raise ValueError("PN00-3 evidence changed; quarantine decision requires review")
            canonical_offset = None
            uncertainty = [[0, header.duration_s]]
            notes.append("18.29.29 / 825 s is an unverified proposal only; no traceable correction source")
        else:
            canonical_offset = offset
            for start in starts:
                for end in ends:
                    validate_interval(start["relative_s"], end["relative_s"], header.duration_s)
        rows.append(dict(
            patient=patient, canonical_file_name=filename, event_id=f"{patient}-{number}", seizure_number=number,
            raw_file_name=raw_file, registration_start_raw=start_raw, registration_end_raw=end_raw,
            registration_start_effective_raw=effective_start, registration_end_effective_raw=effective_end,
            registration_inherited=inherited, seizure_start_raw=onset_raw, seizure_end_raw=offset_raw,
            edf_start_date=header.start_date, edf_start_time=header.start_time, timing_reference="EDF header",
            registration_discrepancy_s=discrepancy, source_clock_onset_relative_s=relative_seconds(chosen["clock_s"], source_start),
            source_registration_end_relative_s=source_end_relative, registration_end_discrepancy_s=end_discrepancy,
            onset_relative_s=onset, offset_relative_s_raw=ends[0]["relative_s"],
            offset_relative_s_canonical=canonical_offset, onset_type=chosen["kind"],
            clinical_onset_s=next((c["relative_s"] for c in starts if c["kind"] == "clinical"), None),
            electrographic_onset_s=next((c["relative_s"] for c in starts if c["kind"] == "electrographic"), None),
            onset_candidates_json=json.dumps(starts), offset_candidates_json=json.dumps(ends),
            proposed_offset_raw="18.29.29" if quarantine else "", proposed_offset_s=825 if quarantine else None,
            proposal_verified=False if quarantine else "", uncertainty_flag=bool(uncertainty),
            uncertain_intervals_json=json.dumps(uncertainty), primary_eligible=not quarantine,
            eligibility_status="quarantined" if quarantine else "primary_eligible",
            exclusion_reason=reason, decision_reason="; ".join(notes) or "Single documented reference; EDF-header alignment",
            interval_convention="[onset, offset)", edf_duration_s=header.duration_s,
            source_path=source, source_sha256=source_hash, policy_version=POLICY,
        ))
    if not rows:
        raise ValueError(f"No events parsed for {patient}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_inventory(root: Path = PROJECT_ROOT) -> dict:
    raw, destination = root / "data/raw/siena-scalp-eeg-1.0.0", root / "metadata"
    events, recordings, channels, sources = [], [], [], {}
    for patient in PILOT_PATIENTS:
        annotation = raw / patient / f"Seizures-list-{patient}.txt"
        source = annotation.relative_to(root).as_posix()
        source_hash = sha256(annotation)
        sources[source] = dict(kind="file", sha256=source_hash)
        headers = {p.name: read_header(p) for p in sorted((raw / patient).glob("*.edf"))}
        if not headers:
            raise ValueError(f"No selected EDFs for {patient}")
        patient_events = parse_events(patient, annotation.read_text(encoding="utf-8"), headers, source, source_hash)
        if set(headers) != {e["canonical_file_name"] for e in patient_events}:
            raise ValueError(f"Unannotated selected EDF for {patient}; eligibility requires review")
        events.extend(patient_events)
        for filename, header in headers.items():
            rows = channel_rows(header, annotation)
            edf_source = header.path.relative_to(root).as_posix()
            header_hash = header_sha256(header.path)
            sources[edf_source] = dict(kind="edf_header", sha256=header_hash, size_bytes=header.path.stat().st_size)
            parent_events = [e for e in patient_events if e["canonical_file_name"] == filename]
            eligible = all(e["primary_eligible"] for e in parent_events)
            recordings.append(dict(patient=patient, canonical_file_name=filename, edf_duration_s=header.duration_s,
                                   edf_start_date=header.start_date, edf_start_time=header.start_time,
                                   primary_eligible=eligible, event_count=len(parent_events),
                                   eligible_event_count=sum(e["primary_eligible"] for e in parent_events),
                                   exclusion_reason="; ".join(sorted({e["exclusion_reason"] for e in parent_events if e["exclusion_reason"]})),
                                   mapping_status="validated", n_eeg=sum(r["modality"] == "EEG" for r in rows),
                                   n_ecg=sum(r["modality"] == "ECG" for r in rows),
                                   edf_header_sha256=header_hash, edf_source_path=edf_source))
            for row in rows:
                channels.append(dict(patient=patient, canonical_file_name=filename, **row,
                                     source_path=source, source_sha256=source_hash, edf_header_sha256=header_hash))
    # All validation completes before replacing any canonical tables. Old audits/results are untouched.
    destination.mkdir(exist_ok=True)
    for name, rows in zip(TABLES, (events, recordings, channels)):
        write_csv(destination / name, rows)
    implementation = {f"src/multimodal_seizure/{name}.py": sha256(root / f"src/multimodal_seizure/{name}.py")
                      for name in ("inventory", "siena", "edf")}
    manifest = dict(policy_version=POLICY, sources=sources, implementation=implementation,
                    tables={name: sha256(destination / name) for name in TABLES},
                    inventory=dict(selected_edfs=len(recordings), eligible_edfs=sum(r["primary_eligible"] for r in recordings),
                                   source_events=len(events), eligible_events=sum(e["primary_eligible"] for e in events),
                                   total_seconds=sum(r["edf_duration_s"] for r in recordings),
                                   eligible_seconds=sum(r["edf_duration_s"] for r in recordings if r["primary_eligible"]),
                                   quarantined_events=[e["event_id"] for e in events if not e["primary_eligible"]],
                                   uncertain_events=[e["event_id"] for e in events if e["uncertainty_flag"]]))
    manifest["metadata_identity"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    from .provenance import artifact_status
    historical = [root / name for name in ("seizure_events_audit.csv", "official_channel_map.csv", "pilot_window_index.csv",
                  "pilot_window_summary.csv", "pn00_smoke_features.csv", "pn00_smoke_benchmark.csv", "ecg_signal_qc.csv",
                  "metadata/seizure_event_audit.csv", "metadata/patient_summary.csv")]
    historical += sorted((root / "results").glob("*/*.csv"))
    manifest["downstream_artifacts"] = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p),
        status=artifact_status(p, manifest["metadata_identity"], root),
        reason="Legacy or changed annotation/mapping provenance; preserve as historical evidence") for p in historical if p.exists()]
    (destination / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
