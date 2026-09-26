"""Fail-closed provenance checks for metadata and derived CSV artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .inventory import PROJECT_ROOT, TABLES, header_sha256, sha256


def metadata_identity(root: Path = PROJECT_ROOT) -> str:
    path = root / "metadata/canonical_manifest.json"
    if not path.exists():
        raise ValueError("Canonical metadata missing; run build_seizure_event_table.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in manifest.items() if k not in {"metadata_identity", "downstream_artifacts"}}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != manifest["metadata_identity"]:
        raise ValueError("Canonical manifest identity mismatch; rebuild metadata")
    for name in TABLES:
        if sha256(root / "metadata" / name) != manifest["tables"][name]:
            raise ValueError(f"Canonical table changed: {name}; rebuild metadata")
    for name, expected in manifest["implementation"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Metadata implementation changed: {name}; rebuild metadata")
    for name, entry in manifest["sources"].items():
        source = root / name
        actual = header_sha256(source) if entry["kind"] == "edf_header" else sha256(source)
        if actual != entry["sha256"] or ("size_bytes" in entry and source.stat().st_size != entry["size_bytes"]):
            raise ValueError(f"Canonical raw evidence changed: {name}; reverify raw integrity")
    return manifest["metadata_identity"]


def cache_identity(metadata_id: str, kind: str, config: dict, dependencies: dict) -> str:
    return hashlib.sha256(json.dumps(dict(metadata_identity=metadata_id, kind=kind, config=config,
                                          dependencies=dependencies), sort_keys=True).encode()).hexdigest()


def sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".provenance.json")


def artifact_status(path: Path, metadata_id: str, root: Path = PROJECT_ROOT) -> str:
    try:
        info = json.loads(sidecar(path).read_text(encoding="utf-8"))
        if info["metadata_identity"] != metadata_id or info["artifact_sha256"] != sha256(path):
            return "stale"
        if cache_identity(metadata_id, info["kind"], info["config"], info["dependencies"]) != info["cache_identity"]:
            return "stale"
        for name, expected in info["dependencies"].items():
            if sha256(root / name) != expected:
                return "stale"
            if sidecar(root / name).exists() and artifact_status(root / name, metadata_id, root) != "current":
                return "stale"
        return "current"
    except (OSError, ValueError, KeyError, TypeError):
        return "stale"


def require_current(path: Path, root: Path = PROJECT_ROOT, expected_identity: str | None = None) -> None:
    identity = metadata_identity(root)
    if artifact_status(path, identity, root) != "current":
        raise ValueError(f"STALE artifact: {path}. Preserve historical evidence; regenerate at a new output path.")
    if expected_identity is not None:
        info = json.loads(sidecar(path).read_text(encoding="utf-8"))
        if info["cache_identity"] != expected_identity:
            raise ValueError(f"STALE cache configuration: {path}")


def artifact_spec(kind: str, dependencies: list[Path], config: dict | None = None,
                  root: Path = PROJECT_ROOT) -> dict:
    identity = metadata_identity(root)
    hashes = {p.resolve().relative_to(root).as_posix(): sha256(p) for p in dependencies}
    config = config or {}
    return dict(metadata_identity=identity, kind=kind, config=config, dependencies=hashes,
                cache_identity=cache_identity(identity, kind, config, hashes))


def stamp_artifact(path: Path, spec: dict) -> None:
    sidecar(path).write_text(json.dumps(dict(**spec, artifact_sha256=sha256(path)), indent=2) + "\n", encoding="utf-8")


def protect_existing(path: Path) -> None:
    if path.exists():
        require_current(path)
