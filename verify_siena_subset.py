from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent / 'data/raw/siena-scalp-eeg-1.0.0'
PATIENTS = {"PN00", "PN06", "PN10", "PN12", "PN14"}

expected = {}
for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    digest, rel = line.split(maxsplit=1)
    if rel.split("/")[0] in PATIENTS or rel in {"LICENSE.txt", "RECORDS", "subject_info.csv"}:
        expected[rel] = digest

failed = []
for rel, digest in sorted(expected.items()):
    path = ROOT / Path(rel)
    if not path.exists():
        print(f"MISSING {rel}")
        failed.append(rel)
        continue
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    status = "OK" if actual == digest else "FAIL"
    print(f"{status} {rel}")
    if status != "OK":
        failed.append(rel)

print(f"VERIFIED={len(expected)-len(failed)}/{len(expected)}")
raise SystemExit(1 if failed else 0)
