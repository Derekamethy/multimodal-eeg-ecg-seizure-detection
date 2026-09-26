from __future__ import annotations
import concurrent.futures
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'data/raw/siena-scalp-eeg-1.0.0'
BASE = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0"
CHUNK = 16 * 1024 * 1024
PART_WORKERS = 12
FILES = [
    "subject_info.csv", "RECORDS", "SHA256SUMS.txt", "LICENSE.txt",
    "PN00/PN00-1.edf", "PN00/PN00-2.edf", "PN00/PN00-3.edf", "PN00/PN00-4.edf", "PN00/PN00-5.edf", "PN00/Seizures-list-PN00.txt",
    "PN06/PN06-1.edf", "PN06/PN06-2.edf", "PN06/PN06-3.edf", "PN06/PN06-4.edf", "PN06/PN06-5.edf", "PN06/Seizures-list-PN06.txt",
    "PN10/PN10-1.edf", "PN10/PN10-2.edf", "PN10/PN10-3.edf", "PN10/PN10-4.5.6.edf", "PN10/PN10-7.8.9.edf", "PN10/PN10-10.edf", "PN10/Seizures-list-PN10.txt",
    "PN12/PN12-1.2.edf", "PN12/PN12-3.edf", "PN12/PN12-4.edf", "PN12/Seizures-list-PN12.txt",
    "PN14/PN14-1.edf", "PN14/PN14-2.edf", "PN14/PN14-3.edf", "PN14/PN14-4.edf", "PN14/Seizures-list-PN14.txt",
]

def remote_size(url: str) -> int:
    p = subprocess.run(["curl.exe", "--silent", "--show-error", "--fail", "--location", "--head",
                        "--http1.1", "--connect-timeout", "15", "--max-time", "30", url],
                       capture_output=True, text=True, check=True)
    vals = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", p.stdout)
    if not vals:
        raise RuntimeError(f"No Content-Length for {url}")
    return int(vals[-1])
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fetch_part(url: str, part: Path, start: int, end: int) -> Path:
    wanted = end - start + 1
    for attempt in range(1, 9):
        part.unlink(missing_ok=True)
        p = subprocess.run(["curl.exe", "--silent", "--show-error", "--fail", "--location", "--http1.1",
                            "--connect-timeout", "15", "--max-time", "120",
                            "--range", f"{start}-{end}", "--output", str(part), url],
                           capture_output=True, text=True)
        got = part.stat().st_size if part.exists() else 0
        if p.returncode == 0 and got == wanted:
            return part
        print(f"PART_RETRY {start}-{end} attempt={attempt} got={got}/{wanted}", flush=True)
    raise RuntimeError(f"Failed range {start}-{end} for {url}")

def append_missing(rel: str, target: int) -> None:
    dst = ROOT.joinpath(*rel.split("/"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    current = dst.stat().st_size if dst.exists() else 0
    if current > target:
        dst.unlink()
        current = 0
    if current == target:
        return
    url = f"{BASE}/{rel}"
    part_dir = ROOT / ".parts" / Path(rel)
    part_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    start = current
    while start < target:
        end = min(start + CHUNK - 1, target - 1)
        jobs.append((start, end, part_dir / f"{start:012d}-{end:012d}.part"))
        start = end + 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(PART_WORKERS, len(jobs))) as pool:
        futures = [pool.submit(fetch_part, url, part, start, end) for start, end, part in jobs]
        for f in futures:
            f.result()
    with dst.open("ab") as out:
        for start, end, part in jobs:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    shutil.rmtree(part_dir, ignore_errors=True)
    if dst.stat().st_size != target:
        raise RuntimeError(f"Size mismatch after assembly: {rel}")

def load_expected() -> dict[str, str]:
    result = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        result[rel] = digest
    return result

def ensure_file(rel: str, expected: dict[str, str]) -> None:
    dst = ROOT.joinpath(*rel.split("/"))
    target = remote_size(f"{BASE}/{rel}")
    for round_no in (1, 2):
        size = dst.stat().st_size if dst.exists() else 0
        if size > target:
            print(f"RESET_SIZE {rel} {size}>{target}", flush=True)
            dst.unlink()
        append_missing(rel, target)
        digest = expected.get(rel)
        if digest is None or sha256_file(dst) == digest:
            print(f"VERIFIED {rel} {target}", flush=True)
            return
        print(f"RESET_HASH {rel} round={round_no}", flush=True)
        dst.unlink(missing_ok=True)
    raise RuntimeError(f"SHA256 failed after redownload: {rel}")

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    ensure_file("SHA256SUMS.txt", {})
    expected = load_expected()
    for rel in FILES:
        if rel != "SHA256SUMS.txt":
            ensure_file(rel, expected)
    print("ALL_DOWNLOADS_VERIFIED", flush=True)
