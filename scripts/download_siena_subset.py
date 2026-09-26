from __future__ import annotations
from pathlib import Path
import concurrent.futures as cf
import hashlib
import math
import os
import subprocess
import urllib.request

BASE = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "siena-scalp-eeg-1.0.0"
SELECTED = ("PN00", "PN06", "PN10", "PN12", "PN14")
FILES = [
    "LICENSE.txt", "RECORDS", "subject_info.csv",
    "PN00/PN00-1.edf", "PN00/PN00-2.edf", "PN00/PN00-3.edf",
    "PN00/PN00-4.edf", "PN00/PN00-5.edf", "PN00/Seizures-list-PN00.txt",
    "PN06/PN06-1.edf", "PN06/PN06-2.edf", "PN06/PN06-3.edf",
    "PN06/PN06-4.edf", "PN06/PN06-5.edf", "PN06/Seizures-list-PN06.txt",
    "PN10/PN10-1.edf", "PN10/PN10-2.edf", "PN10/PN10-3.edf",
    "PN10/PN10-4.5.6.edf", "PN10/PN10-7.8.9.edf", "PN10/PN10-10.edf",
    "PN10/Seizures-list-PN10.txt",
    "PN12/PN12-1.2.edf", "PN12/PN12-3.edf", "PN12/PN12-4.edf",
    "PN12/Seizures-list-PN12.txt",
    "PN14/PN14-1.edf", "PN14/PN14-2.edf", "PN14/PN14-3.edf",
    "PN14/PN14-4.edf", "PN14/Seizures-list-PN14.txt",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_checksums() -> dict[str, str]:
    sums = {}
    text = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        digest, rel = line.split(maxsplit=1)
        sums[rel.replace("\\", "/")] = digest
    return sums

def remote_size(rel: str) -> int:
    req = urllib.request.Request(f"{BASE}/{rel}", method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as response:
        return int(response.headers["Content-Length"])

def chunk_count(size: int) -> int:
    if size < 32 * 1024 * 1024:
        return 1
    if size < 512 * 1024 * 1024:
        return 4
    return 8
def download_chunk(task: tuple[str, int, int, int, int]) -> tuple[str, int]:
    rel, index, total, start, end = task
    final = ROOT / Path(rel)
    chunk = Path(str(final) + f".part{index:02d}")
    chunk.parent.mkdir(parents=True, exist_ok=True)
    if chunk.exists():
        chunk.unlink()
    cmd = [
        "curl.exe", "--silent", "--show-error", "--fail", "--location",
        "--retry", "5", "--retry-delay", "2",
        "--range", f"{start}-{end}", "--output", str(chunk),
        f"{BASE}/{rel}",
    ]
    result = subprocess.run(cmd)
    if result.returncode:
        raise RuntimeError(f"curl failed ({result.returncode}): {rel} chunk {index}")
    expected = end - start + 1
    actual = chunk.stat().st_size
    if actual != expected:
        raise RuntimeError(f"size mismatch: {rel} chunk {index}: {actual} != {expected}")
    print(f"CHUNK {rel} {index + 1}/{total}", flush=True)
    return rel, index

def assemble(rel: str, size: int, checksum: str) -> None:
    final = ROOT / Path(rel)
    n = chunk_count(size)
    temp = Path(str(final) + ".assembling")
    with temp.open("wb") as out:
        for i in range(n):
            part = Path(str(final) + f".part{i:02d}")
            with part.open("rb") as src:
                while block := src.read(8 * 1024 * 1024):
                    out.write(block)
    if temp.stat().st_size != size:
        raise RuntimeError(f"assembled size mismatch: {rel}")
    digest = sha256(temp)
    if digest != checksum:
        raise RuntimeError(f"SHA256 mismatch after fresh download: {rel}")
    os.replace(temp, final)
    for i in range(n):
        Path(str(final) + f".part{i:02d}").unlink(missing_ok=True)
    print(f"VERIFIED {rel} {size}", flush=True)

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    sums = load_checksums()
    verified, pending = [], []
    for rel in FILES:
        final = ROOT / Path(rel)
        if final.is_file() and sha256(final) == sums[rel]:
            verified.append(rel)
        else:
            pending.append(rel)
            final.unlink(missing_ok=True)
            Path(str(final) + ".assembling").unlink(missing_ok=True)
            for i in range(8):
                Path(str(final) + f".part{i:02d}").unlink(missing_ok=True)
    print(f"PREVERIFIED {len(verified)}", flush=True)
    print(f"PENDING {len(pending)}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        sizes = dict(zip(pending, ex.map(remote_size, pending)))
    tasks = []
    for rel in pending:
        size = sizes[rel]
        n = chunk_count(size)
        step = math.ceil(size / n)
        for i in range(n):
            start = i * step
            end = min(size - 1, (i + 1) * step - 1)
            tasks.append((rel, i, n, start, end))
    print(f"CHUNKS {len(tasks)}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(download_chunk, tasks))
    for rel in pending:
        assemble(rel, sizes[rel], sums[rel])
    print("ALL_SELECTED_FILES_VERIFIED", flush=True)

if __name__ == "__main__":
    main()
