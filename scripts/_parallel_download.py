"""One-off helper: parallel chunked resume of model.safetensors.

HuggingFace throttles single connections; this fetches the remaining bytes
in N parallel HTTP range requests, each with its own stall-detection and
resume, then concatenates onto the existing partial file.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import sys
import time

import requests

URL = "https://huggingface.co/mlx-community/medgemma-4b-it-4bit/resolve/main/model.safetensors"
TARGET = "models/medgemma-4b-it-4bit/model.safetensors"
NCHUNKS = 8


def total_size() -> int:
    r = requests.head(URL, allow_redirects=True, timeout=30)
    cl = r.headers.get("Content-Length")
    if cl:
        return int(cl)
    # Fallback: a 1-byte ranged GET exposes the size via Content-Range.
    r = requests.get(URL, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=30)
    return int(r.headers["Content-Range"].split("/")[-1])


def fetch_chunk(idx: int, a: int, b: int) -> None:
    """Download byte range [a, b] into a .partN file, resuming on failure."""
    part = f"{TARGET}.part{idx}"
    length = b - a + 1
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= length:
            return
        headers = {"Range": f"bytes={a + have}-{b}"}
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(part, "ab") as fh:
                    for block in r.iter_content(1 << 20):
                        fh.write(block)
        except Exception as exc:  # noqa: BLE001 - stall/timeout -> resume
            print(f"  [chunk {idx}] {type(exc).__name__}, resuming...", flush=True)
            time.sleep(2)


def main() -> int:
    total = total_size()
    have = os.path.getsize(TARGET) if os.path.exists(TARGET) else 0
    print(f"total={total/1e6:.1f}MB  have={have/1e6:.1f}MB  remaining={(total-have)/1e6:.1f}MB")
    if have >= total:
        print("already complete")
        return 0

    # Split the remaining range into NCHUNKS contiguous pieces.
    span = total - have
    step = span // NCHUNKS
    ranges = []
    start = have
    for i in range(NCHUNKS):
        end = total - 1 if i == NCHUNKS - 1 else start + step - 1
        ranges.append((i, start, end))
        start = end + 1

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(NCHUNKS) as pool:
        futures = [pool.submit(fetch_chunk, *r) for r in ranges]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    print(f"all chunks done in {time.time() - t0:.0f}s; concatenating...")

    # Append the parts in order onto the existing partial file.
    with open(TARGET, "ab") as out:
        for i, _, _ in ranges:
            part = f"{TARGET}.part{i}"
            with open(part, "rb") as p:
                shutil.copyfileobj(p, out, length=1 << 20)
            os.remove(part)

    final = os.path.getsize(TARGET)
    if final != total:
        print(f"SIZE MISMATCH: got {final}, expected {total}")
        return 1
    print(f"=== DOWNLOAD COMPLETE: {final/1e6:.1f}MB ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
