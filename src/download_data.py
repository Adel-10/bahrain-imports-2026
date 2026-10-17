"""Step 1a: download the raw datasets from the Bahrain Open Data Portal.

Run from anywhere:
    python src/download_data.py                  # download anything not yet downloaded
    python src/download_data.py --force          # download everything again
    python src/download_data.py --only imports_2026 airport_cargo

What it does
------------
For each dataset listed in ``config.DATASETS`` it asks the portal's export API for the whole
dataset as one CSV and saves it, byte for byte, to ``data/raw/<name>.csv``. The raw files are
never edited; every correction happens later in ``build_dataset.py``, where it is visible and
tested.

Each download is recorded in ``data/raw/MANIFEST.json``: the portal's title and last-modified
date, the download time, the row count and a SHA-256 checksum. The portal's terms of use ask for
the download date to be cited, and the checksum lets anyone confirm they are running the analysis
on exactly the same files.

Design choices
--------------
* Files are streamed to a temporary ``.part`` file and renamed only when complete, so an
  interrupted download can never be mistaken for a finished one.
* Failed requests are retried three times with a growing pause, and the script waits a second
  between datasets so it doesn't hammer a public service.
* Files that are already downloaded are skipped unless ``--force`` is given. The portal adds new
  months to the 2026 file as they are published, so ``--force --only imports_2026`` is how the
  analysis is brought up to date.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import DATASETS, DATASETS_BY_NAME, EXPORT_URL, MANIFEST_PATH, META_URL, RAW_DIR

HEADERS = {"User-Agent": "bahrain-imports-2026 research project (python-requests)"}
CHUNK = 1 << 20  # 1 MB
RETRIES = 3


def fetch_metadata(session: requests.Session, dataset_id: str) -> dict:
    """Title and last-modified date as the portal reports them (best effort)."""
    try:
        r = session.get(META_URL.format(dataset_id=dataset_id), timeout=30)
        r.raise_for_status()
        meta = r.json()["metas"]["default"]
        return {"title": meta.get("title"), "portal_modified": meta.get("modified")}
    except (requests.RequestException, KeyError, ValueError):
        return {"title": None, "portal_modified": None}


def download_file(session: requests.Session, dataset_id: str, dest: Path) -> int:
    """Stream the CSV export to dest via a temporary .part file. Returns bytes written."""
    url = EXPORT_URL.format(dataset_id=dataset_id)
    tmp = dest.with_suffix(".csv.part")
    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, params={"delimiter": ","}, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                written = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        written += len(chunk)
                        print(f"\r    {written / 1e6:7.1f} MB", end="", flush=True)
            print()
            tmp.replace(dest)  # only a complete download gets the real name
            return written
        except requests.RequestException as err:
            print(f"\n    attempt {attempt}/{RETRIES} failed: {err}")
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {dataset_id}")


def count_rows(path: Path) -> int:
    """Data rows in a CSV (header excluded), parsed properly so quoted commas don't miscount."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def sha256(path: Path) -> str:
    """Checksum of a file, read in 1 MB blocks so large files don't need to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    """The record of earlier downloads (empty on the first run)."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    parser.add_argument("--only", nargs="+", choices=sorted(DATASETS_BY_NAME), help="download only these datasets")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    targets = [DATASETS_BY_NAME[n] for n in args.only] if args.only else DATASETS
    failed = []

    with requests.Session() as session:
        session.headers.update(HEADERS)
        for i, ds in enumerate(targets, 1):
            dest = RAW_DIR / f"{ds.name}.csv"
            print(f"[{i}/{len(targets)}] {ds.name}  ({ds.dataset_id})")
            if dest.exists() and not args.force and ds.name in manifest:
                print("    already downloaded, skipping (use --force to refresh)")
                continue
            try:
                size = download_file(session, ds.dataset_id, dest)
            except RuntimeError as err:
                print(f"    FAILED: {err}")
                failed.append(ds.name)
                continue

            # A different row count from the one seen on the portal is not an error (the 2026 file
            # grows as months are published), so it is reported rather than treated as a failure.
            rows = count_rows(dest)
            note = "" if rows == ds.rows_seen else f"  (portal showed {ds.rows_seen:,} on {ds.checked})"
            print(f"    {rows:,} rows{note}")
            manifest[ds.name] = {
                "dataset_id": ds.dataset_id,
                **fetch_metadata(session, ds.dataset_id),
                "url": EXPORT_URL.format(dataset_id=ds.dataset_id) + "?delimiter=,",
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rows": rows,
                "bytes": size,
                "sha256": sha256(dest),
            }
            # The manifest is saved after every file, so a crash part-way through keeps the record so far.
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(1)  # be gentle with the portal

    if failed:
        print(f"\nFinished with failures: {', '.join(failed)}. Run the script again to retry them.")
        return 1
    print(f"\nDone. Files are in {RAW_DIR}. Next: python src/check_downloads.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
