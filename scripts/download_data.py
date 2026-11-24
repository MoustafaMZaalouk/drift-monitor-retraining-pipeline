"""Download the two UCI datasets and verify pinned SHA-256 checksums.

Datasets are CC BY 4.0 and are NOT committed to the repo; they are fetched at
build time from the UCI Machine Learning Repository and checksum-verified.

  Bike Sharing:            Fanaee-T, H. (2013). DOI 10.24432/C5W894
  Online Shoppers Intention: Sakar et al. (2018). DOI 10.24432/C5F88Q
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ARCHIVES = {
    "bike_sharing.zip": {
        "url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "sha256": "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
        "members": ["hour.csv"],
    },
    "online_shoppers.zip": {
        "url": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        "sha256": "2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14",
        "members": ["online_shoppers_intention.csv"],
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, meta in ARCHIVES.items():
        dest = DATA_DIR / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(meta["url"], dest)
        digest = sha256(dest)
        if digest != meta["sha256"]:
            dest.unlink()
            print(f"ERROR: checksum mismatch for {name}: {digest}", file=sys.stderr)
            return 1
        with zipfile.ZipFile(dest) as z:
            z.extractall(DATA_DIR, members=meta["members"])
        print(f"{name}: checksum ok, extracted {meta['members']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
