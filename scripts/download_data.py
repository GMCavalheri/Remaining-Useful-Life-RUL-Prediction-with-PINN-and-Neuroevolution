"""Download an N-CMAPSS subset into ``data/raw/``.

Source and provenance (verified 2026-09-22): the official NASA PCoE
turbofan run-to-failure dataset ("Turbofan Engine Degradation Simulation
Data Set 2", i.e. N-CMAPSS) is a single ~14.7 GB zip of all DS01-DS08 files
at
https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
Individual subsets are also mirrored per-file on Figshare (CC BY 4.0,
uploaded by Hao Li, a co-author of the dataset's originating paper), which
is what this script downloads by default since it avoids fetching subsets
you don't need.

Usage
-----
    uv run python scripts/download_data.py --subset DS02
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

# subset -> (figshare direct-download URL, expected file size in bytes, MD5)
_SOURCES = {
    "DS02": (
        "https://ndownloader.figshare.com/files/36563133",
        2_450_472_504,
        "61056251b36290e11371e017eed70eac",
    ),
}


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_subset(subset: str, raw_dir: Path, *, verify: bool = True) -> Path:
    if subset not in _SOURCES:
        raise ValueError(
            f"No known download source for subset '{subset}'. "
            f"Known subsets: {sorted(_SOURCES)}. See this file's docstring "
            "for the full NASA bundle URL if you need a different subset."
        )
    url, expected_size, expected_md5 = _SOURCES[subset]
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"N-CMAPSS_{subset}-006.h5"

    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"{dest} already present ({expected_size:,} bytes), skipping download.")
    else:
        print(f"Downloading {url} -> {dest} ({expected_size:,} bytes expected)...")
        urlretrieve(url, dest)

    if verify:
        actual_size = dest.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Downloaded size {actual_size} != expected {expected_size} for {dest}"
            )
        actual_md5 = _md5(dest)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {dest}: got {actual_md5}, expected {expected_md5}"
            )
        print(f"Verified {dest}: size and MD5 match.")

    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="DS02", choices=sorted(_SOURCES))
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--no-verify", action="store_true", help="Skip size/MD5 verification.")
    args = parser.parse_args()

    download_subset(args.subset, args.raw_dir, verify=not args.no_verify)


if __name__ == "__main__":
    main()
