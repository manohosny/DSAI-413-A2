"""Download the MIMIC-CXR dataset from Kaggle into ``data/raw``.

Usage:
    python scripts/download_data.py

Requires Kaggle credentials — either a ``~/.kaggle/kaggle.json`` file or
``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` set in ``.env`` (see ``.env.example``).
After downloading it prints the detected schema so you can confirm the
loader auto-detected the right columns.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# Make the `cxr` package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr.config import CONFIG  # noqa: E402


def download() -> None:
    data_dir = CONFIG.path("data_raw")
    slug = CONFIG.dataset.kaggle_slug

    # Kaggle reads KAGGLE_USERNAME / KAGGLE_KEY from the environment;
    # config.py already loaded them from .env.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading '{slug}' -> {data_dir} ...")
    api.dataset_download_files(slug, path=str(data_dir), unzip=True, quiet=False)

    # Some Kaggle versions leave the archive behind even with unzip=True.
    for archive in data_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
        archive.unlink()

    print("Download complete.\n")


def report_schema() -> None:
    """Print detected columns + dataset stats (verification step 1 of the plan)."""
    from cxr.data.loader import dataset_summary, load_dataframe

    df = load_dataframe()
    print("Normalised columns : study_id, image, report")
    print(f"Rows (non-empty)   : {len(df)}")
    print("Sample row         :")
    print(df.iloc[0].to_dict())
    print("\nDataset summary    :")
    for key, value in dataset_summary().items():
        print(f"  {key:20s}: {value}")


if __name__ == "__main__":
    download()
    report_schema()
