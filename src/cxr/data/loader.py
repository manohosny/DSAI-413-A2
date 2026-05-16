"""MIMIC-CXR data loading.

The exact column names of the Kaggle dataset are not known ahead of time,
so this module *auto-detects* the image-path and report-text columns from
the candidate lists in ``config.yaml``. Everything downstream consumes the
normalised :class:`CxrRecord` dataclass and never touches raw column names.
"""

from __future__ import annotations

import functools
# literal_eval parses only Python literals (no code execution); aliased here.
from ast import literal_eval as _parse_literal
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image

from cxr.config import CONFIG


@dataclass(frozen=True)
class CxrRecord:
    """One chest X-ray study: an image plus its reference radiology report."""

    study_id: str
    image_path: Path
    report_text: str

    def load_image(self) -> Image.Image:
        """Load the X-ray as an RGB PIL image (MedGemma/CLIP expect 3 channels)."""
        return Image.open(self.image_path).convert("RGB")


def _find_csv(data_dir: Path) -> Path:
    """Return the dataset CSV inside ``data/raw`` (largest CSV wins ties)."""
    csvs = sorted(data_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found under {data_dir}. Run `python scripts/download_data.py` first."
        )
    return csvs[0]


def _first_list_item(value: object) -> str:
    """Return the first element of a stringified list cell.

    In this MIMIC-CXR export the ``image`` and ``text`` columns store
    *stringified Python lists* (e.g. ``"['a.jpg', 'b.jpg']"``) because a
    study has several views and the report is list-wrapped. We take the
    first element; plain (non-list) values pass through unchanged.
    """
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = _parse_literal(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0]).strip()
        return ""
    return text


def _detect_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    """Pick the first dataframe column matching a candidate (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(
        f"Could not auto-detect the {kind} column. "
        f"Columns present: {list(df.columns)}. "
        f"Add the correct name to config.yaml -> dataset.{kind}_col_candidates."
    )


def _resolve_image_path(raw_value: str, data_dir: Path) -> Path:
    """Map a CSV image reference to an absolute file path on disk.

    Handles both absolute paths and bare filenames by searching ``data/raw``.
    """
    candidate = Path(str(raw_value))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = data_dir / candidate
    if direct.exists():
        return direct
    # Fall back to a recursive search by filename (datasets nest images).
    matches = list(data_dir.rglob(candidate.name))
    return matches[0] if matches else direct


@functools.lru_cache(maxsize=1)
def load_dataframe() -> pd.DataFrame:
    """Load the raw dataset CSV with normalised ``study_id`` / ``image`` / ``report`` columns."""
    data_dir = CONFIG.path("data_raw")
    csv_path = _find_csv(data_dir)
    df = pd.read_csv(csv_path)

    img_col = _detect_column(df, list(CONFIG.dataset.image_col_candidates), "image")
    rep_col = _detect_column(df, list(CONFIG.dataset.report_col_candidates), "report")

    out = pd.DataFrame()
    out["study_id"] = (
        df["study_id"].astype(str) if "study_id" in df.columns
        else df.index.astype(str)
    )
    # image / report cells are stringified lists in this export - unwrap them.
    out["image"] = df[img_col].map(_first_list_item)
    out["report"] = df[rep_col].fillna("").map(_first_list_item).str.strip()
    # Drop rows with an empty report — useless for both modes.
    out = out[out["report"].str.len() > 0].reset_index(drop=True)
    return out


def iter_records(limit: int | None = None) -> Iterator[CxrRecord]:
    """Yield :class:`CxrRecord` objects, optionally capped at ``limit``."""
    data_dir = CONFIG.path("data_raw")
    df = load_dataframe()
    if limit is not None:
        df = df.head(limit)
    for row in df.itertuples(index=False):
        yield CxrRecord(
            study_id=row.study_id,
            image_path=_resolve_image_path(row.image, data_dir),
            report_text=row.report,
        )


def load_records(limit: int | None = None) -> list[CxrRecord]:
    """Eagerly load records into a list (convenient for indexing / evaluation)."""
    return list(iter_records(limit))


def dataset_summary() -> dict[str, object]:
    """Quick stats for sanity-checking a fresh download."""
    df = load_dataframe()
    word_counts = df["report"].str.split().str.len()
    return {
        "num_studies": len(df),
        "avg_report_words": round(float(word_counts.mean()), 1),
        "min_report_words": int(word_counts.min()),
        "max_report_words": int(word_counts.max()),
    }
