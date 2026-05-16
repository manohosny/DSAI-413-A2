"""Central configuration loader.

Loads ``config.yaml`` once and exposes it as a nested-attribute object so
the rest of the codebase never hardcodes paths, model IDs, or hyper-params.
Secrets come from the ``.env`` file (loaded here) and are read via helpers
that fail loudly when a required key is missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = two levels above this file: src/cxr/config.py -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from the repo root if present (no error if it is absent).
load_dotenv(PROJECT_ROOT / ".env")


class _Section:
    """Recursively wraps a dict so values are reachable as attributes."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, _Section(value) if isinstance(value, dict) else value)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"_Section({self.__dict__})"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class Config(_Section):
    """Top-level config with absolute-path resolution for the ``paths`` block."""

    def __init__(self, path: str | Path | None = None) -> None:
        cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        super().__init__(raw)

    def path(self, name: str) -> Path:
        """Return an absolute Path for an entry in the ``paths:`` block.

        The directory is created on access so callers never worry about it.
        """
        rel = getattr(self.paths, name)
        resolved = (PROJECT_ROOT / rel).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


def get_secret(name: str, required: bool = True) -> str | None:
    """Read a secret from the environment.

    Raises a clear error when a required secret is missing rather than
    failing deep inside an API client with a cryptic message.
    """
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(
            f"Missing required secret '{name}'. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


# A single shared instance — import this everywhere.
CONFIG = Config()
