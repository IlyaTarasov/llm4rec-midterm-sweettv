"""Configuration and path helpers for the SWEET.TV midterm pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "midterm" / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
