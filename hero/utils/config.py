from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(project_root: str | Path, target: str | Path) -> Path:
    project_root = Path(project_root).resolve()
    target = Path(target)
    return target if target.is_absolute() else (project_root / target).resolve()

