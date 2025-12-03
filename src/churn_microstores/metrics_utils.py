from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def save_metrics(metrics: Dict[str, float], output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{model_name}_metrics.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
