from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    data_raw: Path = BASE_DIR / "data" / "synthetic"
    data_processed: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "models"


@dataclass
class GenerationConfig:
    n_customers: int = 500
    random_seed: int = 42


paths = PathsConfig()
generation = GenerationConfig()
