from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from churn_microstores.validation import validate_features_schema, validate_label_ranges


def load_features(path: Path) -> pd.DataFrame:
    features_df = pd.read_csv(path)
    validate_features_schema(features_df)
    return features_df


def load_labels(path: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(path)
    validate_label_ranges(labels_df)
    return labels_df
