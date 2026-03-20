from __future__ import annotations

import os
from typing import Dict

import pandas as pd

from .sample_data import GenerationConfig, PROGRAM_TO_INDICATORS, generate_sample_datasets


REQUIRED_FILES = [
    "gender_indicators.csv",
    "budget_allocations.csv",
    "indicator_metadata.csv",
]


def _data_dir(data_dir: str = "data") -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), data_dir)


def _ensure_sample_data(data_dir: str) -> None:
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(data_dir, f))]
    if not missing:
        return

    os.makedirs(data_dir, exist_ok=True)
    # Write datasets into the resolved repo-relative `data/` directory.
    # Generate into the resolved repo-relative `data/` directory.
    # This avoids relying on the current working directory.
    cfg = GenerationConfig(out_dir=data_dir)
    generate_sample_datasets(cfg=cfg)


def load_datasets(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load analytics-ready datasets for the app.

    If files are missing, we generate them using `src.sample_data`.
    """
    resolved_dir = _data_dir(data_dir)
    _ensure_sample_data(resolved_dir)

    gender_indicators = pd.read_csv(os.path.join(resolved_dir, "gender_indicators.csv"))
    budget_allocations = pd.read_csv(os.path.join(resolved_dir, "budget_allocations.csv"))
    indicator_metadata = pd.read_csv(os.path.join(resolved_dir, "indicator_metadata.csv"))

    # Ensure expected dtypes (helps filtering + comparisons in the UI).
    if "year" in gender_indicators.columns:
        gender_indicators["year"] = pd.to_numeric(gender_indicators["year"], errors="coerce").astype("Int64")
    if "year" in budget_allocations.columns:
        budget_allocations["year"] = pd.to_numeric(budget_allocations["year"], errors="coerce").astype("Int64")

    # Keep currency + category columns as strings.
    for col in ["country", "region", "district", "gender", "income_level", "disability_status", "indicator"]:
        if col in gender_indicators.columns:
            gender_indicators[col] = gender_indicators[col].astype(str)

    for col in ["country", "region", "district", "program_category", "gender_target", "income_target", "disability_target"]:
        if col in budget_allocations.columns:
            budget_allocations[col] = budget_allocations[col].astype(str)

    if "value" in gender_indicators.columns:
        gender_indicators["value"] = pd.to_numeric(gender_indicators["value"], errors="coerce")
    if "budget_amount" in budget_allocations.columns:
        budget_allocations["budget_amount"] = pd.to_numeric(budget_allocations["budget_amount"], errors="coerce")

    return {
        "gender_indicators": gender_indicators,
        "budget_allocations": budget_allocations,
        "indicator_metadata": indicator_metadata,
        "program_to_indicators": PROGRAM_TO_INDICATORS,
    }

