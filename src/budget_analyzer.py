from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _to_bool(series: pd.Series) -> pd.Series:
    # Converts common truthy strings to boolean.
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y", "gender_responsive", "grb"])


def normalize_budget_df(df: pd.DataFrame, *, source_name: str = "uploaded_budget") -> pd.DataFrame:
    """
    Normalize an uploaded budget file into the app's expected schema.

    Expected (after normalization):
    - year, country, region, district
    - program_category
    - budget_amount
    - currency
    - grb_marker (boolean)
    - gender_target, income_target, disability_target
    """
    work = df.copy()

    rename_map = {
        "Year": "year",
        "year": "year",
        "Country": "country",
        "country": "country",
        "Region": "region",
        "region": "region",
        "District": "district",
        "district": "district",
        "Program": "program_category",
        "Program_Category": "program_category",
        "program_category": "program_category",
        "Category": "program_category",
        "budget_amount": "budget_amount",
        "Budget_Amount": "budget_amount",
        "Amount": "budget_amount",
        "amount": "budget_amount",
        "Value": "budget_amount",
        "value": "budget_amount",
        "Currency": "currency",
        "currency": "currency",
        "grb_marker": "grb_marker",
        "GRB": "grb_marker",
        "grb": "grb_marker",
        "gender_target": "gender_target",
        "Gender_Target": "gender_target",
        "income_target": "income_target",
        "Income_Target": "income_target",
        "disability_target": "disability_target",
        "Disability_Target": "disability_target",
    }

    # Rename columns that match known keys.
    for col in list(work.columns):
        if col in rename_map:
            work = work.rename(columns={col: rename_map[col]})

    # Fill missing columns with defaults (prototype-friendly).
    if "year" not in work.columns:
        work["year"] = pd.NA
    if "country" not in work.columns:
        work["country"] = "All"
    if "region" not in work.columns:
        work["region"] = "All"
    if "district" not in work.columns:
        work["district"] = "All"
    if "program_category" not in work.columns:
        # Use a generic fallback so the app still runs.
        work["program_category"] = "General"
    if "budget_amount" not in work.columns:
        work["budget_amount"] = 0.0
    if "currency" not in work.columns:
        work["currency"] = "NA"
    if "grb_marker" not in work.columns:
        work["grb_marker"] = False
    if "gender_target" not in work.columns:
        work["gender_target"] = "All"
    if "income_target" not in work.columns:
        work["income_target"] = "All"
    if "disability_target" not in work.columns:
        work["disability_target"] = "All"

    # Coerce types for analysis.
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["budget_amount"] = pd.to_numeric(work["budget_amount"], errors="coerce").fillna(0.0)
    work["grb_marker"] = _to_bool(work["grb_marker"]).fillna(False)

    # Traceability.
    work["source_file"] = source_name

    return work[
        [
            "year",
            "country",
            "region",
            "district",
            "program_category",
            "budget_amount",
            "currency",
            "grb_marker",
            "gender_target",
            "income_target",
            "disability_target",
            "source_file",
        ]
    ]


def filter_budget_data(
    df: pd.DataFrame,
    *,
    countries: List[str],
    year_range: tuple,
    regions: List[str],
    districts: List[str],
) -> pd.DataFrame:
    work = df.copy()
    # Treat "All" in uploaded/normalized budgets as a wildcard.
    work = work[(work["country"].isin(countries)) | (work["country"] == "All")]
    work = work[(work["year"] >= year_range[0]) & (work["year"] <= year_range[1])]
    work = work[(work["region"].isin(regions)) | (work["region"] == "All")]
    work = work[(work["district"].isin(districts)) | (work["district"] == "All")]
    return work


def compute_budget_signals(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    def _sum_where(mask):
        return float(work.loc[mask, "budget_amount"].sum())

    rows: List[Dict[str, object]] = []
    for program_category, sub in work.groupby("program_category", dropna=False):
        total = float(sub["budget_amount"].sum())
        if total <= 0:
            rows.append(
                {
                    "program_category": program_category,
                    "total_budget": 0.0,
                    "grb_share": float("nan"),
                    "women_targeted_share": float("nan"),
                    "disability_targeted_share": float("nan"),
                    "low_income_targeted_share": float("nan"),
                }
            )
            continue

        grb_budget = float(sub.loc[sub["grb_marker"] == True, "budget_amount"].sum())  # noqa: E712
        women_budget = float(sub.loc[sub["gender_target"] == "Women", "budget_amount"].sum())
        disability_budget = float(sub.loc[sub["disability_target"] == "With disability", "budget_amount"].sum())
        low_income_budget = float(sub.loc[sub["income_target"] == "Low", "budget_amount"].sum())

        rows.append(
            {
                "program_category": program_category,
                "total_budget": total,
                "grb_share": grb_budget / total,
                "women_targeted_share": women_budget / total,
                "disability_targeted_share": disability_budget / total,
                "low_income_targeted_share": low_income_budget / total,
            }
        )

    return pd.DataFrame(rows).sort_values("total_budget", ascending=False)


def compute_indicator_budget_responsiveness(
    budget_signals: pd.DataFrame,
    *,
    program_to_indicators: Dict[str, List[str]],
    indicator: str,
) -> Dict[str, float]:
    """
    Roll up program-level budget signals into an indicator-level view.
    """
    # Find programs that fund this indicator.
    funding_programs = [p for p, inds in program_to_indicators.items() if indicator in inds]
    sub = budget_signals[budget_signals["program_category"].isin(funding_programs)]
    if len(sub) == 0:
        return {
            "indicator": indicator,
            "total_budget_for_indicator": 0.0,
            "grb_share_for_indicator": float("nan"),
            "women_targeted_share_for_indicator": float("nan"),
            "disability_targeted_share_for_indicator": float("nan"),
            "low_income_targeted_share_for_indicator": float("nan"),
        }

    # Budget-weighted averages.
    total = float(sub["total_budget"].sum())
    if total <= 0:
        return {
            "indicator": indicator,
            "total_budget_for_indicator": 0.0,
            "grb_share_for_indicator": float("nan"),
            "women_targeted_share_for_indicator": float("nan"),
            "disability_targeted_share_for_indicator": float("nan"),
            "low_income_targeted_share_for_indicator": float("nan"),
        }

    def wavg(col: str) -> float:
        return float((sub[col] * sub["total_budget"]).sum() / sub["total_budget"].sum())

    return {
        "indicator": indicator,
        "total_budget_for_indicator": total,
        "grb_share_for_indicator": wavg("grb_share"),
        "women_targeted_share_for_indicator": wavg("women_targeted_share"),
        "disability_targeted_share_for_indicator": wavg("disability_targeted_share"),
        "low_income_targeted_share_for_indicator": wavg("low_income_targeted_share"),
    }

