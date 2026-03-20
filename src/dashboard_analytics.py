from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def _latest_year(df: pd.DataFrame) -> int:
    # Assumes `year` is present.
    years = pd.to_numeric(df["year"], errors="coerce").dropna()
    return int(years.max())


def compute_latest_gender_values(
    df: pd.DataFrame,
    indicator: str,
    *,
    latest_year: Optional[int] = None,
    genders: Tuple[str, str] = ("Men", "Women"),
) -> Dict[str, float]:
    """
    Compute mean indicator values for baseline and comparison genders in the latest year.
    """
    work = df.copy()
    work = work[work["indicator"] == indicator]
    if len(work) == 0:
        return {"latest_year": float("nan"), genders[0]: float("nan"), genders[1]: float("nan")}

    if latest_year is None:
        latest_year = _latest_year(work)
    work = work[work["year"] == latest_year]

    work["value"] = pd.to_numeric(work["value"], errors="coerce")

    baseline_gender, compare_gender = genders
    out: Dict[str, float] = {"latest_year": float(latest_year)}

    for g in [baseline_gender, compare_gender]:
        sub = work[work["gender"] == g]
        mean_val = float(sub["value"].mean()) if len(sub) else float("nan")
        out[g] = mean_val

    return out


def compute_gender_gap(
    df: pd.DataFrame,
    indicator: str,
    *,
    baseline_gender: str = "Men",
    compare_gender: str = "Women",
    latest_year: Optional[int] = None,
) -> Dict[str, float]:
    """
    Gap is computed as baseline - compare (so a positive gap means the comparison group is worse).
    """
    values = compute_latest_gender_values(
        df,
        indicator,
        latest_year=latest_year,
        genders=(baseline_gender, compare_gender),
    )

    baseline = values.get(baseline_gender, float("nan"))
    compare = values.get(compare_gender, float("nan"))

    out: Dict[str, float] = {
        "indicator": float("nan"),
        "latest_year": values["latest_year"],
        "baseline_gender_value": baseline,
        "compare_gender_value": compare,
        "gap_points": float("nan"),
        "gap_pct": float("nan"),
    }

    if pd.notna(baseline) and pd.notna(compare):
        gap = float(baseline - compare)
        out["gap_points"] = gap
        out["gap_pct"] = float((gap / baseline) * 100.0) if baseline != 0 else float("nan")
    return out


def compute_gaps_for_indicators(
    df: pd.DataFrame,
    indicators: List[str],
    *,
    baseline_gender: str = "Men",
    compare_gender: str = "Women",
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for indicator in indicators:
        gap = compute_gender_gap(
            df,
            indicator,
            baseline_gender=baseline_gender,
            compare_gender=compare_gender,
        )
        gap["indicator"] = indicator  # type: ignore[assignment]

        # Add missingness info in the latest year for that indicator (for evidence).
        if pd.notna(gap["latest_year"]):
            latest_year = int(gap["latest_year"])
            sub = df[(df["indicator"] == indicator) & (df["year"] == latest_year)]
            missing_rate = float(sub["data_quality_flag"].eq("missing_value").mean()) if len(sub) else 0.0
        else:
            missing_rate = 1.0
        gap["missing_rate"] = missing_rate  # type: ignore[assignment]

        rows.append(gap)  # type: ignore[arg-type]

    return pd.DataFrame(rows)

