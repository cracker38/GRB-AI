from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def generate_recommendations(
    gaps_df: pd.DataFrame,
    indicator_budget_df: pd.DataFrame,
    *,
    women_gap_threshold_points: float = 4.0,
    disability_in_selected: bool = False,
    low_income_in_selected: bool = False,
) -> List[Dict[str, str]]:
    """
    Rule-based, explainable recommendations that combine:
    - observed gender gaps (from dashboard)
    - budget GRB/targeting signals (from budget analyzer)
    """
    recs: List[Dict[str, str]] = []

    gap_work = gaps_df.copy()
    gap_work = gap_work.dropna(subset=["gap_points"])
    gap_work = gap_work.sort_values("gap_points", ascending=False)

    # Index budget signals by indicator for quick lookup.
    bwork = indicator_budget_df.set_index("indicator", drop=False)

    for _, gap_row in gap_work.iterrows():
        indicator = str(gap_row["indicator"])
        gap_points = float(gap_row["gap_points"])
        latest_year = int(gap_row["latest_year"])
        missing_rate = float(gap_row.get("missing_rate", 0.0))

        if gap_points < women_gap_threshold_points:
            # Stop early: since sorted by gap, smaller gaps won't help much.
            break

        if indicator not in bwork.index:
            continue

        budget_row = bwork.loc[indicator]

        grb_share = float(budget_row.get("grb_share_for_indicator", float("nan")))
        women_target_share = float(budget_row.get("women_targeted_share_for_indicator", float("nan")))
        disability_target_share = float(budget_row.get("disability_targeted_share_for_indicator", float("nan")))
        low_income_target_share = float(budget_row.get("low_income_targeted_share_for_indicator", float("nan")))

        evidence_parts: List[str] = []
        evidence_parts.append(
            f"In {latest_year}, women are {gap_points:.1f} points below men for `{indicator}` (filtered intersection)."
        )

        if pd.notna(women_target_share):
            evidence_parts.append(f"Women-targeted budget share for the related programs is {women_target_share*100:.0f}%.")
        if pd.notna(grb_share):
            evidence_parts.append(f"GRB-marked budget share is {grb_share*100:.0f}%.")

        if missing_rate >= 0.2:
            evidence_parts.append(f"Data completeness is low (missingness ~{missing_rate*100:.0f}%).")

        # Build action recommendation.
        actions: List[str] = []
        if pd.notna(women_target_share) and women_target_share < 0.2:
            actions.append(f"Increase funding that explicitly targets women for programs tied to `{indicator}`.")
        else:
            actions.append(f"Strengthen implementation quality for programs tied to `{indicator}` (not just labeling).")

        if disability_in_selected and pd.notna(disability_target_share) and disability_target_share < 0.2:
            actions.append(f"Add disability-inclusive components (access, outreach, reasonable accommodations) to `{indicator}` programs.")
        if low_income_in_selected and pd.notna(low_income_target_share) and low_income_target_share < 0.2:
            actions.append(f"Ensure low-income targeting and affordability measures in `{indicator}` programs.")

        if pd.isna(grb_share) or grb_share < 0.25:
            actions.append("Improve GRB marker usage and reporting standards so targeting and responsiveness are auditable.")

        recs.append(
            {
                "indicator": indicator,
                "title": f"Prioritize GRB actions for `{indicator}`",
                "evidence": " ".join(evidence_parts),
                "action": " ".join(actions),
                "expected_impact": "Better targeting should reduce the observed gender gap and improve service access for the most disadvantaged groups.",
            }
        )

        # Keep list short for readability in the UI.
        if len(recs) >= 5:
            break

    if not recs:
        recs = [
            {
                "indicator": "General",
                "title": "No major gaps detected in the current filters",
                "evidence": "Within the current selection, the gender gaps are below the prioritization threshold or budget signals are missing.",
                "action": "Try expanding the year range, selecting more locations, or including both `Men` and `Women` in the Dashboard filters.",
                "expected_impact": "This will surface clearer evidence for actionable GRB priorities.",
            }
        ]

    return recs

