"""
Sample dataset generator for GRB-AI Navigator.

This produces two analytics-ready tables:
- gender_indicators.csv
- budget_allocations.csv

Both are generated in a normalized "long format" so the rest of the app can
apply intersectionality filters consistently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


COUNTRIES: List[str] = ["Rwanda", "Tunisia", "Armenia", "Colombia"]
YEARS: List[int] = [2020, 2021, 2022, 2023, 2024]
GENDERS: List[str] = ["Women", "Men"]
INCOME_LEVELS: List[str] = ["Low", "Middle", "High"]
DISABILITY_STATUSES: List[str] = ["No disability", "With disability"]

# A small location hierarchy for the demo (enough to show intersectionality).
LOCATION_HIERARCHY: Dict[str, Dict[str, List[str]]] = {
    "Rwanda": {
        "Eastern": ["Bugesera", "Kayonza"],
        "Western": ["Musanze", "Rubavu"],
        "Kigali": ["Gasabo", "Kicukiro"],
    },
    "Tunisia": {
        "Tunis": ["Ariana", "La Marsa"],
        "Sousse": ["Sousse_North", "Sousse_South"],
        "Sfax": ["Sfax_City", "El_Houaria"],
    },
    "Armenia": {
        "Yerevan": ["Kentron", "Avan"],
        "Ararat": ["Vedi", "Artashat"],
        "Shirak": ["Gyumri", "Ashot"],
    },
    "Colombia": {
        "Bogota": ["Teusaquillo", "Bosa"],
        "Antioquia": ["Medellin_Centro", "Medellin_Oeste"],
        "Valle_del_Cauca": ["Cali_Norte", "Cali_Sur"],
    },
}

COUNTRY_CURRENCY: Dict[str, str] = {
    "Rwanda": "RWF",
    "Tunisia": "TND",
    "Armenia": "AMD",
    "Colombia": "COP",
}


INDICATORS: List[Dict[str, object]] = [
    {
        "indicator": "School_enrollment",
        "description": "Enrollment rates for eligible learners (%).",
        "unit": "%",
        "higher_is_better": True,
        "min": 0.0,
        "max": 100.0,
        "gender_gap_strength": 10.0,
        "disability_penalty": 12.0,
        "low_income_penalty": 9.0,
    },
    {
        "indicator": "Health_services_access",
        "description": "Access to essential health services (%).",
        "unit": "%",
        "higher_is_better": True,
        "min": 0.0,
        "max": 100.0,
        "gender_gap_strength": 9.0,
        "disability_penalty": 14.0,
        "low_income_penalty": 10.0,
    },
    {
        "indicator": "GBV_services_access",
        "description": "Access to GBV services and support (%).",
        "unit": "%",
        "higher_is_better": True,
        "min": 0.0,
        "max": 100.0,
        "gender_gap_strength": 7.0,
        "disability_penalty": 15.0,
        "low_income_penalty": 11.0,
    },
    {
        "indicator": "Employment_rate",
        "description": "Employment rate (% of working-age population).",
        "unit": "%",
        "higher_is_better": True,
        "min": 0.0,
        "max": 100.0,
        "gender_gap_strength": 12.0,
        "disability_penalty": 10.0,
        "low_income_penalty": 13.0,
    },
    {
        "indicator": "Decision_making_participation",
        "description": "Participation in local decision-making (%).",
        "unit": "%",
        "higher_is_better": True,
        "min": 0.0,
        "max": 100.0,
        "gender_gap_strength": 8.0,
        "disability_penalty": 11.0,
        "low_income_penalty": 9.0,
    },
]


PROGRAM_TO_INDICATORS: Dict[str, List[str]] = {
    "Education": ["School_enrollment"],
    "Health": ["Health_services_access"],
    "GBV_response": ["GBV_services_access"],
    "Economic_support": ["Employment_rate"],
    "Governance_participation": ["Decision_making_participation"],
}


DATA_SOURCES: List[str] = [
    "Household_Survey",
    "Administrative_Health_Reports",
    "Labour_Force_Survey",
    "Community_GBV_Monitoring",
    "Civic_Participation_Records",
]


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    out_dir: str = "data"
    missing_rate_base: float = 0.03


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rand_choice(rng: np.random.Generator, values: List[str]) -> str:
    idx = int(rng.integers(0, len(values)))
    return values[idx]


def _country_context_factor(country: str) -> float:
    # A gentle baseline difference across countries (demo only).
    return {
        "Rwanda": 0.88,
        "Tunisia": 0.93,
        "Armenia": 0.86,
        "Colombia": 0.91,
    }[country]


def _income_multiplier(income_level: str) -> float:
    return {"Low": 0.82, "Middle": 0.95, "High": 1.06}[income_level]


def _disability_multiplier(disability_status: str) -> float:
    return {"No disability": 1.0, "With disability": 0.84}[disability_status]


def _gender_multiplier(indicator: str, gender: str) -> float:
    # Prototype assumption: Women tend to score lower in many indicators.
    # We encode the strength per indicator.
    if gender == "Men":
        return 1.0
    strength = next(x for x in INDICATORS if x["indicator"] == indicator)["gender_gap_strength"]  # type: ignore[index]
    # Convert "gap strength" (points) to a multiplier around 1.0.
    return 1.0 - (strength / 100.0)


def _missing_probability(
    rng: np.random.Generator,
    cfg: GenerationConfig,
    indicator: str,
    income_level: str,
    disability_status: str,
) -> float:
    # More missing values where data is typically harder to collect.
    prob = cfg.missing_rate_base
    if disability_status == "With disability":
        prob += 0.12
    if income_level == "Low":
        prob += 0.06
    if indicator in {"GBV_services_access"}:
        prob += 0.03
    # Small randomness for realism.
    prob += float(rng.uniform(-0.01, 0.01))
    return float(np.clip(prob, 0.0, 0.7))


def generate_gender_indicators(cfg: GenerationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, object]] = []

    for country in COUNTRIES:
        context = _country_context_factor(country)
        for region, districts in LOCATION_HIERARCHY[country].items():
            for district in districts:
                location_jitter = float(rng.uniform(-0.03, 0.03))
                for year in YEARS:
                    year_progress = (year - YEARS[0]) * float(rng.uniform(0.005, 0.015))
                    for income_level in INCOME_LEVELS:
                        for disability_status in DISABILITY_STATUSES:
                            for gender in GENDERS:
                                for ind in INDICATORS:
                                    indicator = str(ind["indicator"])

                                    base = 60.0 * context
                                    # Income/disability/gender adjustments
                                    val = base
                                    val *= _income_multiplier(income_level)
                                    val *= _disability_multiplier(disability_status)
                                    val *= _gender_multiplier(indicator, gender)
                                    val *= 1.0 + location_jitter

                                    # Indicator-specific penalties
                                    val -= float(ind["low_income_penalty"]) * (income_level == "Low")
                                    val -= float(ind["disability_penalty"]) * (disability_status == "With disability")

                                    # General year improvement
                                    val += 8.0 * year_progress

                                    # Noise
                                    val += float(rng.normal(0, 2.5))
                                    val = float(np.clip(val, float(ind["min"]), float(ind["max"])))

                                    # Missingness for realism
                                    miss_prob = _missing_probability(
                                        rng=rng,
                                        cfg=cfg,
                                        indicator=indicator,
                                        income_level=income_level,
                                        disability_status=disability_status,
                                    )
                                    if rng.random() < miss_prob:
                                        value: Optional[float] = None
                                        qflag = "missing_value"
                                    else:
                                        value = val
                                        qflag = "ok"

                                    source = _rand_choice(rng, DATA_SOURCES)

                                    rows.append(
                                        {
                                            "year": year,
                                            "country": country,
                                            "region": region,
                                            "district": district,
                                            "income_level": income_level,
                                            "disability_status": disability_status,
                                            "gender": gender,
                                            "indicator": indicator,
                                            "value": value,
                                            "unit": str(ind["unit"]),
                                            "source": source,
                                            "data_quality_flag": qflag,
                                        }
                                    )
    return pd.DataFrame(rows)


def generate_budget_allocations(cfg: GenerationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)  # slightly different stream
    rows: List[Dict[str, object]] = []

    program_categories = list(PROGRAM_TO_INDICATORS.keys())

    for country in COUNTRIES:
        currency = COUNTRY_CURRENCY[country]
        for region, districts in LOCATION_HIERARCHY[country].items():
            for district in districts:
                for year in YEARS:
                    # Baseline total budget at location-year level.
                    total_budget = float(rng.uniform(8e6, 20e6))

                    # Allocate across programs with some variation.
                    raw_shares = rng.uniform(0.6, 1.4, size=len(program_categories))
                    shares = raw_shares / raw_shares.sum()

                    for program_category, share in zip(program_categories, shares):
                        base_amount = total_budget * float(share)

                        # Non-targeted / less gender-responsive baseline.
                        baseline_gender_target = "Both"
                        baseline_disability_target = "All"
                        baseline_income_target = "All"
                        baseline_grb_marker = rng.random() < 0.25  # occasional marking

                        rows.append(
                            {
                                "year": year,
                                "country": country,
                                "region": region,
                                "district": district,
                                "program_category": program_category,
                                "budget_amount": float(base_amount * rng.uniform(0.6, 0.95)),
                                "currency": currency,
                                "grb_marker": bool(baseline_grb_marker),
                                "gender_target": baseline_gender_target,
                                "income_target": baseline_income_target,
                                "disability_target": baseline_disability_target,
                                "source_file": f"sample_budget_{year}.xlsx",
                            }
                        )

                        # Gender-responsive supplement (targeting often incomplete).
                        if rng.random() < 0.65:
                            supplement_amount = float(base_amount * rng.uniform(0.12, 0.38))

                            gender_target = rng.choice(["Women", "Both"], p=[0.7, 0.3]).item()  # type: ignore[call-arg]
                            disability_target = (
                                "With disability"
                                if rng.random() < 0.35
                                else rng.choice(["No disability", "All"], p=[0.15, 0.85]).item()  # type: ignore[call-arg]
                            )
                            income_target = rng.choice(["Low", "Middle", "All"], p=[0.55, 0.3, 0.15]).item()  # type: ignore[call-arg]

                            rows.append(
                                {
                                    "year": year,
                                    "country": country,
                                    "region": region,
                                    "district": district,
                                    "program_category": program_category,
                                    "budget_amount": float(supplement_amount),
                                    "currency": currency,
                                    "grb_marker": True,
                                    "gender_target": str(gender_target),
                                    "income_target": str(income_target),
                                    "disability_target": str(disability_target),
                                    "source_file": f"sample_budget_{year}.xlsx",
                                }
                            )

    return pd.DataFrame(rows)


def get_indicator_metadata() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ind in INDICATORS:
        rows.append(
            {
                "indicator": str(ind["indicator"]),
                "description": str(ind["description"]),
                "unit": str(ind["unit"]),
                "higher_is_better": bool(ind["higher_is_better"]),
                # For gap calculations we use men/no-disability as baseline in v1.
                "baseline_gender": "Men",
                "baseline_disability_status": "No disability",
                "baseline_income_level": "Middle",
            }
        )
    return pd.DataFrame(rows)


def generate_sample_datasets(cfg: Optional[GenerationConfig] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate and save sample datasets to disk, returning DataFrames too.
    """
    if cfg is None:
        cfg = GenerationConfig()

    _ensure_dir(cfg.out_dir)

    indicators_df = generate_gender_indicators(cfg)
    budgets_df = generate_budget_allocations(cfg)
    metadata_df = get_indicator_metadata()

    indicators_path = os.path.join(cfg.out_dir, "gender_indicators.csv")
    budgets_path = os.path.join(cfg.out_dir, "budget_allocations.csv")
    metadata_path = os.path.join(cfg.out_dir, "indicator_metadata.csv")

    indicators_df.to_csv(indicators_path, index=False)
    budgets_df.to_csv(budgets_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    return {
        "gender_indicators": indicators_df,
        "budget_allocations": budgets_df,
        "indicator_metadata": metadata_df,
    }


if __name__ == "__main__":
    # Allow running this as a script:
    #   python src/sample_data.py
    # (it will write CSVs into ./data)
    generate_sample_datasets()
    print("Sample datasets generated in ./data")

