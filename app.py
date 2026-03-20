import altair as alt
import pandas as pd
import streamlit as st

from src.data_lab import load_datasets
from src.dashboard_analytics import compute_gaps_for_indicators
from src.budget_analyzer import (
    compute_budget_signals,
    compute_indicator_budget_responsiveness,
    filter_budget_data,
    normalize_budget_df,
)
from src.recommender import generate_recommendations


st.set_page_config(page_title="GRB-AI Navigator", layout="wide")
st.title("GRB-AI Navigator: Smart Gender Budgeting & Data Assistant")
st.caption("Prototype for Gender Responsive Budgeting (GRB) with intersectionality-aware analytics.")

# Curated scenarios are meant for hackathon/demo flow so judges can quickly see impact.
SCENARIOS = {
    "Rwanda: Low-income + disability (Eastern, Bugesera)": {
        "countries": ["Rwanda"],
        "regions": ["Eastern"],
        "districts": ["Bugesera"],
        "year_range": (2022, 2024),
        "income_levels": ["Low"],
        "disability_statuses": ["With disability"],
        "genders": ["Women", "Men"],
    },
    "Tunisia: Middle-income, no disability (Sousse, Sousse_North)": {
        "countries": ["Tunisia"],
        "regions": ["Sousse"],
        "districts": ["Sousse_North"],
        "year_range": (2021, 2024),
        "income_levels": ["Middle"],
        "disability_statuses": ["No disability"],
        "genders": ["Women", "Men"],
    },
    "Armenia: Low-income (Yerevan, Kentron)": {
        "countries": ["Armenia"],
        "regions": ["Yerevan"],
        "districts": ["Kentron"],
        "year_range": (2020, 2024),
        "income_levels": ["Low"],
        "disability_statuses": ["No disability"],
        "genders": ["Women", "Men"],
    },
    "Colombia: High-income + disability (Bogota, Teusaquillo)": {
        "countries": ["Colombia"],
        "regions": ["Bogota"],
        "districts": ["Teusaquillo"],
        "year_range": (2020, 2024),
        "income_levels": ["High"],
        "disability_statuses": ["With disability"],
        "genders": ["Women", "Men"],
    },
}


@st.cache_data
def _load_all():
    return load_datasets("data")


data = _load_all()
gdf = data["gender_indicators"]
bdf = data["budget_allocations"]
indicator_metadata = data["indicator_metadata"]
program_to_indicators = data["program_to_indicators"]


def _safe_multiselect(label: str, options, default=None):
    if default is None:
        default = list(options)
    return st.sidebar.multiselect(label, options=options, default=default)


# ---------------- Sidebar filters ----------------
st.sidebar.header("Intersectionality Filters")

scenario_mode = st.sidebar.toggle("Curated scenario mode (demo)", value=False)
scenario_name = st.sidebar.selectbox("Scenario", options=list(SCENARIOS.keys()), disabled=not scenario_mode)

countries = sorted(gdf["country"].unique().tolist())
selected_countries = _safe_multiselect("Country", countries, default=countries)

years = sorted(gdf["year"].dropna().unique().tolist())
min_year, max_year = int(min(years)), int(max(years))
selected_year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Regions/districts depend on selected countries
regions_all = sorted(gdf.loc[gdf["country"].isin(selected_countries), "region"].unique().tolist())
selected_regions = _safe_multiselect("Region", regions_all, default=regions_all)

districts_all = sorted(
    gdf.loc[gdf["region"].isin(selected_regions) & gdf["country"].isin(selected_countries), "district"].unique().tolist()
)
selected_districts = _safe_multiselect("District", districts_all, default=districts_all)

income_levels = INCOME_LEVELS = sorted(gdf["income_level"].unique().tolist())
selected_income_levels = _safe_multiselect("Income level", income_levels, default=income_levels)

disability_statuses = DISABILITY_STATUSES = sorted(gdf["disability_status"].unique().tolist())
selected_disability_statuses = _safe_multiselect("Disability status", disability_statuses, default=disability_statuses)

genders = sorted(gdf["gender"].unique().tolist())
selected_genders = _safe_multiselect("Gender", genders, default=genders)

if scenario_mode:
    scenario = SCENARIOS[scenario_name]
    active_countries = scenario["countries"]
    active_year_range = scenario["year_range"]
    active_regions = scenario["regions"]
    active_districts = scenario["districts"]
    active_income_levels = scenario["income_levels"]
    active_disability_statuses = scenario["disability_statuses"]
    active_genders = scenario["genders"]
else:
    active_countries = selected_countries
    active_year_range = selected_year_range
    active_regions = selected_regions
    active_districts = selected_districts
    active_income_levels = selected_income_levels
    active_disability_statuses = selected_disability_statuses
    active_genders = selected_genders


def _filter_gender_data(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["country"].isin(active_countries)]
    filtered = filtered[(filtered["year"] >= active_year_range[0]) & (filtered["year"] <= active_year_range[1])]
    filtered = filtered[filtered["region"].isin(active_regions)]
    filtered = filtered[filtered["district"].isin(active_districts)]
    filtered = filtered[filtered["income_level"].isin(active_income_levels)]
    filtered = filtered[filtered["disability_status"].isin(active_disability_statuses)]
    filtered = filtered[filtered["gender"].isin(active_genders)]
    return filtered


def _filter_gender_data_for_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same intersection filter as the dashboard, but always keeps both genders
    so we can compute a Men vs Women gap for recommendations.
    """
    filtered = df.copy()
    filtered = filtered[filtered["country"].isin(active_countries)]
    filtered = filtered[(filtered["year"] >= active_year_range[0]) & (filtered["year"] <= active_year_range[1])]
    filtered = filtered[filtered["region"].isin(active_regions)]
    filtered = filtered[filtered["district"].isin(active_districts)]
    filtered = filtered[filtered["income_level"].isin(active_income_levels)]
    filtered = filtered[filtered["disability_status"].isin(active_disability_statuses)]
    filtered = filtered[filtered["gender"].isin(["Men", "Women"])]
    return filtered


filtered_gdf = _filter_gender_data(gdf)
filtered_gdf_for_gaps = _filter_gender_data_for_gaps(gdf)

missing_rate = float(filtered_gdf["data_quality_flag"].eq("missing_value").mean()) if len(filtered_gdf) else 0.0
st.sidebar.metric("Filtered records", f"{len(filtered_gdf):,}")
st.sidebar.metric("Missingness rate", f"{missing_rate*100:.1f}%")


# ---------------- Tabs ----------------
tab_dashboard, tab_budget, tab_ai, tab_lab = st.tabs(
    ["Gender Data Dashboard", "Budget Analyzer", "AI Recommendations", "Gender Data Lab (Simulated)"]
)


with tab_dashboard:
    st.subheader("Gender Data Dashboard")
    if scenario_mode:
        st.info(f"Demo scenario in effect: `{scenario_name}` (sidebar widgets are ignored).")
    st.markdown(
        "Use the filters (or the demo scenario) to explore *who* is most affected. Charts show trends; the gap table highlights where women lag behind men in the latest year."
    )
    indicator_options = sorted(indicator_metadata["indicator"].unique().tolist())
    selected_indicators = st.multiselect("Indicators", options=indicator_options, default=indicator_options[:3])

    if not selected_indicators:
        st.info("Select at least one indicator to view charts.")
    else:
        # Judge-friendly quick summary: top 3 gender gaps for the current filters/scenario.
        all_indicator_options = sorted(indicator_metadata["indicator"].unique().tolist())
        gaps_all_df = compute_gaps_for_indicators(
            filtered_gdf_for_gaps,
            all_indicator_options,
            baseline_gender="Men",
            compare_gender="Women",
        ).sort_values("gap_points", ascending=False).head(3)

        st.markdown("### Key Insights (Latest Year)")
        for _, r in gaps_all_df.iterrows():
            if pd.isna(r.get("gap_points")):
                continue
            gap_points = float(r.get("gap_points"))
            miss = float(r.get("missing_rate", 0.0)) if not pd.isna(r.get("missing_rate", 0.0)) else 0.0
            st.write(
                f"- `{r['indicator']}`: women are about `{gap_points:.1f}` points below men (missingness ~{miss*100:.0f}%)."
            )

        view = filtered_gdf[filtered_gdf["indicator"].isin(selected_indicators)].copy()
        view["value"] = pd.to_numeric(view["value"], errors="coerce")
        # In v1 we show the mean value for each year and gender.
        chart_df = (
            view.groupby(["year", "gender", "indicator"], dropna=False)["value"]
            .mean()
            .reset_index()
            .dropna(subset=["value"])
        )

        for indicator in selected_indicators:
            one = chart_df[chart_df["indicator"] == indicator]
            st.markdown(f"**{indicator}**")
            if len(one) == 0:
                st.write("No valid data for the current filters.")
            else:
                c = (
                    alt.Chart(one)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("year:Q", title="Year"),
                        y=alt.Y("value:Q", title="Value (%)"),
                        color=alt.Color("gender:N", title="Gender"),
                        tooltip=["year", "gender", "value"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(c, use_container_width=True)

        st.divider()
        st.markdown("Filtered data preview")
        st.dataframe(filtered_gdf.sort_values(["year", "country", "region", "district"]).head(300), use_container_width=True)

        st.divider()
        st.markdown("### Gender Gap Summary (Latest Year)")
        gaps_df = compute_gaps_for_indicators(
            _filter_gender_data_for_gaps(gdf),
            selected_indicators,
            baseline_gender="Men",
            compare_gender="Women",
        )
        if len(gaps_df):
            st.dataframe(
                gaps_df.sort_values("gap_points", ascending=False)[
                    ["indicator", "latest_year", "baseline_gender_value", "compare_gender_value", "gap_points", "gap_pct", "missing_rate"]
                ],
                use_container_width=True,
            )
        else:
            st.info("No gap data available for the current filters.")


with tab_budget:
    st.subheader("Budget Analyzer")
    st.write("Upload a budget file (CSV/XLSX) or use the built-in sample budget dataset.")
    st.markdown(
        "The app normalizes programs into a common GRB schema, then shows (1) GRB-marked spending share and (2) how much linked programs explicitly target women, disability, and low-income groups."
    )
    uploaded = st.file_uploader("Upload budget (optional)", type=["csv", "xlsx"])

    if uploaded is not None:
        st.info("Upload received. We’ll parse it and run the budget GRB analysis (prototype mapping).")
        st.write(f"Filename: {uploaded.name}")
        # Parsing/normalization are implemented here for Step 5.
    else:
        st.success("Using sample budget dataset.")

    # Decide which budget dataset to analyze
    budget_df = bdf
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                raw = pd.read_csv(uploaded)
            else:
                raw = pd.read_excel(uploaded)
            budget_df = normalize_budget_df(raw, source_name=uploaded.name)
            st.success("Budget upload normalized successfully (prototype mapping).")
        except Exception as e:
            st.error(f"Could not parse/normalize the uploaded file: {e}")
            st.info("Falling back to the sample budget dataset.")
            budget_df = bdf
    else:
        st.success("Using sample budget dataset.")

    st.session_state["budget_df_for_reco"] = budget_df

    budget_filtered = filter_budget_data(
        budget_df,
        countries=active_countries,
        year_range=active_year_range,
        regions=active_regions,
        districts=active_districts,
    )

    st.markdown("### Budget GRB Signals (by Program Category)")
    if len(budget_filtered) == 0:
        st.info("No budget records match the current intersection filters.")
    else:
        budget_signals = compute_budget_signals(budget_filtered)
        st.dataframe(budget_signals.head(20), use_container_width=True)

        st.divider()
        st.markdown("### Indicator-level Budget Responsiveness")
        indicator_options = sorted(indicator_metadata["indicator"].unique().tolist())
        selected_budget_indicators = st.multiselect(
            "Indicators to connect budgets to",
            options=indicator_options,
            default=indicator_options[:3],
            key="budget_indicator_select",
        )

        if selected_budget_indicators:
            rows = []
            for indicator in selected_budget_indicators:
                rows.append(
                    compute_indicator_budget_responsiveness(
                        budget_signals,
                        program_to_indicators=program_to_indicators,
                        indicator=indicator,
                    )
                )
            ind_budget_df = pd.DataFrame(rows)

            # Connect budget signals to outcome needs (gender gaps) for transparency.
            gaps_for_budget = compute_gaps_for_indicators(
                filtered_gdf_for_gaps,
                selected_budget_indicators,
                baseline_gender="Men",
                compare_gender="Women",
            )

            gaps_for_budget = gaps_for_budget.rename(
                columns={
                    "gap_points": "women_gap_points",
                    "missing_rate": "data_missing_rate",
                }
            )

            merged = ind_budget_df.merge(
                gaps_for_budget[["indicator", "latest_year", "women_gap_points", "data_missing_rate", "gap_pct"]],
                on="indicator",
                how="left",
            )

            # Higher misalignment => large gap + low women-targeted budget share.
            merged["misalignment_index"] = merged["women_gap_points"] * (1.0 - merged["women_targeted_share_for_indicator"])
            merged = merged.sort_values("misalignment_index", ascending=False)

            st.markdown("#### Outcome gap vs. budget targeting alignment")
            st.write(
                "Interpretation: high `misalignment_index` means women’s gap is large while programs linked to this indicator "
                "target women with a small share of budget."
            )
            st.dataframe(
                merged[
                    [
                        "indicator",
                        "latest_year",
                        "women_gap_points",
                        "women_targeted_share_for_indicator",
                        "misalignment_index",
                        "data_missing_rate",
                        "gap_pct",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.info("Select at least one indicator to view budget responsiveness.")


with tab_ai:
    st.subheader("AI Recommendation Engine")
    if scenario_mode:
        st.info(f"Demo scenario in effect: `{scenario_name}`.")
    st.write(
        "Explainable rule-based recommendations that prioritize interventions where (1) the dashboard shows a gender gap "
        "and (2) budgets show weak targeting / GRB signaling."
    )
    st.markdown(
        "Each recommendation includes evidence from the dashboard (gender gap size + data missingness) and budget signals (GRB share + targeting shares)."
    )

    indicator_options = sorted(indicator_metadata["indicator"].unique().tolist())
    selected_ai_indicators = st.multiselect(
        "Prioritize indicators",
        options=indicator_options,
        default=indicator_options[:3],
        key="ai_indicator_select",
    )

    disability_in_selected = "With disability" in selected_disability_statuses
    low_income_in_selected = "Low" in selected_income_levels

    if st.button("Generate recommendations"):
        if not selected_ai_indicators:
            st.warning("Select at least one indicator.")
        else:
            gaps_df = compute_gaps_for_indicators(
                _filter_gender_data_for_gaps(gdf),
                selected_ai_indicators,
                baseline_gender="Men",
                compare_gender="Women",
            )

            # Use the sample budget signals for recommendation v1
            budget_for_ai = st.session_state.get("budget_df_for_reco", bdf)
            budget_filtered = filter_budget_data(
                budget_for_ai,
                countries=active_countries,
                year_range=active_year_range,
                regions=active_regions,
                districts=active_districts,
            )
            if len(budget_filtered) == 0:
                st.error("No sample budget records match the current filters; cannot compute budget signals.")
            else:
                budget_signals = compute_budget_signals(budget_filtered)
                rows = []
                for indicator in selected_ai_indicators:
                    rows.append(
                        compute_indicator_budget_responsiveness(
                            budget_signals,
                            program_to_indicators=program_to_indicators,
                            indicator=indicator,
                        )
                    )
                ind_budget_df = pd.DataFrame(rows)

                recs = generate_recommendations(
                    gaps_df,
                    ind_budget_df,
                    disability_in_selected=disability_in_selected,
                    low_income_in_selected=low_income_in_selected,
                )

                st.markdown("### Recommendations")
                for rec in recs:
                    with st.expander(rec["title"]):
                        st.write(rec["evidence"])
                        st.write(f"Action: {rec['action']}")
                        st.write(f"Expected impact: {rec['expected_impact']}")


with tab_lab:
    st.subheader("Gender Data Lab (Simulated)")
    st.write(
        "This simulates a Gender Data Lab that normalizes fragmented datasets into a single analytics-ready schema."
    )

    st.markdown("Available indicators")
    st.dataframe(indicator_metadata, use_container_width=True)

    st.markdown("Budget program -> indicators mapping (used for GRB analysis)")
    st.json(program_to_indicators)

