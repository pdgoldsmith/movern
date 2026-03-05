"""Standards mapping panel widget."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from movern.ui.report.standards_map import STANDARDS_MAP

_SECTIONS = [
    ("Fairness", "⚖️", "Metrics that measure equitable outcomes across demographic groups."),
    ("Accountability", "📋", "Metrics that establish model performance, robustness, privacy, and data quality."),
    ("Transparency", "🔍", "Metrics that explain how the model arrives at its predictions."),
]


def standards_panel(results: List[Dict[str, Any]]) -> None:
    """Render a compliance mapping organised into Fairness, Accountability, and Transparency."""
    metric_keys = list({r.get("metric", "") for r in results if r.get("metric")})

    # Build rows grouped by category
    by_category: Dict[str, List[Dict]] = {label: [] for label, _, _ in _SECTIONS}
    for key in metric_keys:
        mapping = STANDARDS_MAP.get(key)
        if not mapping:
            continue
        category = mapping.get("category", "Accountability")
        if category not in by_category:
            category = "Accountability"
        by_category[category].append({
            "Metric": mapping["display"],
            "Description": mapping["description"],
            "🇪🇺 EU AI Act": mapping["eu_ai_act"],
            "🇺🇸 NIST AI RMF": mapping["nist_ai_rmf"],
            "🌐 ISO 42001": mapping["iso_42001"],
        })

    has_any = any(rows for rows in by_category.values())
    if not has_any:
        st.info("No standards mappings found for the assessed metrics.")
        return

    st.subheader("Compliance Mapping")
    st.markdown(
        "Assessment results mapped to EU AI Act, NIST AI RMF, and ISO 42001 requirements, "
        "organised by governance category."
    )

    for label, icon, description in _SECTIONS:
        rows = by_category.get(label, [])
        if not rows:
            continue
        st.markdown(f"#### {icon} {label}")
        st.caption(description)
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("")
