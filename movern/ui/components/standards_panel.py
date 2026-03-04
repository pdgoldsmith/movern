"""Standards mapping panel widget."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from movern.ui.report.standards_map import STANDARDS_MAP


def standards_panel(results: List[Dict[str, Any]]) -> None:
    """Render a compliance mapping table for the metrics in ``results``.

    Parameters
    ----------
    results : list of dict
        Assessment results containing ``"metric"`` keys.
    """
    metric_keys = list({r.get("metric", "") for r in results if r.get("metric")})
    rows = []
    for key in metric_keys:
        mapping = STANDARDS_MAP.get(key)
        if mapping:
            rows.append({
                "Metric": mapping["display"],
                "EU AI Act": mapping["eu_ai_act"],
                "NIST AI RMF": mapping["nist_ai_rmf"],
                "ISO 42001": mapping["iso_42001"],
                "Description": mapping["description"],
            })

    if not rows:
        st.info("No standards mappings found for the assessed metrics.")
        return

    df = pd.DataFrame(rows)
    st.subheader("Compliance Mapping")
    st.markdown(
        "The table below maps each assessed metric to the relevant regulatory requirements."
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
