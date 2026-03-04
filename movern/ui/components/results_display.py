"""Render assessment results in the Streamlit UI."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def display_results(results: List[Dict[str, Any]]) -> None:
    """Render a list of metric result dicts as expandable tables + bar charts.

    Parameters
    ----------
    results : list of dict
        Each dict: ``{"evaluator": str, "metric": str, "value": float|str, "group": str}``
    """
    if not results:
        st.info("No results to display.")
        return

    # Group by evaluator
    by_evaluator: Dict[str, List] = {}
    for r in results:
        ev = r.get("evaluator", "General")
        by_evaluator.setdefault(ev, []).append(r)

    for ev_name, ev_results in by_evaluator.items():
        with st.expander(f"**{ev_name}**", expanded=True):
            df = pd.DataFrame(ev_results)
            cols_to_show = [c for c in ["metric", "group", "value"] if c in df.columns]
            df_display = df[cols_to_show].copy()
            if "value" in df_display.columns:
                df_display["value"] = df_display["value"].apply(
                    lambda v: f"{v:.4f}" if isinstance(v, float) else v
                )
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Bar chart for numeric scalar metrics (no group breakdown)
            numeric_rows = [
                r for r in ev_results
                if isinstance(r.get("value"), (int, float))
                and r.get("group", "Overall") == "Overall"
            ]
            if numeric_rows:
                chart_df = pd.DataFrame(numeric_rows).set_index("metric")["value"]
                st.bar_chart(chart_df)
