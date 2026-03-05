"""
PDF audit report builder for Movern.

Usage::

    from movern.ui.report.pdf_builder import build_report
    report_bytes = build_report(results, metadata)
    # pass report_bytes to st.download_button
"""

import io
from datetime import date
from typing import Any, Dict, List, Optional

from movern.ui.report.standards_map import STANDARDS_MAP


def build_report(
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    evaluator_names: Optional[List[str]] = None,
) -> bytes:
    """Build a PDF audit report and return it as bytes.

    Parameters
    ----------
    results : list of dict
        Each dict has keys: ``metric``, ``value``, ``group`` (optional),
        ``evaluator``.
    metadata : dict
        Model metadata from ``load_demo_model()`` or user config.
    evaluator_names : list of str, optional
        Names of evaluators that were run.

    Returns
    -------
    bytes
        Raw .pdf file bytes suitable for ``st.download_button``.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        raise ImportError(
            "reportlab is required for report generation. "
            "Install it with: pip install reportlab"
        )

    # ---- Colours ----
    BRAND_BLUE = colors.HexColor("#2E75B6")
    HEADER_TEXT = colors.white
    LIGHT_BLUE = colors.HexColor("#D9E1F2")
    ALT_ROW = colors.HexColor("#F5F8FC")

    # ---- Styles ----
    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Title"],
        fontSize=22,
        textColor=BRAND_BLUE,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=base_styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#444444"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=base_styles["Heading1"],
        fontSize=14,
        textColor=BRAND_BLUE,
        spaceBefore=14,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=base_styles["Heading2"],
        fontSize=12,
        textColor=BRAND_BLUE,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=base_styles["Normal"],
        fontSize=10,
        spaceAfter=4,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=base_styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=2,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=base_styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#777777"),
        spaceAfter=2,
    )

    def _table_style(has_alt_rows=True, col_count=3):
        cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), HEADER_TEXT),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("WORDWRAP", (0, 0), (-1, -1), True),
        ]
        return TableStyle(cmds)

    def _alt_rows(n_rows, n_data_rows):
        cmds = []
        for i in range(1, n_data_rows + 1):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
        return cmds

    # ---- Build story ----
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Movern AI Governance Assessment Report",
        author="Movern",
    )

    story = []
    W = A4[0] - 4 * cm  # usable width

    # =========================================================================
    # COVER
    # =========================================================================
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Movern AI Governance Assessment Report", title_style))
    story.append(HRFlowable(width="80%", thickness=2, color=BRAND_BLUE, spaceAfter=12))
    story.append(Paragraph(f"Model: <b>{metadata.get('name', 'Unknown')}</b>", subtitle_style))
    story.append(Paragraph(f"Assessment Date: {date.today().isoformat()}", subtitle_style))
    if evaluator_names:
        story.append(Paragraph(f"Evaluators: {', '.join(evaluator_names)}", subtitle_style))
    story.append(
        Paragraph(
            f"EU AI Act Risk Tier: {metadata.get('eu_ai_act_risk', 'Not assessed')}",
            subtitle_style,
        )
    )
    story.append(PageBreak())

    # =========================================================================
    # 1. MODEL OVERVIEW
    # =========================================================================
    story.append(Paragraph("1. Model Overview", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BLUE, spaceAfter=8))

    def kv(label, value):
        story.append(Paragraph(f"<b>{label}:</b> {value}", body_style))

    kv("Description", metadata.get("description", "—"))
    kv("Features", ", ".join(metadata.get("features", [])) or "—")
    kv("Sensitive Features", ", ".join(metadata.get("sensitive_features", [])) or "None specified")
    kv("EU AI Act Risk Tier", metadata.get("eu_ai_act_risk", "Not assessed"))
    kv("Applicable EU AI Act Articles", ", ".join(metadata.get("eu_ai_act_articles", [])) or "—")
    story.append(Spacer(1, 0.4 * cm))

    # =========================================================================
    # 2. ASSESSMENT RESULTS
    # =========================================================================
    story.append(Paragraph("2. Assessment Results", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BLUE, spaceAfter=8))

    by_evaluator: Dict[str, List] = {}
    for r in results:
        ev = r.get("evaluator", "General")
        by_evaluator.setdefault(ev, []).append(r)

    for ev_name, ev_results in by_evaluator.items():
        story.append(Paragraph(ev_name, h2_style))

        col_widths = [W * 0.45, W * 0.25, W * 0.30]
        data = [["Metric", "Value", "Group"]]
        for r in ev_results:
            val = r.get("value")
            if isinstance(val, float):
                val = f"{val:.4f}"
            data.append([
                r.get("metric", ""),
                str(val) if val is not None else "—",
                r.get("group", "Overall"),
            ])

        t = Table(data, colWidths=col_widths, repeatRows=1)
        ts = _table_style()
        ts.add(*("BACKGROUND", (0, 1), (-1, 1), ALT_ROW))
        for i in range(1, len(data)):
            if i % 2 == 0:
                ts.add("BACKGROUND", (0, i), (-1, i), ALT_ROW)
        t.setStyle(ts)
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

    # =========================================================================
    # 3. COMPLIANCE MAPPING
    # =========================================================================
    story.append(Paragraph("3. Compliance Mapping", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BLUE, spaceAfter=8))
    story.append(
        Paragraph(
            "The table below maps each assessed metric to relevant regulatory "
            "and standards requirements.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    col_widths = [W * 0.22, W * 0.10, W * 0.24, W * 0.22, W * 0.22]
    data = [["Metric", "Value", "EU AI Act", "NIST AI RMF", "ISO 42001"]]
    seen = set()
    for r in results:
        key = r.get("metric", "")
        if key in seen:
            continue
        seen.add(key)
        mapping = STANDARDS_MAP.get(key, {})
        val = r.get("value")
        if isinstance(val, float):
            val = f"{val:.4f}"
        data.append([
            mapping.get("display", key),
            str(val) if val is not None else "—",
            mapping.get("eu_ai_act", "—"),
            mapping.get("nist_ai_rmf", "—"),
            mapping.get("iso_42001", "—"),
        ])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = _table_style(col_count=5)
    for i in range(1, len(data)):
        if i % 2 == 0:
            ts.add("BACKGROUND", (0, i), (-1, i), ALT_ROW)
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 0.6 * cm))

    # =========================================================================
    # 4. APPENDIX
    # =========================================================================
    story.append(Paragraph("4. Appendix — Attribution", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BLUE, spaceAfter=8))
    story.append(
        Paragraph(
            "This report was generated by Movern (https://github.com/movern-ai/movern), "
            "a community fork of Lens by Credo AI "
            "(https://github.com/credo-ai/credoai_lens). "
            "The assessment engine and evaluators originate from that project. "
            "Original authors: Ian Eisenberg and the Credo AI team. "
            "License: Apache 2.0.",
            small_style,
        )
    )

    doc.build(story)
    return buf.getvalue()
