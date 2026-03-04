"""Home / welcome page."""

import streamlit as st


def render():
    st.title("Movern — AI Model Governance Assessment")
    st.markdown(
        """
**Movern** is an open-source framework for evaluating AI models against responsible-AI standards.
Use this tool to run structured assessments and export compliance reports for internal audits.

---

### Standards Coverage

| Standard | Areas covered |
|----------|--------------|
| **EU AI Act** | Art. 9 (Risk management), Art. 10 (Data governance), Art. 13 (Transparency), Art. 15 (Robustness) |
| **NIST AI RMF** | GOVERN, MAP, MEASURE, MANAGE functions |
| **ISO 42001** | Clause 6 (Planning), Clause 9 (Performance evaluation), Annex A controls |

---

### How to use this tool

1. **Demo Models** — Pick one of the four pre-trained demo models (Credit Risk, Hiring Screener, Healthcare Outcome, Fraud Detection) and run a full assessment.
2. **Own Model** — Upload your own trained sklearn model (`.pkl`) and assessment data (`.csv`) to run a custom assessment.
3. Download a structured **Word (.docx) audit report** after any assessment run.

---

### Attribution

Movern is a community fork of [Lens by Credo AI](https://github.com/credo-ai/credoai_lens).
The evaluators and core assessment engine originate from that project.
"""
    )
