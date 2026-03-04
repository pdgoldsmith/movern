# Movern — AI Model Governance Assessment

**Movern** (Model + Govern) is an open-source framework for evaluating AI models against responsible-AI standards.
It provides a local UI for auditors to run structured assessments and export compliance reports mapped to **EU AI Act**, **NIST AI RMF**, and **ISO 42001**.

---

## Attribution

Movern is a community fork of [**Lens** by Credo AI](https://github.com/credo-ai/credoai_lens).
The evaluators and core assessment engine originate from that project.
Original authors: Ian Eisenberg and the Credo AI team. Original license: Apache 2.0.

---

## Quick Start

```bash
pip install movern

# Launch the interactive assessment UI
movern ui
```

---

## Standards Coverage

| Standard | Areas covered |
|----------|--------------|
| **EU AI Act** | Art. 9 (Risk management), Art. 10 (Data governance), Art. 13 (Transparency), Art. 15 (Robustness) |
| **NIST AI RMF** | GOVERN, MAP, MEASURE, MANAGE functions |
| **ISO 42001** | Clause 6 (Planning), Clause 9 (Performance evaluation), Annex A controls |

---

## Demo Models

Movern ships four pre-trained demo models so auditors can explore the tool without providing their own model:

| Model | Task | Focus |
|-------|------|-------|
| **Credit Risk** | Binary classification | Fairness (age, sex), privacy, explainability |
| **Hiring Screener** | Binary classification | Fairness (gender, race), data fairness |
| **Healthcare Outcome** | Binary classification | Fairness (age, race, sex), privacy |
| **Fraud Detection** | Binary classification | Performance, robustness, feature drift |

---

## Running Your Own Model

```python
from movern.lens import Lens
from movern.artifacts import ClassificationModel, TabularData
from movern.evaluators import Performance, ModelFairness

model = ClassificationModel(name="my_model", model_like=my_sklearn_pipeline)
assessment_data = TabularData(
    name="test_data",
    X=X_test,
    y=y_test,
    sensitive_features=sensitive_df,
)

lens = Lens(model=model, assessment_data=assessment_data)
lens.add(Performance())
lens.add(ModelFairness(metrics=["demographic_parity_difference"]))
lens.run()
lens.get_results()
```

---

## Architecture

Movern revolves around three core concepts: **Artifacts**, **Evaluators**, and **Lens**.

1. Wrap your model/data in **Artifact** classes (`ClassificationModel`, `TabularData`, etc.)
2. Create a **Lens** instance with those artifacts
3. Add **Evaluators** to the pipeline
4. Call `lens.run()` — it validates compatibility, runs each evaluator, and collects results as `EvidenceContainer` objects

### Available Evaluators

| Evaluator | Purpose | Extra dep |
|-----------|---------|-----------|
| `Performance` | Accuracy, AUC, precision, recall | — |
| `ModelFairness` | Demographic parity, equalized odds | — |
| `DataFairness` | Training data distribution analysis | — |
| `Privacy` | Membership inference attacks | `adversarial-robustness-toolbox` |
| `ShapExplainer` | SHAP feature importance | `shap` |
| `DataProfiler` | Full dataset profiling | `ydata-profiling` |
| `DeepChecks` | Suite of model/data checks | `deepchecks` |
| `FeatureDrift` | Population stability index | — |
| `SurvivalAnalysis` | Kaplan-Meier, Cox models | `lifelines` |

Install all extras:
```bash
pip install "movern[full]"
```

---

## Development

```bash
# Run tests
scripts/test.sh

# Run with coverage
scripts/test-reports.sh

# Format code
black movern/

# Build docs
cd docs && make html
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
