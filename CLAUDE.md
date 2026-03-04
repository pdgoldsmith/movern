# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Movern** (Model + Govern) is a Python framework for AI model governance assessment. It is a community fork of [Lens by Credo AI](https://github.com/credo-ai/credoai_lens). The active package is `movern/` (not `credoai/`, which is preserved for reference). Movern adds a Streamlit UI, four pre-trained demo models, and Word report generation mapped to EU AI Act, NIST AI RMF, and ISO 42001.

## Commands

### Testing

```bash
# Run all tests
scripts/test.sh

# Run a specific test file
scripts/test.sh tests/test_binary_classification.py

# Run with verbose output
scripts/test.sh -v

# Run with live reload during development
ptw --runner "pytest -s"

# Run with coverage report
scripts/test-reports.sh
```

### Documentation

```bash
cd docs && make html
# Output in docs/_build/html/
```

### Code Formatting

```bash
black movern/
```

### Launch UI

```bash
movern ui
# or:
streamlit run movern/ui/app.py
```

### Regenerate demo models

```bash
python3 -m movern.datasets.demo_models.train_demos
```

## Architecture

The framework revolves around three core concepts: **Artifacts**, **Evaluators**, and **Lens**.

### Data Flow

1. Users wrap their models/data in **Artifact** classes (`ClassificationModel`, `TabularData`, etc.)
2. A **Lens** instance is created with those artifacts
3. **Evaluators** are added to the Lens pipeline
4. `lens.run()` orchestrates execution — validates artifact compatibility, runs each evaluator, collects results as `EvidenceContainer` objects
5. Results can be retrieved locally or exported as Word reports via the UI

### Key Components

**`movern/lens/`** — Orchestration engine
- `lens.py`: Main `Lens` class; entry point for users
- `pipeline_creator.py`: Builds evaluator pipelines

**`movern/artifacts/`** — Wrappers that standardize model/data interfaces
- `data/`: `TabularData`, `ComparisonData`
- `model/`: `ClassificationModel`, `RegressionModel`, `ComparisonModel`

**`movern/evaluators/`** — Assessment modules, each inheriting from abstract `Evaluator`
- Each evaluator declares `required_artifacts` and implements `evaluate()` returning a list of `EvidenceContainer`
- Examples: `ModelFairness`, `Privacy`, `Performance`, `ShapExplainer`, `DataProfiler`, `DeepChecks`

**`movern/modules/`** — Metric definitions and statistical utilities
- `constants_metrics.py`: Metric definitions by category
- `metrics.py`: Core `Metric` dataclass

**`movern/prism/`** — Evidence comparison engine

**`movern/governance/`** — Integration with the Credo AI platform via `credoai-connect` package

**`movern/datasets/demo_models/`** — Four pre-trained sklearn demo models
- `load_demo_model(name)` → `(pipeline, X_test, y_test, sensitive, metadata)`
- Models: `credit`, `hiring`, `healthcare`, `fraud`

**`movern/ui/`** — Streamlit assessment UI
- `app.py`: Main entry point
- `pages/`: Home, Demo Models, Own Model
- `components/`: Reusable widgets (evaluator selector, results display, standards panel)
- `report/`: Word report builder (`docx_builder.py`) and standards mapping (`standards_map.py`)

### Evaluator Pattern

All evaluators follow this pattern:

```python
class MyEvaluator(Evaluator):
    @property
    def required_artifacts(self):
        return {"model": ClassificationModel, "assessment_data": TabularData}

    def evaluate(self):
        # compute metrics, return list of EvidenceContainer
        ...
```

Lens injects matching artifacts by name when calling each evaluator.

### Test Organization

Tests in `tests/` are organized by model type (`test_binary_classification.py`, `test_regression.py`) plus integration and artifact tests. Fixtures are in `tests/fixtures/` — `datasets.py`, `lens_artifacts.py`, and `lens_inits.py` provide reusable test setups. pytest-assume is used for soft assertions.

### Optional Dependencies

Several evaluators require extra packages not installed by default:
- `shap` → `ShapExplainer`
- `lifelines` → `SurvivalAnalysis`
- `ydata-profiling` → `DataProfiler`
- `deepchecks` → `DeepChecks`
- `adversarial-robustness-toolbox` → `Privacy`
