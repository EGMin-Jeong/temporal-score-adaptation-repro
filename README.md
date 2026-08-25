# STREDA

STREDA is a post-hoc calibration method for temporally shifted risk scores.
It learns calibration anchors and context residuals from a fitting set, then
uses labeled validation data to select independent probability, ranking, and
decision outputs. It is agnostic to the model that generated the scores.

## Install

STREDA supports Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Use

```python
from streda import STREDA

calibrator = STREDA()
calibrator.fit(
    scores=fit_scores,
    targets=fit_targets,
    context=fit_context,
    validation_scores=validation_scores,
    validation_targets=validation_targets,
    validation_context=validation_context,
)
outputs = calibrator.predict_details(test_scores, test_context)
```

`fit_scores`, `targets`, and `context` must have equal lengths. The fitting and
validation targets must each contain both classes. `predict_details` returns
`probability`, `ranking_score`, and `decision_score`.

## Verify

```bash
python -m unittest discover -s tests -v
```

The test uses synthetic data only and validates the public API.

## Scope

This repository releases the STREDA method implementation only. It does not
include datasets, base-model training code, prediction files, experiment
pipelines, or paper-result artifacts.

## Citation and license

Use `CITATION.cff` to cite this software. STREDA is released under the MIT
License. Third-party license texts for included Beta calibration and
Venn-Abers code are in `THIRD_PARTY_LICENSES.md`.
