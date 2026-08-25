import unittest

import numpy as np

from streda import STREDA


class STREDATest(unittest.TestCase):
    def test_fit_and_predict_details(self):
        rng = np.random.default_rng(42)
        fit_scores = rng.uniform(0.05, 0.95, size=80)
        fit_targets = (fit_scores + rng.normal(0.0, 0.20, size=80) > 0.55).astype(int)
        select_scores = rng.uniform(0.05, 0.95, size=80)
        select_targets = (select_scores + rng.normal(0.0, 0.20, size=80) > 0.55).astype(int)
        fit_context = np.column_stack([np.arange(80), fit_scores - 0.5])
        select_context = np.column_stack([np.arange(80, 160), select_scores - 0.5])
        calibrator = STREDA(anchor_candidates=["beta", "platt", "raw"])
        calibrator.fit(
            fit_scores,
            fit_targets,
            fit_context,
            validation_scores=select_scores,
            validation_targets=select_targets,
            validation_context=select_context,
        )
        details = calibrator.predict_details(select_scores, select_context)
        self.assertTrue({"probability", "ranking_score", "decision_score"}.issubset(details))
        self.assertEqual(len(details["probability"]), len(select_scores))
        self.assertTrue(np.all((details["probability"] > 0.0) & (details["probability"] < 1.0)))
