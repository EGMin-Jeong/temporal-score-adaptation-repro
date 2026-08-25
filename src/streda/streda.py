from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from streda.anchors import (
    BetaCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    clip_scores,
)
from streda.venn_abers import VennAbersCalibrator


class STREDA:
    name = "streda"
    supported_anchor_names = {"beta", "platt", "isotonic", "venn_abers", "raw"}

    def __init__(
        self,
        eps: float = 1e-6,
        residual_reg_lambda: float = 1e-2,
        residual_cap_quantile: float = 0.95,
        probability_gammas: list[float] | None = None,
        ranking_gammas: list[float] | None = None,
        decision_gammas: list[float] | None = None,
        anchor_candidates: list[str] | None = None,
        ece_weight: float = 0.25,
        nll_weight: float = 0.05,
        fpr_weight: float = 1.0,
        fpr_budget: float | None = None,
        fpr_slack: float = 0.0025,
        min_decision_f1_ratio: float = 0.9,
        min_predicted_positive: int = 1,
    ):
        self.eps = float(eps)
        self.residual_reg_lambda = float(residual_reg_lambda)
        self.residual_cap_quantile = float(np.clip(residual_cap_quantile, 0.5, 1.0))
        self.probability_gammas = probability_gammas or [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.ranking_gammas = ranking_gammas or [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        self.decision_gammas = decision_gammas or [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        requested_anchors = anchor_candidates or ["beta", "platt", "isotonic", "venn_abers", "raw"]
        self.anchor_candidates = [name for name in requested_anchors if name in self.supported_anchor_names]
        if not self.anchor_candidates:
            self.anchor_candidates = ["beta", "platt", "isotonic", "venn_abers", "raw"]
        self.ece_weight = float(ece_weight)
        self.nll_weight = float(nll_weight)
        self.fpr_weight = float(fpr_weight)
        self.fpr_budget = None if fpr_budget is None else float(fpr_budget)
        self.fpr_slack = float(fpr_slack)
        self.min_decision_f1_ratio = float(np.clip(min_decision_f1_ratio, 0.0, 1.0))
        self.min_predicted_positive = int(max(min_predicted_positive, 1))

        self.context_feature_names_ = None
        self.feature_scaler_ = None
        self.anchor_calibrator_ = None
        self.anchor_name_ = None
        self.coefficients_ = None
        self.residual_center_ = 0.0
        self.residual_cap_ = self.eps
        self.anchor_states_ = {}
        self.probability_head_ = None
        self.ranking_head_ = None
        self.decision_head_ = None
        self.probability_anchor_name_ = None
        self.ranking_anchor_name_ = None
        self.decision_anchor_name_ = None
        self.decision_score_kind_ = "point"
        self.probability_gamma_ = 0.0
        self.ranking_gamma_ = 0.0
        self.decision_gamma_ = 0.0
        self.decision_threshold_ = 0.5
        self.decision_candidate_threshold_ = 0.5
        self.selected_mode_ = "streda"
        self.selected_branch_ = "probability_ranking_decision"
        self.selected_candidate_ = None
        self.selection_reference_ = "validation_temporal_holdout_no_backbone_identity"
        self.decision_source_ = "validation_selected_monotone_decision_score"
        self.training_reference_ = "fit_window_anchor_context_residual_select_window_choice"
        self.validation_nll_ = None
        self.selection_scores_ = {}

    @staticmethod
    def _sigmoid(logits):
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _logit(self, probabilities):
        p = clip_scores(probabilities, eps=self.eps)
        return np.log(p / (1.0 - p))

    def _context_array(self, context):
        x = np.asarray(context, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _feature_matrix(self, anchor_probs, context, fit=False):
        features = np.column_stack([self._logit(anchor_probs), self._context_array(context)])
        if fit:
            self.feature_scaler_ = StandardScaler()
            return self.feature_scaler_.fit_transform(features)
        return self.feature_scaler_.transform(features)

    def _head_features(self, anchor_probs, context, scaler=None, fit=False):
        features = np.column_stack([self._logit(anchor_probs), self._context_array(context)])
        if fit:
            scaler = StandardScaler()
            return scaler.fit_transform(features), scaler
        return scaler.transform(features)

    def _fit_anchor(self, name, scores, targets, sample_weight):
        if name == "beta":
            calibrator = BetaCalibrator(eps=self.eps, max_iter=1000)
        elif name == "platt":
            calibrator = PlattCalibrator(max_iter=1000)
        elif name == "isotonic":
            calibrator = IsotonicCalibrator()
        elif name == "venn_abers":
            calibrator = VennAbersCalibrator(eps=self.eps, score_round_decimals=6)
        elif name == "raw":
            calibrator = IdentityCalibrator()
        else:
            raise ValueError(f"Unsupported STREDA dual-head anchor: {name}")
        if name == "venn_abers":
            calibrator.fit(scores, targets)
        else:
            calibrator.fit(scores, targets, sample_weight=sample_weight)
        return calibrator

    def _ridge_residual(self, features, residuals, weights):
        n, d = features.shape
        if n < 2:
            return np.zeros(d, dtype=float)
        w = np.clip(np.asarray(weights, dtype=float), 0.0, np.inf)
        if float(w.sum()) <= 0:
            w = np.ones(n, dtype=float)
        normalizer = float(w.sum())
        ridge = self.residual_reg_lambda + float(np.sqrt(d / max(n, 1))) * 1e-3
        gram = (features.T * w) @ features / normalizer
        rhs = features.T @ (w * residuals) / normalizer
        return np.linalg.solve(gram + ridge * np.eye(d), rhs)

    def _residual(self, features):
        residual = features @ self.coefficients_ - self.residual_center_
        return np.clip(residual, -self.residual_cap_, self.residual_cap_)

    def _state_residual(self, state, features):
        residual = features @ state["coefficients"] - state["residual_center"]
        return np.clip(residual, -state["residual_cap"], state["residual_cap"])

    def _combine(self, anchor_probs, residual, gamma):
        return np.clip(anchor_probs + float(gamma) * residual, self.eps, 1.0 - self.eps)

    def _nll(self, probabilities, targets):
        y = np.asarray(targets, dtype=float)
        p = clip_scores(probabilities, eps=self.eps)
        return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))

    def _ece(self, targets, probabilities, n_bins=10):
        y = np.asarray(targets, dtype=float)
        p = clip_scores(probabilities, eps=self.eps)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        total = len(y)
        ece = 0.0
        for idx in range(n_bins):
            lo, hi = edges[idx], edges[idx + 1]
            mask = (p >= lo) & (p <= hi if idx == n_bins - 1 else p < hi)
            if not np.any(mask):
                continue
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
        return float(ece) if total else 0.0

    @staticmethod
    def _f1_fpr(targets, scores, threshold):
        y = np.asarray(targets, dtype=int)
        pred = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
        _precision, _recall, f1, _ = precision_recall_fscore_support(
            y, pred, labels=[1], average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(f1), float(fpr), int(tp + fp)

    def _threshold_grid(self, scores):
        s = clip_scores(scores, eps=self.eps)
        quantiles = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49)))
        fixed = np.linspace(0.05, 0.95, 19)
        values = [quantiles, fixed, [0.5]]
        unique_scores = np.unique(s)
        if len(unique_scores) <= 5000:
            values.append(unique_scores)
            if len(unique_scores) > 1:
                values.append((unique_scores[:-1] + unique_scores[1:]) / 2.0)
        else:
            values.append(np.quantile(s, np.linspace(0.001, 0.999, 999)))
        return np.unique(np.clip(np.concatenate(values), self.eps, 1.0 - self.eps))

    def _decision_score(self, candidate_scores):
        shifted = self._logit(candidate_scores) - self._logit(np.array([self.decision_candidate_threshold_]))[0]
        return np.clip(self._sigmoid(shifted), self.eps, 1.0 - self.eps)

    def fit(
        self,
        scores,
        targets,
        context,
        sample_weight=None,
        validation_scores=None,
        validation_targets=None,
        validation_context=None,
    ):
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(targets, dtype=int)
        context_array = self._context_array(context)
        if len(scores) != len(y) or len(scores) != len(context_array):
            raise ValueError("STREDA scores, targets, and context must have equal length.")
        if len(np.unique(y)) < 2:
            raise ValueError("STREDA requires both classes in its fitting window.")
        if validation_scores is None or validation_targets is None or validation_context is None:
            raise ValueError("STREDA requires labeled validation/select data.")
        val_scores = np.asarray(validation_scores, dtype=float)
        val_y = np.asarray(validation_targets, dtype=int)
        val_context = self._context_array(validation_context)
        weights = np.ones(len(y), dtype=float) if sample_weight is None else np.clip(np.asarray(sample_weight, dtype=float), 0.0, np.inf)
        if float(weights.sum()) <= 0:
            weights = np.ones(len(y), dtype=float)

        anchor_states = {}
        best_probability = None
        for anchor_name in self.anchor_candidates:
            anchor = self._fit_anchor(anchor_name, scores, y, weights)
            fit_anchor = clip_scores(anchor.predict(scores), eps=self.eps)
            val_anchor = clip_scores(anchor.predict(val_scores), eps=self.eps)
            val_decision_variants = {"point": val_anchor}
            fit_features, scaler = self._head_features(fit_anchor, context_array, fit=True)
            val_features = self._head_features(val_anchor, val_context, scaler=scaler, fit=False)
            residual_target = y.astype(float) - fit_anchor
            coefficients = self._ridge_residual(fit_features, residual_target, weights)
            fit_raw_residual = fit_features @ coefficients
            residual_center = float(np.dot(weights, fit_raw_residual) / weights.sum())
            fit_residual = fit_raw_residual - residual_center
            residual_cap = max(float(np.quantile(np.abs(fit_residual), self.residual_cap_quantile)), self.eps)
            val_residual = np.clip(val_features @ coefficients - residual_center, -residual_cap, residual_cap)
            anchor_states[anchor_name] = {
                "name": anchor_name,
                "calibrator": anchor,
                "scaler": scaler,
                "coefficients": coefficients,
                "residual_center": residual_center,
                "residual_cap": residual_cap,
                "val_anchor": val_anchor,
                "val_residual": val_residual,
                "val_decision_variants": val_decision_variants,
            }

            for gamma in self.probability_gammas:
                val_prob = self._combine(val_anchor, val_residual, gamma)
                brier = float(brier_score_loss(val_y, val_prob))
                ece = self._ece(val_y, val_prob)
                nll = self._nll(val_prob, val_y)
                loss = brier + self.ece_weight * ece + self.nll_weight * nll
                key = f"{anchor_name}_prob_gamma_{str(gamma).replace('.', '_')}"
                self.selection_scores_[key] = {"brier": brier, "ece": ece, "nll": nll, "selection_loss": loss}
                candidate = {"loss": loss, "anchor_name": anchor_name, "gamma": float(gamma)}
                if best_probability is None or (candidate["loss"], candidate["anchor_name"], candidate["gamma"]) < (
                    best_probability["loss"],
                    best_probability["anchor_name"],
                    best_probability["gamma"],
                ):
                    best_probability = candidate

        self.anchor_states_ = anchor_states
        self.probability_head_ = anchor_states[best_probability["anchor_name"]]
        self.probability_anchor_name_ = best_probability["anchor_name"]
        self.probability_gamma_ = best_probability["gamma"]

        best_ranking = None
        for anchor_name, state in anchor_states.items():
            for gamma in self.ranking_gammas:
                val_rank = self._combine(state["val_anchor"], state["val_residual"], gamma)
                ap = float(average_precision_score(val_y, val_rank))
                key = f"{anchor_name}_rank_gamma_{str(gamma).replace('.', '_')}"
                self.selection_scores_[key] = {"pr_auc": ap}
                candidate = {"pr_auc": ap, "anchor_name": anchor_name, "gamma": float(gamma)}
                if best_ranking is None or (candidate["pr_auc"], candidate["anchor_name"], candidate["gamma"]) > (
                    best_ranking["pr_auc"],
                    best_ranking["anchor_name"],
                    best_ranking["gamma"],
                ):
                    best_ranking = candidate
        self.ranking_head_ = anchor_states[best_ranking["anchor_name"]]
        self.ranking_anchor_name_ = best_ranking["anchor_name"]
        self.ranking_gamma_ = best_ranking["gamma"]

        decision_budget = self.fpr_budget
        if decision_budget is None:
            best_anchor_fpr = None
            best_anchor_f1 = None
            for state in anchor_states.values():
                anchor_f1, anchor_fpr, _ = self._f1_fpr(val_y, state["val_anchor"], 0.5)
                if best_anchor_fpr is None or (anchor_fpr, -anchor_f1) < (best_anchor_fpr, -best_anchor_f1):
                    best_anchor_fpr = anchor_fpr
                    best_anchor_f1 = anchor_f1
            anchor_f1 = best_anchor_f1
            anchor_fpr = best_anchor_fpr
            decision_budget = min(1.0, anchor_fpr + self.fpr_slack)
            self.selection_scores_["auto_fpr_budget"] = {"f1": anchor_f1, "fpr": anchor_fpr, "budget": decision_budget}

        decision_records = []
        for anchor_name, state in anchor_states.items():
            for gamma in self.decision_gammas:
                for score_kind, decision_base in state["val_decision_variants"].items():
                    val_candidate = self._combine(decision_base, state["val_residual"], gamma)
                    for threshold in self._threshold_grid(val_candidate):
                        f1, fpr, predicted_positive = self._f1_fpr(val_y, val_candidate, threshold)
                        if predicted_positive < self.min_predicted_positive:
                            continue
                        budget_excess = max(0.0, fpr - decision_budget)
                        score = f1 - self.fpr_weight * budget_excess
                        feasible = fpr <= decision_budget
                        decision_records.append(
                            {
                                "anchor_name": anchor_name,
                                "score_kind": score_kind,
                                "feasible": feasible,
                                "score": score,
                                "f1": f1,
                                "fpr": fpr,
                                "gamma": float(gamma),
                                "threshold": float(threshold),
                                "predicted_positive": predicted_positive,
                            }
                        )
        best_decision = None
        if decision_records:
            best_f1 = max(record["f1"] for record in decision_records)
            f1_floor = self.min_decision_f1_ratio * best_f1
            viable = [record for record in decision_records if record["f1"] >= f1_floor]
            feasible_viable = [record for record in viable if record["feasible"]]
            pool = feasible_viable or viable or decision_records
            if feasible_viable:
                best_record = max(pool, key=lambda record: (record["f1"], -record["fpr"], record["score"]))
            elif viable:
                best_record = min(pool, key=lambda record: (record["fpr"], -record["f1"], -record["score"]))
            else:
                best_record = max(pool, key=lambda record: (record["score"], record["f1"], -record["fpr"]))
            best_decision = (
                best_record["feasible"],
                best_record["score"],
                best_record["f1"],
                -best_record["fpr"],
                best_record["anchor_name"],
                best_record["score_kind"],
                best_record["gamma"],
                best_record["threshold"],
                best_record["predicted_positive"],
            )
        if best_decision is not None:
            (
                _feasible,
                _score,
                f1,
                neg_fpr,
                decision_anchor_name,
                self.decision_score_kind_,
                self.decision_gamma_,
                self.decision_candidate_threshold_,
                predicted_positive,
            ) = best_decision
            self.decision_head_ = anchor_states[decision_anchor_name]
            self.decision_anchor_name_ = decision_anchor_name
            self.selection_scores_["decision"] = {
                "f1": float(f1),
                "fpr": float(-neg_fpr),
                "anchor_index": float(list(anchor_states.keys()).index(decision_anchor_name)),
                "score_kind_index": float(
                    ["point"].index(self.decision_score_kind_)
                ),
                "gamma": float(self.decision_gamma_),
                "threshold": float(self.decision_candidate_threshold_),
                "predicted_positive": float(predicted_positive),
                "budget": float(decision_budget),
            }
        else:
            self.decision_head_ = self.probability_head_
            self.decision_anchor_name_ = self.probability_anchor_name_
        self.decision_threshold_ = 0.5
        self.anchor_name_ = self.probability_anchor_name_
        self.anchor_calibrator_ = self.probability_head_["calibrator"]
        self.feature_scaler_ = self.probability_head_["scaler"]
        self.coefficients_ = self.probability_head_["coefficients"]
        self.residual_center_ = self.probability_head_["residual_center"]
        self.residual_cap_ = self.probability_head_["residual_cap"]
        self.selected_candidate_ = (
            f"prob_{self.probability_anchor_name_}_gamma_{self.probability_gamma_}_"
            f"rank_{self.ranking_anchor_name_}_gamma_{self.ranking_gamma_}_"
            f"decision_{self.decision_anchor_name_}_{self.decision_score_kind_}_gamma_{self.decision_gamma_}"
        )
        self.validation_nll_ = self._nll(
            self._combine(
                self.probability_head_["val_anchor"],
                self.probability_head_["val_residual"],
                self.probability_gamma_,
            ),
            val_y,
        )
        return self

    def _predict_head_scores(self, state, scores, context, gamma, score_kind="point"):
        if score_kind != "point":
            raise ValueError(f"Unsupported frozen STREDA decision score kind: {score_kind}")
        anchor_probs = clip_scores(state["calibrator"].predict(scores), eps=self.eps)
        residual_anchor_probs = anchor_probs
        features = self._head_features(residual_anchor_probs, context, scaler=state["scaler"], fit=False)
        residual = self._state_residual(state, features)
        return anchor_probs, residual, self._combine(anchor_probs, residual, gamma)

    def predict_details(self, scores, context):
        if self.probability_head_ is None:
            raise ValueError("STREDA must be fitted before prediction.")
        scores = np.asarray(scores, dtype=float)
        anchor_probs, residual, probabilities = self._predict_head_scores(
            self.probability_head_, scores, context, self.probability_gamma_
        )
        _ranking_anchor, _ranking_residual, ranking_score = self._predict_head_scores(
            self.ranking_head_, scores, context, self.ranking_gamma_
        )
        _decision_anchor, _decision_residual, decision_candidate = self._predict_head_scores(
            self.decision_head_, scores, context, self.decision_gamma_, score_kind=self.decision_score_kind_
        )
        decision_score = self._decision_score(decision_candidate)
        n = len(probabilities)
        return {
            "probability": probabilities,
            "ranking_score": ranking_score,
            "decision_score": decision_score,
            "global_anchor_score": anchor_probs,
            "global_beta_score": anchor_probs,
            "local_residual_score": self._combine(anchor_probs, residual, 1.0),
            "trust_gate": np.full(n, self.probability_gamma_, dtype=float),
            "residual_probability": residual,
            "probability_gamma": np.full(n, self.probability_gamma_, dtype=float),
            "ranking_gamma": np.full(n, self.ranking_gamma_, dtype=float),
            "decision_gamma": np.full(n, self.decision_gamma_, dtype=float),
            "probability_anchor_name": np.full(n, self.probability_anchor_name_, dtype=object),
            "ranking_anchor_name": np.full(n, self.ranking_anchor_name_, dtype=object),
            "decision_anchor_name": np.full(n, self.decision_anchor_name_, dtype=object),
            "decision_score_kind": np.full(n, self.decision_score_kind_, dtype=object),
            "decision_candidate_threshold": np.full(n, self.decision_candidate_threshold_, dtype=float),
            "selected_mode": np.full(n, self.selected_mode_, dtype=object),
            "selected_branch": np.full(n, self.selected_branch_, dtype=object),
            "selected_candidate": np.full(n, self.selected_candidate_, dtype=object),
            "anchor_name": np.full(n, self.anchor_name_, dtype=object),
            "training_reference": np.full(n, self.training_reference_, dtype=object),
        }

    def predict(self, scores, context):
        return self.predict_details(scores, context)["probability"]
