# -*- coding: utf-8 -*-
"""
Uniform ML model wrappers (LR / Linear SVM / Random Forest) for time-aware FX strategies.

Big picture
-----------
This module gives you a consistent interface for classic ML baselines on your
ternary labels {-1, 0, +1}, while staying leak-safe with your existing pipeline:

Upstream (already in your repo):
- Features are built from *past-only* indicator values and shifted by 1 bar
  to prevent look-ahead; results are cached to Parquet for reproducibility.  (see feature_store)
- Labels are ternary {-1,0,+1} using fixed-horizon forward returns with optional
  volatility gating; tail is purged; features & labels are aligned safely.      (see labels)
- Splits are strictly time-aware (walk-forward, expanding, blocked K-fold) and
  can use gap/embargo to mirror feature shift and minimize leakage.             (see splits)

This file only wraps sklearn models so Optuna/backtests don't care which model
you pick. Each wrapper implements:
    fit(X, y, sample_weight=None)
    predict_scores(X)         → continuous score (proba- or margin-based)
    predict_labels(X, threshold=0.0) → {-1, 0, +1} via dead-zone around 0.

Scoring rule
------------
For *probability-capable* models, the score is:
    score = P(y=+1) - P(y=-1)

For *margin* models (e.g., LinearSVC without calibration), the score is
the difference between decision margins for +1 and -1 (or a single signed margin
in binary cases).

You can turn on calibration (sigmoid) for well-behaved P(·) with `calibrate=True`.
Calibration uses the *last* `calibrate_frac` chunk of the training fold as a
chronological holdout, avoiding data leakage.

Notes
-----
- Works with missing classes in a given fold (e.g., no -1s): we degrade gracefully.
- Class weights default to an "auto" scheme derived from your label frequencies,
  or set class_weight='balanced'/dict/None. You can still pass sample_weight; sklearn
  multiplies them internally.
- For stability, LR uses l2 penalty and 'lbfgs' multinomial by default; SVM is LinearSVC;
  RF defaults are conservative.

Dependencies
------------
scikit-learn ≥ 1.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Pull your class weight helper (ternary-aware) from labels.py
# -> returns dict like {-1: w1, 0: w0, 1: w2} based on observed frequencies.
from scripts.machine_learning.labels import class_weights_from_series  # type: ignore

# ----------------------------- Utilities ----------------------------------


def _safe_check_X(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Ensure 2D numpy array and no NaNs/Infs."""
    if isinstance(X, pd.DataFrame):
        arr = X.values
    else:
        arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(
            "X contains NaN or Inf. Clean upstream (feature shifting/dropping)."
        )
    return arr


def _safe_check_y(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Ensure 1D numpy array and integer class labels (-1,0,1)."""
    if isinstance(y, pd.Series):
        arr = y.values
    else:
        arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {arr.shape}")
    return arr


def _combine_sample_and_class_weights(
    y: np.ndarray,
    *,
    class_weight: Union[str, Dict[int, float], None] = "auto",
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Optional[Dict[int, float]], Optional[np.ndarray]]:
    """
    Decide how to pass weights to sklearn. We return two things:
      1) class_weight parameter for the estimator (None / 'balanced' / dict)
      2) sample_weight vector to pass to fit (possibly multiplied by per-class weights)

    Strategy:
    - If class_weight == 'balanced' or dict: pass through as estimator param; keep sample_weight.
    - If class_weight == 'auto' (default): compute from y using your helper and pass as dict.
    - If class_weight is None: just pass sample_weight (or None).
    """
    out_class_weight = None
    out_sample_weight = None

    if class_weight == "balanced":
        out_class_weight = "balanced"
        out_sample_weight = sample_weight
    elif isinstance(class_weight, dict):
        out_class_weight = {int(k): float(v) for k, v in class_weight.items()}
        out_sample_weight = sample_weight
    elif class_weight == "auto":
        cw = class_weights_from_series(pd.Series(y))
        out_class_weight = cw if cw else None
        out_sample_weight = sample_weight
    elif class_weight is None:
        out_class_weight = None
        out_sample_weight = sample_weight
    else:
        raise ValueError(
            "class_weight must be one of: None, 'balanced', 'auto', or dict."
        )

    return out_class_weight, out_sample_weight


def _prob_diff(proba: np.ndarray, classes_: np.ndarray) -> np.ndarray:
    """
    Compute score = P(+1) - P(-1), safely handling missing classes in a fold.
    If P(+1) only: score = P(+1)
    If P(-1) only: score = -P(-1)
    If neither exists (degenerate): return zeros.
    """
    # classes_ like array([-1, 0, 1]) but may be subset
    idx_pos = np.where(classes_ == 1)[0]
    idx_neg = np.where(classes_ == -1)[0]

    p_pos = proba[:, idx_pos[0]] if idx_pos.size else 0.0
    p_neg = proba[:, idx_neg[0]] if idx_neg.size else 0.0

    if isinstance(p_pos, float) and isinstance(p_neg, float):
        return np.zeros(proba.shape[0], dtype=float)  # no +/- classes present
    if isinstance(p_pos, float):  # only -1 present
        return -p_neg
    if isinstance(p_neg, float):  # only +1 present
        return p_pos
    return p_pos - p_neg


def _margin_diff(margins: np.ndarray, classes_: np.ndarray) -> np.ndarray:
    """
    For decision_function outputs:
      - Binary case: margins shape = (n_samples,), sign indicates positive class.
      - Multi-class (OvR): margins shape = (n_samples, n_classes), where each column is
        the margin for "this class vs rest". Return margin(+1) - margin(-1) if both exist.
    """
    if margins.ndim == 1:
        # Binary. We need to know which class is treated as "positive" column.
        # sklearn encodes classes_ sorted ascending; in binary case, the decision_function
        # is for class classes_[1] vs classes_[0]. If classes_ = [-1, 1], positive is +1.
        # If classes_ = [0, 1], positive is +1; if classes_ = [-1, 0], positive is 0.
        # We want score wrt +/- classes. If +1 not present, return margins with best-effort sign.
        if classes_.size == 2:
            pos_class = classes_[1]  # sklearn convention
            if pos_class == 1:
                return margins
            elif pos_class == -1:
                return -margins
            else:
                # Binary without +1/-1 pair (e.g., {-1,0} or {0,1}). Map best-effort:
                return margins if pos_class in (1, 0) else -margins
        else:
            return margins  # unexpected shape; pass through
    else:
        # Multi-class OvR; pick +1 and -1 columns if they exist
        idx_pos = np.where(classes_ == 1)[0]
        idx_neg = np.where(classes_ == -1)[0]
        if idx_pos.size and idx_neg.size:
            return margins[:, idx_pos[0]] - margins[:, idx_neg[0]]
        elif idx_pos.size:
            return margins[:, idx_pos[0]]
        elif idx_neg.size:
            return -margins[:, idx_neg[0]]
        else:
            # No +/- classes; e.g., only {0}. Return zeros to be conservative.
            return np.zeros(margins.shape[0], dtype=float)


def _labels_from_scores(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Map continuous scores to {-1, 0, +1} using a dead-zone |score| <= threshold ⇒ 0.
    """
    labels = np.zeros(scores.shape[0], dtype=int)
    labels[scores > float(threshold)] = 1
    labels[scores < -float(threshold)] = -1
    return labels


# ----------------------------- Base API -----------------------------------


class MLModel:
    """Abstract base wrapper."""

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None) -> "MLModel":
        raise NotImplementedError

    def predict_scores(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_labels(self, X, threshold: float = 0.0) -> np.ndarray:
        scores = self.predict_scores(X)
        return _labels_from_scores(scores, threshold=threshold)


# ----------------------------- Concrete Models ----------------------------


@dataclass
class LRConfig:
    C: float = 1.0
    penalty: str = "l2"
    multi_class: str = "auto"  # 'auto' -> multinomial when applicable
    max_iter: int = 200
    n_jobs: int = 1
    random_state: int = 42
    class_weight: Union[str, Dict[int, float], None] = "auto"  # 'auto' uses your helper
    calibrate: bool = False
    calibrate_frac: float = 0.2  # use the LAST frac of the train fold for calibration
    # If calibrate=True, CalibratedClassifierCV(method='sigmoid', cv='prefit') is used.


class LogisticRegressionModel(MLModel):
    def __init__(self, cfg: LRConfig = LRConfig()):
        self.cfg = cfg
        self._pipe: Optional[Pipeline] = None
        self._clf: Optional[BaseEstimator] = (
            None  # final predictor (pipe or calibrator)
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, X, y, sample_weight: Optional[np.ndarray] = None
    ) -> "LogisticRegressionModel":
        X = _safe_check_X(X)
        y = _safe_check_y(y)

        cw_param, sw = _combine_sample_and_class_weights(
            y, class_weight=self.cfg.class_weight, sample_weight=sample_weight
        )

        # LR generally benefits from scaling
        base = LogisticRegression(
            C=self.cfg.C,
            penalty=self.cfg.penalty,
            multi_class=self.cfg.multi_class,
            max_iter=self.cfg.max_iter,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            class_weight=cw_param,
        )
        pipe = Pipeline(
            [("scaler", StandardScaler(with_mean=True, with_std=True)), ("lr", base)]
        )

        # Fit on full training (or train portion if calibrating)
        if self.cfg.calibrate:
            # Chronological split: tail portion for calibration
            n = X.shape[0]
            hold = max(1, int(round(self.cfg.calibrate_frac * n)))
            tr_end = n - hold
            if tr_end <= 0:
                raise ValueError("Not enough samples for calibration split.")

            pipe.fit(
                X[:tr_end],
                y[:tr_end],
                lr__sample_weight=sw[:tr_end] if sw is not None else None,
            )
            # Wrap in a calibrator using the tail as calibration set
            calibrator = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
            calibrator.fit(X[tr_end:], y[tr_end:])
            self._clf = calibrator
        else:
            pipe.fit(X, y, lr__sample_weight=sw if sw is not None else None)
            self._clf = pipe

        # Save for downstream access
        self._pipe = pipe
        # classes_ available on inner estimator; for calibrator, exposed at top level
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_scores(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted.")
        X = _safe_check_X(X)

        # Prefer probabilities (calibrated or native)
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _prob_diff(proba, classes_)
        # Fallback to decision_function (rare for LR)
        if hasattr(self._clf, "decision_function"):
            margins = self._clf.decision_function(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _margin_diff(np.asarray(margins), classes_)
        # Extreme fallback: signed label of prediction (not ideal)
        preds = self._clf.predict(X)
        return preds.astype(float)


@dataclass
class LinearSVMConfig:
    C: float = 1.0
    max_iter: int = 5000
    random_state: int = 42
    class_weight: Union[str, Dict[int, float], None] = "auto"
    calibrate: bool = False
    calibrate_frac: float = 0.2


class LinearSVMModel(MLModel):
    """
    Linear SVM (OvR for multi-class). Scaled inputs; scores from margin (+1 vs -1) unless calibrated.
    """

    def __init__(self, cfg: LinearSVMConfig = LinearSVMConfig()):
        self.cfg = cfg
        self._pipe: Optional[Pipeline] = None
        self._clf: Optional[BaseEstimator] = None  # could be calibrator
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None) -> "LinearSVMModel":
        X = _safe_check_X(X)
        y = _safe_check_y(y)

        cw_param, sw = _combine_sample_and_class_weights(
            y, class_weight=self.cfg.class_weight, sample_weight=sample_weight
        )

        base = LinearSVC(
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            random_state=self.cfg.random_state,
            class_weight=cw_param,
        )
        pipe = Pipeline(
            [("scaler", StandardScaler(with_mean=True, with_std=True)), ("svm", base)]
        )

        if self.cfg.calibrate:
            # Chronological holdout for calibration
            n = X.shape[0]
            hold = max(1, int(round(self.cfg.calibrate_frac * n)))
            tr_end = n - hold
            if tr_end <= 0:
                raise ValueError("Not enough samples for calibration split.")

            pipe.fit(
                X[:tr_end],
                y[:tr_end],
                svm__sample_weight=sw[:tr_end] if sw is not None else None,
            )
            calibrator = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
            calibrator.fit(X[tr_end:], y[tr_end:])
            self._clf = calibrator
        else:
            pipe.fit(X, y, svm__sample_weight=sw if sw is not None else None)
            self._clf = pipe

        self._pipe = pipe
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_scores(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted.")
        X = _safe_check_X(X)

        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _prob_diff(proba, classes_)
        # Uncalibrated: use decision function margins
        if hasattr(self._clf, "decision_function"):
            margins = self._clf.decision_function(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _margin_diff(np.asarray(margins), classes_)
        preds = self._clf.predict(X)
        return preds.astype(float)


@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    max_features: Union[int, float, str, None] = "sqrt"
    n_jobs: int = -1
    random_state: int = 42
    class_weight: Union[str, Dict[int, float], None] = "auto"
    calibrate: bool = False
    calibrate_frac: float = 0.2


class RandomForestModel(MLModel):
    def __init__(self, cfg: RFConfig = RFConfig()):
        self.cfg = cfg
        self._clf: Optional[BaseEstimator] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self, X, y, sample_weight: Optional[np.ndarray] = None
    ) -> "RandomForestModel":
        X = _safe_check_X(X)
        y = _safe_check_y(y)

        cw_param, sw = _combine_sample_and_class_weights(
            y, class_weight=self.cfg.class_weight, sample_weight=sample_weight
        )

        rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            max_features=self.cfg.max_features,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            class_weight=cw_param,
        )

        if self.cfg.calibrate:
            # Chronological holdout for calibration
            n = X.shape[0]
            hold = max(1, int(round(self.cfg.calibrate_frac * n)))
            tr_end = n - hold
            if tr_end <= 0:
                raise ValueError("Not enough samples for calibration split.")

            rf.fit(
                X[:tr_end],
                y[:tr_end],
                sample_weight=sw[:tr_end] if sw is not None else None,
            )
            calibrator = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
            calibrator.fit(X[tr_end:], y[tr_end:])
            self._clf = calibrator
        else:
            rf.fit(X, y, sample_weight=sw if sw is not None else None)
            self._clf = rf

        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict_scores(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted.")
        X = _safe_check_X(X)

        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)
            classes_ = getattr(self._clf, "classes_", self.classes_)
            return _prob_diff(proba, classes_)
        # Fallback: mean vote margin via decision_path not exposed -> use predicted class {-1,0,1}
        preds = self._clf.predict(X)
        return preds.astype(float)


# ----------------------------- Factory & typing ---------------------------

ModelLike = Union[LogisticRegressionModel, LinearSVMModel, RandomForestModel]


def make_model(name: str, **kwargs) -> ModelLike:
    """
    Factory for concise config from strings.

    Examples:
        model = make_model("lr", C=0.5, calibrate=True)
        model = make_model("svm", C=2.0, calibrate=False)
        model = make_model("rf", n_estimators=500, calibrate=True, calibrate_frac=0.15)
    """
    name = name.strip().lower()
    if name in ("lr", "logreg", "logistic", "logistic_regression"):
        cfg = LRConfig(**kwargs)
        return LogisticRegressionModel(cfg)
    if name in ("svm", "linear_svm", "linsvm", "linearsvc"):
        cfg = LinearSVMConfig(**kwargs)
        return LinearSVMModel(cfg)
    if name in ("rf", "randforest", "random_forest", "randomforest"):
        cfg = RFConfig(**kwargs)
        return RandomForestModel(cfg)
    raise ValueError(f"Unknown model name: {name}")
