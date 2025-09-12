# strategies/paper_ml_2021.py
from scripts.strategies.machine_learning_base import (
    MLClassificationStrategy,
)  # your generic
from scripts.strategies.strategy_core import register_strategy

_STRATEGY_KEY = "Mabrouk2021"


@register_strategy(
    key=_STRATEGY_KEY,
    required_kwargs=[
        "forex_pair",
        "timeframe",
        "indicator_registry",
    ],  # feature_specs optional
    description="Mabrouk et al. 2021 preset: binary next-bar, classic indicators, LR/LinearSVM/RF.",
)
class MabroukML2021Strategy(MLClassificationStrategy):
    NAME = _STRATEGY_KEY
    DESCRIPTION = "Paper-specific preset based on Mabrouk et al. 2021 on top of the generic MLClassificationStrategy."

    def __init__(
        self,
        *,
        forex_pair,
        timeframe,
        indicator_registry,
        feature_specs=None,
        label_horizon=1,
        label_threshold=0.0,
        fee_bps=0.0,
        model_key="linearsvm",
        model_params=None,
        thresholds=None,
        rules=None,
        risk=None,
        same_bar_flip_entry=False,
        split=None,
    ):
        super().__init__(
            forex_pair=forex_pair,
            timeframe=timeframe,
            indicator_registry=indicator_registry,
            feature_specs=(list(feature_specs) if feature_specs else []),
            label_horizon=label_horizon,
            label_threshold=label_threshold,
            fee_bps=fee_bps,
            model_key=model_key,  # "lr" | "linearsvm" | "rf"
            model_params=model_params or {},
            thresholds=thresholds
            or {"long_enter": 0.15, "short_enter": -0.15, "hysteresis": 0.05},
            rules=rules or {"min_hold": 1, "cooldown": 0},
            risk=risk
            or {
                "risk_pct": 0.01,
                "atr_col": "ATR_14",
                "atr_multiplier": 1.5,
                "tp_multiple": None,
            },
            same_bar_flip_entry=same_bar_flip_entry,
            split=split,  # NEW ← forward to base
        )
