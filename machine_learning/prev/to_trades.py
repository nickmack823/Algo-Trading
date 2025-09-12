# -*- coding: utf-8 -*-
"""
to_trades.py
------------
Score/position → TradePlan helpers tailored to your Backtester + StrategyCore.

Why this file exists
--------------------
Your Backtester expects strategy.generate_trade_plan(current_index, current_position, balance, quote_to_usd_rate)
to return a *list of TradePlan objects* for that single bar. It also handles stop-loss / take-profit exits
via PositionManager.update(...) on each open trade **before** consuming any explicit EXIT plans.

This module gives you two layers:
1) scores_to_positions(scores, thresholds, rules)    → a stable {-1,0,+1} position series with hysteresis/min-hold/cooldown
2) MLSignalPlanner(...).plans_at_index(i, ...)      → emit EXIT and/or ENTRY TradePlan(s) for bar i that your Backtester will consume

Design choices that match your runtime loop
-------------------------------------------
- We emit an EXIT plan *first* when a flip is detected (e.g., +1 → -1); by default we **do not** also emit a same-bar ENTRY,
  because your Backtester updates self.position only *after* processing plans, so a same-bar entry would be ignored.
  (You can enable same_bar_flip_entry=True to attempt an immediate re-entry on flips, but the Backtester will still block it.)
- ENTRY plans include direction, units, entry_price, stop_loss, optional take_profit, and a PositionManager configured for
  breakeven/trailing logic. The Backtester uses these fields to open trades and then manages SL/TP over time.

Minimal usage inside an ML Strategy
-----------------------------------
# in your ML strategy.prepare_data(...):
scores = model_scores_series  # index-aligned with your OHLCV df
positions = scores_to_positions(scores, thresholds=Thresholds(0.2,-0.2,hysteresis=0.05), rules=ExecutionRules(min_hold=2, cooldown=1))

self.planner = MLSignalPlanner(
    forex_pair=self.FOREX_PAIR,
    df=self.data_with_indicators,  # must include 'Close' and (optionally) an ATR column if you select atr-based risk
    positions=positions,
    risk=RiskParams(risk_pct=0.01, atr_col="ATR_14", atr_multiplier=1.5, tp_multiple=None, breakeven_trigger_pips=None, trail_start_pips=None, trail_distance_pips=None),
    same_bar_flip_entry=False,    # safer default with current Backtester
    source="ML",
)

# in your ML strategy.generate_trade_plan(...):
return self.planner.plans_at_index(current_index, current_position, balance, quote_to_usd_rate)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Prefer absolute import when inside your repo; fall back to local for notebooks/tests.
try:
    from scripts.strategies import strategy_core as sc
except Exception:  # pragma: no cover
    import strategy_core as sc  # type: ignore


# ------------------------------ Thresholding ----------------------------------


@dataclass(frozen=True)
class Thresholds:
    long_enter: float = 0.2
    short_enter: float = -0.2
    # Optional explicit exits; if None, use enter +/- hysteresis
    long_exit: Optional[float] = None
    short_exit: Optional[float] = None
    hysteresis: float = 0.05  # applied if exit thresholds are None


@dataclass(frozen=True)
class ExecutionRules:
    min_hold: int = 1  # bars to hold after entry before eligible to exit/flip
    cooldown: int = 0  # bars to stay flat after an exit
    max_hold: Optional[int] = None  # optional time stop (bars)


def scores_to_positions(
    scores: pd.Series,
    *,
    thresholds: Thresholds = Thresholds(),
    rules: ExecutionRules = ExecutionRules(),
) -> pd.Series:
    """
    Convert a continuous score series (e.g., P(+1)-P(-1)) into a stable {-1,0,+1} position series.
    - Hysteresis exits prevent whipsaw: long_exit defaults to long_enter-hysteresis; short_exit to short_enter+hysteresis
    - min_hold, cooldown, max_hold control churn and timing behavior.

    Returns
    -------
    pd.Series in {-1,0,+1}, same index as `scores`.
    """
    if not isinstance(scores, pd.Series):
        scores = pd.Series(scores)
    scores = scores.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    idx = scores.index
    assert idx.is_monotonic_increasing, "scores index must be sorted ascending"

    le = thresholds.long_enter
    se = thresholds.short_enter
    lx = (
        thresholds.long_exit
        if thresholds.long_exit is not None
        else le - abs(thresholds.hysteresis)
    )
    sx = (
        thresholds.short_exit
        if thresholds.short_exit is not None
        else se + abs(thresholds.hysteresis)
    )

    pos = np.zeros(len(scores), dtype=int)
    hold = 0
    cool = 0

    for i, s in enumerate(scores.values):
        if cool > 0:
            pos[i] = 0
            cool -= 1
            continue

        cur = pos[i - 1] if i > 0 else 0

        if cur == 0:
            if s >= le:
                cur = 1
                hold = 0
            elif s <= se:
                cur = -1
                hold = 0
            else:
                cur = 0
        else:
            hold += 1
            can_exit = hold >= max(1, rules.min_hold)

            if rules.max_hold is not None and hold >= rules.max_hold:
                cur = 0
                hold = 0
                cool = rules.cooldown
            elif cur == 1:
                if can_exit and s <= lx:
                    cur = 0
                    hold = 0
                    cool = rules.cooldown
                elif can_exit and s <= se:
                    # flip → exit now; entry handled by planner (often next bar)
                    cur = 0
                    hold = 0
                    cool = rules.cooldown
            else:  # cur == -1
                if can_exit and s >= sx:
                    cur = 0
                    hold = 0
                    cool = rules.cooldown
                elif can_exit and s >= le:
                    cur = 0
                    hold = 0
                    cool = rules.cooldown

        pos[i] = cur

    return pd.Series(pos, index=idx, name="position")


# ------------------------------ Risk / Sizing ---------------------------------


@dataclass(frozen=True)
class RiskParams:
    """
    Risk & exit configuration used for ENTRY TradePlans.
    - If `atr_col` is provided, we size stops & units from ATR × atr_multiplier.
    - Otherwise, we require `fixed_sl_pips` and use that for stops & sizing.
    - Optional TP and PM (breakeven/trailing) are in pips.
    """

    risk_pct: float = 0.01
    atr_col: Optional[str] = None
    atr_multiplier: float = 1.5  # when using ATR
    fixed_sl_pips: Optional[float] = None  # when not using ATR

    # Optional take-profit as multiple of stop distance (e.g., 1.5R)
    tp_multiple: Optional[float] = None

    # PositionManager config (all in pips relative to entry)
    breakeven_trigger_pips: Optional[float] = None
    trail_start_pips: Optional[float] = None
    trail_distance_pips: Optional[float] = None


def _pip_size(forex_pair: str) -> float:
    return sc.PositionCalculator.calculate_pip_size(forex_pair)


def _stop_distance_pips(
    risk: RiskParams, df: pd.DataFrame, i: int, forex_pair: str
) -> float:
    if risk.atr_col is not None:
        if risk.atr_col not in df.columns:
            raise KeyError(f"ATR column '{risk.atr_col}' not found in df.")
        atr_val = float(df[risk.atr_col].iloc[i])
        # Convert ATR value (in price units) to pips, then multiply
        atr_pips = sc.PositionCalculator.calculate_atr_pips(forex_pair, atr_val)
        return float(atr_pips * max(0.0, risk.atr_multiplier))
    if risk.fixed_sl_pips is None:
        raise ValueError("Provide either atr_col or fixed_sl_pips in RiskParams.")
    return float(risk.fixed_sl_pips)


def _units_for_entry(
    forex_pair: str,
    balance: float,
    risk: RiskParams,
    entry_price: float,
    df: pd.DataFrame,
    i: int,
    quote_to_usd_rate: Optional[float],
) -> int:
    # When ATR is used we can call the repo helper directly for units
    if risk.atr_col is not None:
        stop_pips = _stop_distance_pips(
            risk, df, i, forex_pair
        )  # respects risk.atr_multiplier
        dollars_per_pip_per_lot = sc.PositionCalculator.calculate_current_pip_value(
            forex_pair, 100_000, float(entry_price), quote_to_usd_rate
        )
        target_pip_value = sc.PositionCalculator.calculate_target_pip_value(
            int(balance), float(risk.risk_pct), float(stop_pips)
        )
        units = target_pip_value * 100_000 / dollars_per_pip_per_lot
        return int(round(units))

    # Otherwise, derive units from desired $ risk per trade and fixed pip stop
    dollars_per_pip_per_lot = sc.PositionCalculator.calculate_current_pip_value(
        forex_pair, 100_000, float(entry_price), quote_to_usd_rate
    )
    stop_pips = _stop_distance_pips(risk, df, i, forex_pair)
    target_pip_value = sc.PositionCalculator.calculate_target_pip_value(
        int(balance), float(risk.risk_pct), float(stop_pips)
    )
    units = target_pip_value * 100_000 / dollars_per_pip_per_lot
    return int(round(units))


def _stop_price_from_pips(
    forex_pair: str, direction: str, entry_price: float, stop_pips: float
) -> float:
    pipsz = _pip_size(forex_pair)
    if direction.upper() == "BUY":
        return float(entry_price - stop_pips * pipsz)
    else:
        return float(entry_price + stop_pips * pipsz)


def _tp_price_from_multiple(
    forex_pair: str,
    direction: str,
    entry_price: float,
    stop_pips: float,
    multiple: float,
) -> float:
    pipsz = _pip_size(forex_pair)
    if direction.upper() == "BUY":
        return float(entry_price + (stop_pips * multiple) * pipsz)
    else:
        return float(entry_price - (stop_pips * multiple) * pipsz)


# ------------------------------ Planner ---------------------------------------


class MLSignalPlanner:
    """
    Holds a precomputed position series and OHLCV(+ATR) DataFrame.
    Emits TradePlan(s) for a given bar index consistent with your Backtester.

    Typical flow inside a Strategy subclass:
        - compute scores → positions in prepare_data()
        - initialize planner with positions & risk config
        - in generate_trade_plan(i, ...), delegate to planner.plans_at_index(...)
    """

    def __init__(
        self,
        *,
        forex_pair: str,
        df: pd.DataFrame,
        positions: pd.Series,
        risk: RiskParams,
        source: str = "ML",
        same_bar_flip_entry: bool = False,
    ):
        assert "Close" in df.columns, "df must contain a 'Close' column"
        assert positions.index.equals(df.index), "positions index must equal df index"
        assert (
            positions.is_monotonic_increasing
        ), "positions index must be sorted ascending"
        self.forex_pair = forex_pair.replace("/", "")
        self.df = df
        self.pos = positions.astype(int).clip(-1, 1)
        self.risk = risk
        self.source = source
        self.same_bar_flip_entry = same_bar_flip_entry

    def _transition_at(self, i: int) -> int:
        """Return position transition at i: pos[i] - pos[i-1] (prev pos=0 for i==0)."""
        cur = int(self.pos.iloc[i])
        prev = int(self.pos.iloc[i - 1]) if i > 0 else 0
        return cur - prev

    def plans_at_index(
        self,
        i: int,
        current_position: int,
        balance: float,
        quote_to_usd_rate: Optional[float],
    ) -> List[sc.TradePlan]:
        """
        Emit a list of TradePlan(s) for bar i.
        Order: EXIT (if needed) then ENTRY (if allowed by same_bar_flip_entry and current_position).
        """
        plans: List[sc.TradePlan] = []
        transition = self._transition_at(i)

        # Flip or exit-to-flat ⇒ emit EXIT first (Backtester will close at current price w/ slippage)
        if (
            (transition < 0 and int(self.pos.iloc[i - 1] if i > 0 else 0) == 1)
            or (transition > 0 and int(self.pos.iloc[i - 1] if i > 0 else 0) == -1)
            or (transition == -1 and int(self.pos.iloc[i]) == 0)
            or (transition == 1 and int(self.pos.iloc[i]) == 0)
        ):
            plans.append(
                sc.TradePlan(
                    strategy=None,
                    direction="",  # ignored for EXIT
                    entry_price=float("nan"),
                    stop_loss=float("nan"),
                    tag="EXIT",
                    units=0,
                    source=self.source,
                )
            )

        # ENTRY on 0→+1 or 0→−1 (or on flip if same_bar_flip_entry=True)
        want_long = transition == 1 or (
            self.same_bar_flip_entry and transition < 0 and self.pos.iloc[i] == 1
        )
        want_short = transition == -1 or (
            self.same_bar_flip_entry and transition > 0 and self.pos.iloc[i] == -1
        )

        # Backtester will only open if currently flat; we still compute the plan.
        if want_long or want_short:
            direction = "BUY" if want_long else "SELL"
            close = float(self.df["Close"].iloc[i])
            stop_pips = float(
                _stop_distance_pips(self.risk, self.df, i, self.forex_pair)
            )
            stop_price = _stop_price_from_pips(
                self.forex_pair, direction, close, stop_pips
            )

            units = _units_for_entry(
                self.forex_pair,
                balance,
                self.risk,
                close,
                self.df,
                i,
                quote_to_usd_rate,
            )

            take_profit = None
            if self.risk.tp_multiple is not None:
                take_profit = _tp_price_from_multiple(
                    self.forex_pair,
                    direction,
                    close,
                    stop_pips,
                    float(self.risk.tp_multiple),
                )

            # Build PositionManager config in *price* units from pips settings
            pm_cfg = {}
            if self.risk.breakeven_trigger_pips is not None:
                pm_cfg["breakeven_trigger"] = float(
                    self.risk.breakeven_trigger_pips
                ) * _pip_size(self.forex_pair)
            if self.risk.trail_start_pips is not None:
                pm_cfg["trail_start"] = float(self.risk.trail_start_pips) * _pip_size(
                    self.forex_pair
                )
            if self.risk.trail_distance_pips is not None:
                pm_cfg["trail_distance"] = float(
                    self.risk.trail_distance_pips
                ) * _pip_size(self.forex_pair)

            pm = sc.PositionManager(
                entry_price=close,
                initial_stop_loss=stop_price,
                direction=direction,
                config=pm_cfg or None,
                take_profit=take_profit,
            )

            plans.append(
                sc.TradePlan(
                    strategy=None,  # Strategy subclass can set self.NAME before returning
                    direction=direction,  # 'BUY' or 'SELL'
                    entry_price=close,  # current close price as market entry
                    stop_loss=stop_price,
                    take_profit=take_profit,
                    units=units,
                    risk_pct=float(self.risk.risk_pct),
                    tag="",
                    position_manager=pm,
                    source=self.source,
                )
            )

        return plans
