from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.config import (
    DEFAULT_ATR_SL_MULTIPLIER,
    DEFAULT_TP_MULTIPLIER,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT_LONG,
    EXIT_SHORT,
    NO_SIGNAL,
    RISK_PER_TRADE,
)
from scripts.indicators import indicator_configs as icfg
from scripts.indicators.indicator_configs import IndicatorConfig
from scripts.strategies.strategy_core import (
    BaseStrategy,
    PositionCalculator,
    PositionManager,
    TradePlan,
    register_strategy,
)
from scripts.utilities import convert_np_to_pd


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# Ensure these are available from the file this class is inserted into
# BaseStrategy, register_strategy, TradePlan, PositionManager, convert_np_to_pd, _compute_atr


@register_strategy(
    "CANDLESTICK",
    [
        "atr",
        "baseline",
        "patterns",
        "forex_pair",
        "timeframe",
    ],
    description=(
        "Candlestick-driven strategy with baseline, volatility, and optional trend filters. "
        "Uses TA-Lib candlestick pattern functions with configurable whitelist and thresholds."
    ),
)
class CandlestickFilteredStrategy(BaseStrategy):
    """
    A flexible candlestick strategy.

    Entry: A candlestick in the whitelist fires and passes filters (baseline bias, min body/ATR, trend bias).
    Exit: Optional opposite signal or baseline cross; default uses ATR-based stops/TPs via TradePlan.
    """

    # ──────────────────────────────────────────────────────────────────────────────
    # CANDLESTICK STRATEGY — WHAT IT DOES (STEP-BY-STEP)
    # ──────────────────────────────────────────────────────────────────────────────
    # Inputs (config):
    #   • atr:            Indicator producing ATR values (volatility).
    #   • baseline:       Indicator producing a price baseline (e.g., MA).
    #   • patterns:       Whitelisted TA-Lib candlestick names to evaluate.
    #   • trend_filter:   (optional) Indicator(s) for directional bias permission.
    #   • min_abs_score:  Minimum |pattern score| to treat a pattern as “present”.
    #   • min_body_atr:   Min body size, as a multiple of ATR (filters tiny candles).
    #   • align_with_baseline: Require LONG above baseline / SHORT below baseline.
    #   • cooldown_bars:  Bars to wait after an entry before another entry.
    #   • opposite_exit:  If True, exit when an opposite pattern appears.
    #   • forex_pair, timeframe: Routing/metadata for backtests/live.
    #
    # Data prep (once, then reused):
    #   1) Compute ATR  → df['ATR'].
    #   2) Compute Baseline  → df['Baseline'] and flags: df['AboveBaseline']/['BelowBaseline'].
    #   3) If trend_filter provided, compute raw columns  → df['TrendRaw_*'].
    #   4) For each whitelisted candlestick pattern:
    #        • Compute score series  → df[f"PAT_{name}"] (typically in −100..+100).
    #   5) Precompute candle body size  → df['Body'] = |Close − Open|.
    #
    # Per bar (signal generation logic):
    #   1) Enforce cooldown:
    #      • If bars_since_last_entry < cooldown_bars → NO_SIGNAL.
    #   2) Basic guards:
    #      • If ATR is NaN/≤0 → NO_SIGNAL.
    #      • If Body < (min_body_atr * ATR) → NO_SIGNAL.
    #   3) Trend permission (if trend_filter used):
    #      • trend_bias = sign(mean(last TrendRaw_* values))  → +1 / 0 / −1.
    #   4) Evaluate candlestick hits on this bar:
    #      • long_hits  = patterns with score ≥ +min_abs_score.
    #      • short_hits = patterns with score ≤ −min_abs_score.
    #   5) Directional permissions:
    #      • baseline_ok("LONG")  = (AboveBaseline == 1) if align_with_baseline else True
    #      • baseline_ok("SHORT") = (BelowBaseline == 1) if align_with_baseline else True
    #      • trend_ok("LONG")     = (trend_bias > 0) if trend provided else True
    #      • trend_ok("SHORT")    = (trend_bias < 0) if trend provided else True
    #   6) Entry rules:
    #      • If any long_hits AND baseline_ok(LONG) AND trend_ok(LONG) AND current_position ≤ 0:
    #          → ENTER_LONG and record _last_entry_bar.
    #      • Elif any short_hits AND baseline_ok(SHORT) AND trend_ok(SHORT) AND current_position ≥ 0:
    #          → ENTER_SHORT and record _last_entry_bar.
    #   7) Exit rules (only if in a position):
    #      • If current_position > 0 (LONG):
    #           - If opposite_exit and short_hits exist → EXIT_LONG
    #           - Else if Close < Baseline → EXIT_LONG
    #      • Elif current_position < 0 (SHORT):
    #           - If opposite_exit and long_hits exist → EXIT_SHORT
    #           - Else if Close > Baseline → EXIT_SHORT
    #      • Else (flat): if no entry triggered → NO_SIGNAL
    #
    # Trade planning when an entry fires:
    #   • entry_price = Close (of the signal bar)
    #   • atr_used    = df['ATR'] (compute on the fly if missing)
    #   • Two legs are created (same direction):
    #       1) “TP1”    – risk = RISK_PER_TRADE, SL = entry ± ATR*DEFAULT_ATR_SL_MULTIPLIER,
    #                     TP = entry ± ATR*DEFAULT_TP_MULTIPLIER (± depends on BUY/SELL).
    #       2) “Runner” – risk = RISK_PER_TRADE, SL as above, no fixed TP;
    #                     PositionManager enables:
    #                       • move to breakeven after +1×ATR,
    #                       • start trailing after +2×ATR,
    #                       • trail distance = 1.5×ATR.
    #
    # Trade planning when an exit fires:
    #   • Emit an EXIT TradePlan (opposite direction) at the current Close (tag="EXIT").
    #
    # Notes:
    #   • Only the candlestick itself can trigger entries; baseline/trend act as filters,
    #     not as triggers.
    #   • Cooldown prevents rapid re-entries.
    #   • All sizes and thresholds are configurable via the constructor parameters.
    # ──────────────────────────────────────────────────────────────────────────────

    NAME = "CANDLESTICK"
    DESCRIPTION = "Candlestick-driven strategy with baseline and volatility filters; configurable pattern set."

    def __init__(
        self,
        atr: IndicatorConfig,
        baseline: IndicatorConfig,
        patterns: List[str],
        forex_pair: str,
        timeframe: str,
        # --- Filters / thresholds ---
        min_abs_score: int = 50,  # TA-Lib scores are typically +/- 100
        min_body_atr: float = 0.3,  # candle body must be >= this * ATR
        align_with_baseline: bool = True,
        trend_filter: IndicatorConfig | None = None,
        cooldown_bars: int = 2,  # prevent rapid re-entries
        opposite_exit: bool = True,  # exit on opposite signal
    ):
        self.atr_cfg = atr
        self.baseline_cfg = baseline
        self.trend_cfg = trend_filter
        self.pattern_names = patterns

        self.min_abs_score = int(min_abs_score)
        self.min_body_atr = float(min_body_atr)
        self.align_with_baseline = bool(align_with_baseline)
        self.cooldown_bars = int(cooldown_bars)
        self.opposite_exit = bool(opposite_exit)

        # Backtest state
        self._last_entry_bar: int | None = None
        self.data_with_indicators: pd.DataFrame | None = None
        self.indicator_cache_dicts_to_insert: list[dict] = []

        parameters = {
            "atr": f"{atr['name']}_{atr['parameters']}",
            "baseline": f"{baseline['name']}_{baseline['parameters']}",
            "trend": (
                None
                if trend_filter is None
                else f"{trend_filter['name']}_{trend_filter['parameters']}"
            ),
            "patterns": ",".join(patterns),
            "min_abs_score": self.min_abs_score,
            "min_body_atr": self.min_body_atr,
            "align_with_baseline": self.align_with_baseline,
            "cooldown_bars": self.cooldown_bars,
            "opposite_exit": self.opposite_exit,
        }

        super().__init__(
            forex_pair=forex_pair, parameters=parameters, timeframe=timeframe
        )

    # ------------------------ Data prep ------------------------
    def prepare_data(self, historical_data: pd.DataFrame, use_cache: bool = True):
        df = historical_data.copy()

        # ATR
        atr_out = self._apply_indicator(self.atr_cfg, df)
        if isinstance(atr_out, pd.DataFrame):
            df["ATR"] = atr_out.iloc[:, 0].rename("ATR")
        else:
            df["ATR"] = atr_out.rename("ATR")

        # Baseline
        base_out = self._apply_indicator(self.baseline_cfg, df)
        # Normalize to a Series named 'Baseline'
        if isinstance(base_out, pd.DataFrame):
            # take first column as primary baseline
            df["Baseline"] = base_out.iloc[:, 0].rename("Baseline")
        else:
            df["Baseline"] = base_out.rename("Baseline")

        # Optional Trend filter (convert its signals to trend bias if provided)
        if self.trend_cfg is not None:
            trend_raw = self._apply_indicator(self.trend_cfg, df)
            # Keep raw columns with a prefix for later signal interpretation
            if isinstance(trend_raw, pd.DataFrame):
                for i, c in enumerate(trend_raw.columns):
                    df[f"TrendRaw_{i}"] = trend_raw[c]
            else:
                df["TrendRaw_0"] = trend_raw

        # Candlesticks: compute selected TA-Lib pattern columns
        # Each returns an int (usually -100..+100). Nonzero => pattern present, sign => direction.
        # Map pattern name -> function from indicator_configs.ta_lib_candlestick
        name_to_fn: Dict[str, callable] = {
            c["name"]: c["function"] for c in icfg.ta_lib_candlestick
        }
        for name in self.pattern_names:
            if name not in name_to_fn:
                raise ValueError(f"Unknown candlestick pattern: {name}")
            series = name_to_fn[name](df)
            df[f"PAT_{name}"] = series.astype(float)

        # Precompute helpers
        body = (df["Close"] - df["Open"]).abs().rename("Body")
        df["Body"] = body
        df["AboveBaseline"] = (df["Close"] > df["Baseline"]).astype(int)
        df["BelowBaseline"] = (df["Close"] < df["Baseline"]).astype(int)

        self.data_with_indicators = df

    def get_cache_jobs(self):
        # Strategy currently computes indicators on the fly; integrate DB cache here if desired
        return self.indicator_cache_dicts_to_insert

    # ------------------------ Core signal logic ------------------------
    def _generate_signals(
        self, data: pd.DataFrame, current_position: int, debug: bool = False
    ):
        """Return ([signals], source_tag) based on the most recent bar in `data`."""
        signals: List[str] = []
        src = "CandleEntry"

        row = data.iloc[-1]
        i = data.index[-1]

        # Cooldown
        if self._last_entry_bar is not None:
            bars_since = len(data) - 1 - self._last_entry_bar
            if bars_since < self.cooldown_bars:
                return [NO_SIGNAL], src

        # Filters shared by both directions
        atr_val = float(row.get("ATR", np.nan))
        if not np.isfinite(atr_val) or atr_val <= 0:
            return [NO_SIGNAL], src

        if float(row["Body"]) < self.min_body_atr * atr_val:
            return [NO_SIGNAL], src

        # Trend permission (if provided)
        trend_bias = None
        if any(c.startswith("TrendRaw_") for c in data.columns):
            # very lightweight: use last value slope/pos vs 0 when Series-like; otherwise try signal fn if provided
            trcols = [c for c in data.columns if c.startswith("TrendRaw_")]
            last_vals = [data[c].iloc[-1] for c in trcols]
            mean_val = np.nanmean(last_vals)
            if np.isfinite(mean_val):
                trend_bias = 1 if mean_val > 0 else (-1 if mean_val < 0 else 0)

        # Evaluate candlestick whitelist on the last bar
        long_hits = []
        short_hits = []
        for name in self.pattern_names:
            v = float(row.get(f"PAT_{name}", 0.0))
            if abs(v) >= self.min_abs_score:
                if v > 0:
                    long_hits.append(name)
                elif v < 0:
                    short_hits.append(name)

        # Build directional candidates that pass baseline alignment
        def baseline_ok(direction: str) -> bool:
            if not self.align_with_baseline:
                return True
            return (
                bool(row["AboveBaseline"])
                if direction == "LONG"
                else bool(row["BelowBaseline"])
            )

        # Trend permission check (if present)
        def trend_ok(direction: str) -> bool:
            if trend_bias is None:
                return True
            return (trend_bias > 0) if direction == "LONG" else (trend_bias < 0)

        enter_long = len(long_hits) > 0 and baseline_ok("LONG") and trend_ok("LONG")
        enter_short = len(short_hits) > 0 and baseline_ok("SHORT") and trend_ok("SHORT")

        if enter_long and current_position <= 0:
            signals.append(ENTER_LONG)
            self._last_entry_bar = len(data) - 1
        elif enter_short and current_position >= 0:
            signals.append(ENTER_SHORT)
            self._last_entry_bar = len(data) - 1

        # Exit logic: opposite pattern or baseline cross if in a trade
        if current_position > 0:  # long
            if self.opposite_exit and len(short_hits) > 0:
                signals.append(EXIT_LONG)
            elif row["Close"] < row["Baseline"]:
                signals.append(EXIT_LONG)
        elif current_position < 0:  # short
            if self.opposite_exit and len(long_hits) > 0:
                signals.append(EXIT_SHORT)
            elif row["Close"] > row["Baseline"]:
                signals.append(EXIT_SHORT)

        if not signals:
            signals = [NO_SIGNAL]
        return signals, src

    # ------------------------ Trade plan ------------------------
    def generate_trade_plan(
        self,
        current_index: int,
        current_position: int,
        balance: float,
        quote_to_usd_rate: float,
    ) -> list[TradePlan]:
        plans: List[TradePlan] = []
        data = self._get_slice(current_index)
        row = data.iloc[-1]
        signals, source = self._generate_signals(data, current_position)

        if ENTER_LONG in signals or ENTER_SHORT in signals:
            signal = ENTER_LONG if ENTER_LONG in signals else ENTER_SHORT
            atr = (
                float(row["ATR"])
                if np.isfinite(row.get("ATR", np.nan))
                else float(_compute_atr(data).iloc[-1])
            )
            entry = float(row["Close"])
            direction = "BUY" if signal == ENTER_LONG else "SELL"

            # Two-part position: TP leg + runner with trailing
            entries = [
                (RISK_PER_TRADE, DEFAULT_TP_MULTIPLIER, "TP1"),
                (RISK_PER_TRADE, None, "Runner"),
            ]
            for risk_pct, tp_mult, tag in entries:
                sl = (
                    entry - atr * DEFAULT_ATR_SL_MULTIPLIER
                    if direction == "BUY"
                    else entry + atr * DEFAULT_ATR_SL_MULTIPLIER
                )
                tp = (
                    entry + atr * DEFAULT_TP_MULTIPLIER
                    if (direction == "BUY" and tp_mult)
                    else (
                        entry - atr * DEFAULT_TP_MULTIPLIER
                        if (direction == "SELL" and tp_mult)
                        else None
                    )
                )
                units = PositionCalculator.calculate_trade_units(
                    self.FOREX_PAIR, balance, risk_pct, entry, atr, quote_to_usd_rate
                )
                plan = TradePlan(
                    strategy=self.NAME,
                    direction=direction,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    units=units,
                    risk_pct=risk_pct,
                    tag=tag,
                    source=source,
                )
                runner_cfg = {}
                if tag == "Runner":
                    runner_cfg = {
                        "breakeven_trigger": atr,
                        "trail_start": 2 * atr,
                        "trail_distance": 1.5 * atr,
                    }
                plan.position_manager = PositionManager(
                    entry_price=entry,
                    initial_stop_loss=sl,
                    direction=direction,
                    config=runner_cfg,
                    take_profit=tp,
                )
                plans.append(plan)

        if EXIT_LONG in signals or EXIT_SHORT in signals:
            direction = "SELL" if EXIT_LONG in signals else "BUY"
            plans.append(
                TradePlan(
                    strategy=self.NAME,
                    direction=direction,
                    entry_price=float(row["Close"]),
                    stop_loss=None,
                    tag="EXIT",
                    source=source,
                )
            )
        return plans

    # ------------------------ Helpers ------------------------
    def _apply_indicator(self, cfg: IndicatorConfig, df: pd.DataFrame):
        fn = cfg["function"]
        params = cfg.get("parameters", {}) or {}
        out = fn(df, **params) if params else fn(df)
        return convert_np_to_pd(out)
