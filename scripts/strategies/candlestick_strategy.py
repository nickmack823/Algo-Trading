from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from scripts.config import (  # signals & constants; optional trend signal semantics (used if trend_filter has a signal_function)
    BEARISH_SIGNAL,
    BEARISH_TREND,
    BULLISH_SIGNAL,
    BULLISH_TREND,
    DEFAULT_ATR_SL_MULTIPLIER,
    DEFAULT_TP_MULTIPLIER,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT_LONG,
    EXIT_SHORT,
    NO_SIGNAL,
    RISK_PER_TRADE,
)
from scripts.data.sql import IndicatorCacheSQLHelper
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
    """Simple ATR (RMA) used only as a fallback if precomputed ATR is missing."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


@register_strategy(
    "Candlestick",
    [
        "atr",
        "baseline",
        "patterns",
        "forex_pair",
        "timeframe",
    ],
    description=(
        "Candlestick-driven strategy with baseline/ATR filters, optional trend filter, cache parity, "
        "cooldown (entries only), and two-leg ATR-based trade planning."
    ),
)
class CandlestickFilteredStrategy(BaseStrategy):
    """
    Entries: TA-Lib candlestick whitelist fires, filtered by baseline alignment, min body/ATR, and optional trend bias.
    Exits: (1) opposite-pattern exit (if enabled) or (2) baseline cross exit.
    Plans: Two-leg entries (TP1 + Runner with BE@+1ATR, trail start +2ATR, distance 1.5ATR).
    """

    NAME = "Candlestick"
    DESCRIPTION = "Candlestick-driven strategy with baseline and volatility filters; configurable pattern set."

    def __init__(
        self,
        atr: IndicatorConfig,
        baseline: IndicatorConfig,
        patterns: List[str],
        forex_pair: str,
        timeframe: str,
        # --- Filters / thresholds ---
        min_abs_score: int = 50,  # TA-Lib scores are typically +/-100
        min_body_atr: float = 0.3,  # |Close-Open| must be >= this * ATR
        align_with_baseline: bool = True,
        trend_filter: IndicatorConfig | None = None,
        cooldown_bars: int = 2,  # prevents rapid re-entries (does NOT block exits)
        opposite_exit: bool = True,  # exit when opposite pattern appears
    ):
        # Indicator configs
        self.atr_cfg = atr
        self.baseline_cfg = baseline
        self.trend_cfg = trend_filter
        self.pattern_names = list(patterns)

        # Parameters
        self.min_abs_score = int(min_abs_score)
        self.min_body_atr = float(min_body_atr)
        self.align_with_baseline = bool(align_with_baseline)
        self.cooldown_bars = int(cooldown_bars)
        self.opposite_exit = bool(opposite_exit)

        # State
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
            "patterns": ",".join(self.pattern_names),
            "min_abs_score": self.min_abs_score,
            "min_body_atr": self.min_body_atr,
            "align_with_baseline": self.align_with_baseline,
            "cooldown_bars": self.cooldown_bars,
            "opposite_exit": self.opposite_exit,
        }
        super().__init__(
            forex_pair=forex_pair, parameters=parameters, timeframe=timeframe
        )

    # ───────────────────────────── Data Prep & Cache Parity ─────────────────────────────

    def prepare_data(self, historical_data: pd.DataFrame, use_cache: bool = True):
        if use_cache:
            self.data_with_indicators = self._calculate_or_retrieve_indicators(
                historical_data
            )
        else:
            self.data_with_indicators = self._calculate_indicators_no_cache(
                historical_data
            )

    def get_cache_jobs(self) -> list[dict]:
        return self.indicator_cache_dicts_to_insert

    def _calculate_or_retrieve_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cache-aware computation for ATR, Baseline, optional Trend, and candlestick patterns.
        Mirrors NNFX cache behavior (feather + DB jobs) for parity.  :contentReference[oaicite:3]{index=3}
        """
        data = data.copy()
        cache = IndicatorCacheSQLHelper()
        start_dt, end_dt = data["Timestamp"].min(), data["Timestamp"].max()

        def retrieve_or_calculate(
            config: IndicatorConfig,
        ) -> tuple[pd.DataFrame | pd.Series, bool]:
            calculated = False
            cached = cache.fetch(
                config["name"],
                config.get("parameters", {}),
                self.FOREX_PAIR,
                self.TIMEFRAME,
                start_dt,
                end_dt,
            )
            if cached is not None:
                return cached, calculated

            raw = config["function"](data, **(config.get("parameters", {}) or {}))
            out = convert_np_to_pd(raw)
            calculated = True
            return out, calculated

        indicator_dfs: list[pd.DataFrame] = []

        # ATR / Baseline / optional Trend — use piece prefixes so we can reconstruct later. :contentReference[oaicite:4]{index=4}
        for piece_name, cfg in [
            ("ATR", self.atr_cfg),
            ("Baseline", self.baseline_cfg),
        ] + ([("Trend", self.trend_cfg)] if self.trend_cfg is not None else []):
            out, calculated = retrieve_or_calculate(cfg)

            if isinstance(out, pd.DataFrame):
                df = out.copy()
            else:
                df = pd.DataFrame({cfg["name"]: out})
            df.columns = df.columns.astype(str)

            # avoid duplicate cache jobs if same config used twice elsewhere
            already = any(
                cfg["name"] == j["indicator_name"]
                for j in self.indicator_cache_dicts_to_insert
            )
            if calculated and not already:
                meta = cache.store_in_feather(
                    df,
                    cfg["name"],
                    cfg.get("parameters", {}),
                    self.FOREX_PAIR,
                    self.TIMEFRAME,
                    start_dt,
                    end_dt,
                )
                self.indicator_cache_dicts_to_insert.append(meta)

            # add piece prefix (ATR_, Baseline_, Trend_) for reconstruction
            df = df.add_prefix(f"{piece_name}_")
            indicator_dfs.append(df)

        # Candlestick patterns — each as its own cached indicator; attach as PAT_<NAME> columns
        name_to_fn: Dict[str, Callable] = {
            c["name"]: c["function"] for c in icfg.ta_lib_candlestick
        }
        for name in self.pattern_names:
            if name not in name_to_fn:
                raise ValueError(f"Unknown candlestick pattern: {name}")
            cfg = {"name": name, "function": name_to_fn[name], "parameters": {}}

            out, calculated = retrieve_or_calculate(cfg)
            if isinstance(out, pd.DataFrame):
                # unlikely for TA-Lib patterns, but support multi-col
                raw_df = out.copy()
            else:
                raw_df = pd.DataFrame({cfg["name"]: out})
            raw_df.columns = raw_df.columns.astype(str)

            already = any(
                cfg["name"] == j["indicator_name"]
                for j in self.indicator_cache_dicts_to_insert
            )
            if calculated and not already:
                meta = cache.store_in_feather(
                    raw_df,
                    cfg["name"],
                    cfg.get("parameters", {}),
                    self.FOREX_PAIR,
                    self.TIMEFRAME,
                    start_dt,
                    end_dt,
                )
                self.indicator_cache_dicts_to_insert.append(meta)

            # rename to PAT_<NAME> (or PAT_<NAME>_<col> for multi-col)
            if raw_df.shape[1] == 1:
                raw_df.columns = [f"PAT_{name}"]
            else:
                raw_df.columns = [f"PAT_{name}_{c}" for c in raw_df.columns]
            indicator_dfs.append(raw_df)

        # Concatenate
        indicators = pd.concat(indicator_dfs, axis=1)
        full = pd.concat(
            [data.reset_index(drop=True), indicators.reset_index(drop=True)], axis=1
        )

        cache.close_connection()
        del cache
        return full

    def _calculate_indicators_no_cache(self, data: pd.DataFrame) -> pd.DataFrame:
        """No-cache path for ATR/Baseline/Trend + candlestick patterns (mirrors _calculate_or_retrieve_indicators)."""
        data = data.copy()
        out_frames: list[pd.DataFrame] = []

        # ATR / Baseline / optional Trend
        for piece_name, cfg in [
            ("ATR", self.atr_cfg),
            ("Baseline", self.baseline_cfg),
        ] + ([("Trend", self.trend_cfg)] if self.trend_cfg is not None else []):
            raw = cfg["function"](data, **(cfg.get("parameters", {}) or {}))
            out = convert_np_to_pd(raw)
            df = (
                out
                if isinstance(out, pd.DataFrame)
                else pd.DataFrame({cfg["name"]: out})
            )
            df.columns = df.columns.astype(str)
            df = df.add_prefix(f"{piece_name}_")
            out_frames.append(df)

        # Patterns
        name_to_fn: Dict[str, Callable] = {
            c["name"]: c["function"] for c in icfg.ta_lib_candlestick
        }
        for name in self.pattern_names:
            if name not in name_to_fn:
                raise ValueError(f"Unknown candlestick pattern: {name}")
            raw = name_to_fn[name](data)
            ser = convert_np_to_pd(raw)
            df = ser if isinstance(ser, pd.DataFrame) else pd.DataFrame({name: ser})
            df.columns = [
                f"PAT_{name}" if df.shape[1] == 1 else f"PAT_{name}_{c}"
                for c in df.columns
            ]
            out_frames.append(df)

        indicators = pd.concat(out_frames, axis=1)
        return pd.concat(
            [data.reset_index(drop=True), indicators.reset_index(drop=True)], axis=1
        )

    # ───────────────────────────── Signal Generation ─────────────────────────────

    def _generate_signals(
        self, data: pd.DataFrame, current_position: int, debug: bool = False
    ):
        """
        Return (signals, source_tag) using the most recent bar.
        Cooldown limits re-entries only; exits are never blocked.
        """
        signals: List[str] = []
        source = "CandleEntry"

        # Require at least a few rows for comparisons
        if len(data) < 5:
            return [NO_SIGNAL], source

        row = data.iloc[-1]
        close_series: pd.Series = data["Close"]

        # Reconstruct ATR & Baseline (prefixed)  :contentReference[oaicite:5]{index=5}
        atr_df_or_series = self._reconstruct_df(data, "ATR")
        baseline_df_or_series = self._reconstruct_df(data, "Baseline")

        atr_val = (
            float(atr_df_or_series.iloc[-1])
            if isinstance(atr_df_or_series, pd.Series)
            else float(atr_df_or_series.iloc[-1, 0])
        )
        baseline_val = (
            float(baseline_df_or_series.iloc[-1])
            if isinstance(baseline_df_or_series, pd.Series)
            else float(baseline_df_or_series.iloc[-1, 0])
        )

        # Guards for both directions
        if not np.isfinite(atr_val) or atr_val <= 0:
            return [NO_SIGNAL], source

        body = float(abs(row["Close"] - row["Open"]))
        if body < self.min_body_atr * atr_val:
            return [NO_SIGNAL], source

        # Optional trend bias:
        # If trend filter provided AND has a signal_function, derive bias from its signals.
        # Otherwise fallback to sign(mean of last raw values).
        trend_bias = None
        if self.trend_cfg is not None:
            try:
                trend_df_or_series = self._reconstruct_df(data, "Trend")
                # Prefer signal function if provided (NNFX-style). :contentReference[oaicite:6]{index=6}
                sig_fn = self.trend_cfg.get("signal_function")
                if callable(sig_fn):
                    trend_signals = self._get_indicator_signals(
                        sig_fn, trend_df_or_series, close_series
                    )
                    if any(s in trend_signals for s in (BULLISH_TREND, BULLISH_SIGNAL)):
                        trend_bias = 1
                    elif any(
                        s in trend_signals for s in (BEARISH_TREND, BEARISH_SIGNAL)
                    ):
                        trend_bias = -1
                    else:
                        trend_bias = 0
                else:
                    # fallback: sign of last value(s)
                    if isinstance(trend_df_or_series, pd.Series):
                        last_vals = [trend_df_or_series.iloc[-1]]
                    else:
                        last_vals = list(trend_df_or_series.iloc[-1].values)
                    m = np.nanmean(last_vals)
                    trend_bias = 1 if m > 0 else (-1 if m < 0 else 0)
            except KeyError:
                trend_bias = None  # no Trend_ columns present

        # Evaluate candlestick whitelist on last bar
        long_hits: list[str] = []
        short_hits: list[str] = []
        for name in self.pattern_names:
            v = float(row.get(f"PAT_{name}", 0.0))
            if abs(v) >= self.min_abs_score:
                if v > 0:
                    long_hits.append(name)
                elif v < 0:
                    short_hits.append(name)

        # Directional permissions
        def baseline_ok(direction: str) -> bool:
            if not self.align_with_baseline:
                return True
            return (
                (row["Close"] > baseline_val)
                if direction == "LONG"
                else (row["Close"] < baseline_val)
            )

        def trend_ok(direction: str) -> bool:
            if trend_bias is None:
                return True
            return (trend_bias > 0) if direction == "LONG" else (trend_bias < 0)

        want_long = bool(long_hits) and baseline_ok("LONG") and trend_ok("LONG")
        want_short = bool(short_hits) and baseline_ok("SHORT") and trend_ok("SHORT")

        # Cooldown: applies to entries only (never block exits)
        can_enter = True
        if self._last_entry_bar is not None:
            bars_since = len(data) - 1 - self._last_entry_bar
            if bars_since < self.cooldown_bars:
                can_enter = False

        # Entries require flat position to avoid flip-on-bar (parity with NNFX)  :contentReference[oaicite:7]{index=7}
        if current_position == 0 and can_enter:
            if want_long:
                signals.append(ENTER_LONG)
                self._last_entry_bar = len(data) - 1
            elif want_short:
                signals.append(ENTER_SHORT)
                self._last_entry_bar = len(data) - 1

        # Exits never blocked by cooldown
        if current_position > 0:
            if self.opposite_exit and bool(short_hits):
                signals.append(EXIT_LONG)
                source = "Opposite Pattern Exit"
            elif row["Close"] < baseline_val:
                signals.append(EXIT_LONG)
                source = "Baseline Cross Exit"

        elif current_position < 0:
            if self.opposite_exit and bool(long_hits):
                signals.append(EXIT_SHORT)
                source = "Opposite Pattern Exit"
            elif row["Close"] > baseline_val:
                signals.append(EXIT_SHORT)
                source = "Baseline Cross Exit"

        if not signals:
            signals = [NO_SIGNAL]
        return signals, source

    # ───────────────────────────── Trade Planning ─────────────────────────────

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

        # Handle entry
        if ENTER_LONG in signals or ENTER_SHORT in signals:
            signal = ENTER_LONG if ENTER_LONG in signals else ENTER_SHORT

            # Reconstruct ATR; fallback compute if absent
            try:
                atr_df_or_series = self._reconstruct_df(data, "ATR")
                atr = (
                    float(atr_df_or_series.iloc[-1])
                    if isinstance(atr_df_or_series, pd.Series)
                    else float(atr_df_or_series.iloc[-1, 0])
                )
            except KeyError:
                atr = float(_compute_atr(data).iloc[-1])

            entry = float(row["Close"])
            direction = "BUY" if signal == ENTER_LONG else "SELL"

            # Two legs: TP1 + Runner with PositionManager
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
                        "breakeven_trigger": atr,  # move SL to BE at +1xATR
                        "trail_start": 2 * atr,  # start trailing at +2xATR
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

        # Handle exits
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
