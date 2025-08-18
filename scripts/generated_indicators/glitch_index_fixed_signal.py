import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_glitch_index_fixed(df: pd.DataFrame) -> list[str]:
    # Robustness checks
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if "gli" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s_gli = df["gli"]
    s_state = df["state"] if "state" in df.columns else pd.Series([pd.NA] * len(df), index=df.index)

    # Last two bars
    g1 = s_gli.iloc[-2]
    g0 = s_gli.iloc[-1]
    st1 = s_state.iloc[-2]
    st0 = s_state.iloc[-1]

    # Helper: finite check
    def _is_num(x) -> bool:
        return pd.notna(x)

    # Zero-line crosses using last two bars
    bull_cross = _is_num(g1) and _is_num(g0) and (g1 <= 0) and (g0 > 0)
    bear_cross = _is_num(g1) and _is_num(g0) and (g1 >= 0) and (g0 < 0)

    # State-based shift (last two bars)
    bull_shift = _is_num(st1) and _is_num(st0) and (st1 <= 0) and (st0 > 0)
    bear_shift = _is_num(st1) and _is_num(st0) and (st1 >= 0) and (st0 < 0)

    tags: list[str] = []

    if bull_cross or bull_shift:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross or bear_shift:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend: based on current gli (fallback to state if needed)
    if _is_num(g0):
        if g0 > 0:
            trend = BULLISH_TREND
        elif g0 < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    elif _is_num(st0):
        if st0 > 0:
            trend = BULLISH_TREND
        elif st0 < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags