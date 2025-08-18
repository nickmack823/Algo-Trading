import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_vortex_indicator(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Basic validation
    if df is None or df.empty or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Identify VI+ and VI- columns (supports any length suffix)
    cols_lower = {c.lower(): c for c in df.columns}
    plus_candidates = [orig for low, orig in cols_lower.items() if low.startswith('vi_plus')]
    minus_candidates = [orig for low, orig in cols_lower.items() if low.startswith('vi_minus')]

    if not plus_candidates or not minus_candidates:
        return [NO_SIGNAL, NEUTRAL_TREND]

    plus_col = plus_candidates[0]
    minus_col = minus_candidates[0]

    # Use last two bars for cross decisions
    plus_prev = df[plus_col].iloc[-2]
    plus_curr = df[plus_col].iloc[-1]
    minus_prev = df[minus_col].iloc[-2]
    minus_curr = df[minus_col].iloc[-1]

    have_prev = pd.notna(plus_prev) and pd.notna(minus_prev)
    have_curr = pd.notna(plus_curr) and pd.notna(minus_curr)

    if have_prev and have_curr:
        bull_cross = (plus_prev <= minus_prev) and (plus_curr > minus_curr)
        bear_cross = (plus_prev >= minus_prev) and (plus_curr < minus_curr)

        if bull_cross:
            tags.append(BULLISH_SIGNAL)
        elif bear_cross:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Determine current trend from the latest bar
    if pd.notna(plus_curr) and pd.notna(minus_curr):
        if plus_curr > minus_curr:
            trend_tag = BULLISH_TREND
        elif plus_curr < minus_curr:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags