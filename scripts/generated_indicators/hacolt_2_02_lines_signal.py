import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_hacolt_2_02_lines(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = df['HACOLT'] if 'HACOLT' in df.columns else pd.Series(index=df.index, dtype=float)

    # Use last two bars only
    try:
        prev_val = float(s.iloc[-2])
        curr_val = float(s.iloc[-1])
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Determine signal based on transition across zero line
    if pd.notna(prev_val) and pd.notna(curr_val):
        if (curr_val > 0) and (prev_val <= 0):
            tags.append(BULLISH_SIGNAL)
        elif (curr_val < 0) and (prev_val >= 0):
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Determine trend from the latest value
    if pd.notna(curr_val):
        if curr_val > 0:
            trend = BULLISH_TREND
        elif curr_val < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags