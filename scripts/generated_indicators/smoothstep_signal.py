import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_smoothstep(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    s = pd.to_numeric(df.get("SmoothStep"), errors="coerce") if isinstance(df, pd.DataFrame) else None
    if s is None or s.shape[0] == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    curr = s.iloc[-1] if s.shape[0] >= 1 else float("nan")
    prev = s.iloc[-2] if s.shape[0] >= 2 else float("nan")

    threshold = 0.5

    # Signal: cross of SmoothStep over/under the midline (0.5)
    signal = None
    if pd.notna(prev) and pd.notna(curr):
        crossed_up = (prev <= threshold) and (curr > threshold)
        crossed_down = (prev >= threshold) and (curr < threshold)
        if crossed_up:
            signal = BULLISH_SIGNAL
        elif crossed_down:
            signal = BEARISH_SIGNAL

    if signal is None:
        tags.append(NO_SIGNAL)
    else:
        tags.append(signal)

    # Trend: position relative to midline
    trend = NEUTRAL_TREND
    if pd.notna(curr):
        if curr > threshold:
            trend = BULLISH_TREND
        elif curr < threshold:
            trend = BEARISH_TREND

    tags.append(trend)
    return tags