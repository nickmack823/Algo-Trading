import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_trimagen(series: pd.Series) -> list[str]:
    tags: list[str] = []
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = series.astype(float)

    # Determine signal based on slope cross (using last two slopes -> needs last 3 values)
    signal_emitted = False
    if len(s) >= 3:
        v3 = s.iloc[-3]
        v2 = s.iloc[-2]
        v1 = s.iloc[-1]
        if pd.notna(v3) and pd.notna(v2) and pd.notna(v1):
            delta_prev = v2 - v3
            delta_curr = v1 - v2
            if delta_prev <= 0 and delta_curr > 0:
                tags.append(BULLISH_SIGNAL)
                signal_emitted = True
            elif delta_prev >= 0 and delta_curr < 0:
                tags.append(BEARISH_SIGNAL)
                signal_emitted = True

    if not signal_emitted:
        tags.append(NO_SIGNAL)

    # Trend tag from latest slope (last two bars)
    v2 = s.iloc[-2]
    v1 = s.iloc[-1]
    trend = NEUTRAL_TREND
    if pd.notna(v2) and pd.notna(v1):
        d = v1 - v2
        if d > 0:
            trend = BULLISH_TREND
        elif d < 0:
            trend = BEARISH_TREND

    tags.append(trend)
    return tags