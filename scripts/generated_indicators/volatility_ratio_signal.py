import pandas as pd
from scripts.config import (
    BULLISH_SIGNAL,
    BEARISH_SIGNAL,
    BULLISH_TREND,
    BEARISH_TREND,
    NEUTRAL_TREND,
    OVERBOUGHT,
    OVERSOLD,
    NO_SIGNAL,
    HIGH_VOLUME,
    LOW_VOLUME,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    INCREASING_VOLATILITY,
    DECREASING_VOLATILITY,
    STABLE_VOLATILITY,
    INCONCLUSIVE,
)

def signal_volatility_ratio(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) == 0 or "VR" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    vr = df["VR"]
    if vr.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last = vr.iloc[-1]
    prev = vr.iloc[-2] if len(vr) > 1 else pd.NA

    if pd.isna(last):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tol = 0.02

    if abs(last - 1.0) <= tol:
        tags.append(STABLE_VOLATILITY)
    elif last > 1.0:
        tags.append(HIGH_VOLATILITY)
    else:
        tags.append(LOW_VOLATILITY)

    if not pd.isna(prev):
        crossed_up = prev < 1.0 and last >= 1.0
        crossed_down = prev > 1.0 and last <= 1.0
        if crossed_up:
            tags.append(INCREASING_VOLATILITY)
        elif crossed_down:
            tags.append(DECREASING_VOLATILITY)
        else:
            delta = last - prev
            if delta > tol:
                tags.append(INCREASING_VOLATILITY)
            elif delta < -tol:
                tags.append(DECREASING_VOLATILITY)

    if not tags:
        tags.append(NO_SIGNAL)

    tags.append(NEUTRAL_TREND)
    # ensure uniqueness while preserving order
    tags = list(dict.fromkeys(tags))
    # ensure exactly one final trend tag
    if tags.count(NEUTRAL_TREND) > 1:
        tags = [t for i, t in enumerate(tags) if t != NEUTRAL_TREND or i == len(tags) - 1]
    return tags