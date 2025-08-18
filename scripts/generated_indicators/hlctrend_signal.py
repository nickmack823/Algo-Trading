import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_hlctrend(df: pd.DataFrame) -> list[str]:
    # Expecting df with columns ['first','second'] from Hlctrend(...)
    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if "first" in df.columns:
        s_first = pd.to_numeric(df["first"], errors="coerce")
    else:
        s_first = pd.Series(float("nan"), index=df.index, dtype="float64")

    if "second" in df.columns:
        s_second = pd.to_numeric(df["second"], errors="coerce")
    else:
        s_second = pd.Series(float("nan"), index=df.index, dtype="float64")

    # Composite: EMAC - HLC_baseline ∝ (first - second)
    comp = s_first - s_second

    # Last two bars
    curr = comp.iloc[-1] if len(comp) >= 1 else float("nan")
    prev = comp.iloc[-2] if len(comp) >= 2 else float("nan")

    tags: list[str] = []

    # Signal on zero-cross of composite
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            tags.append(BULLISH_SIGNAL)
        elif prev >= 0 and curr < 0:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend by current composite sign
    if pd.notna(curr):
        if curr > 0:
            tags.append(BULLISH_TREND)
        elif curr < 0:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags