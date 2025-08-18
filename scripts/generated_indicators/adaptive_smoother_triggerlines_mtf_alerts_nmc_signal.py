import pandas as pd
from scripts.config import (
    BULLISH_SIGNAL,
    BEARISH_SIGNAL,
    BULLISH_TREND,
    BEARISH_TREND,
    NEUTRAL_TREND,
    NO_SIGNAL,
)


def signal_adaptive_smoother_triggerlines_mtf_alerts_nmc(df: pd.DataFrame) -> list[str]:
    # Basic guards
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Ensure required columns exist
    if "lstrend" not in df.columns and "lwtrend" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Try to use lstrend; fall back to lwtrend for NaNs on the last two bars
    lstrend = df["lstrend"] if "lstrend" in df.columns else pd.Series([float("nan")] * len(df), index=df.index)
    lwtrend = df["lwtrend"] if "lwtrend" in df.columns else pd.Series([float("nan")] * len(df), index=df.index)

    prev = lstrend.iloc[-2]
    cur = lstrend.iloc[-1]

    if pd.isna(prev):
        prev = lwtrend.iloc[-2]
    if pd.isna(cur):
        cur = lwtrend.iloc[-1]

    tags: list[str] = []

    # Signal on change in trend slope (+1 to -1 or -1 to +1)
    if pd.notna(prev) and pd.notna(cur) and prev != cur:
        if cur > 0:
            tags.append(BULLISH_SIGNAL)
        elif cur < 0:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag from current slope
    if pd.notna(cur):
        if cur > 0:
            tags.append(BULLISH_TREND)
        elif cur < 0:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags