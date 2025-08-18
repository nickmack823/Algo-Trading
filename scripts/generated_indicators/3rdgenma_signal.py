import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_3rdgenma(df: pd.DataFrame) -> list[str]:
    # Expect df to contain columns: "MA3G" (the 3rd gen MA) and "MA1" (first-pass MA as price proxy)
    try:
        ma3g = df["MA3G"]
        ma1 = df["MA1"]
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if ma3g is None or ma1 is None or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Use last two bars
    ma3g_prev = ma3g.iloc[-2] if len(ma3g) >= 2 else pd.NA
    ma3g_curr = ma3g.iloc[-1] if len(ma3g) >= 1 else pd.NA
    ma1_prev = ma1.iloc[-2] if len(ma1) >= 2 else pd.NA
    ma1_curr = ma1.iloc[-1] if len(ma1) >= 1 else pd.NA

    # Robust to NaNs: if any needed value is NaN, return neutral
    vals = [ma3g_prev, ma3g_curr, ma1_prev, ma1_curr]
    if any(pd.isna(v) for v in vals):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Signal: cross of MA1 (price proxy) vs MA3G
    prev_above = ma1_prev > ma3g_prev
    curr_above = ma1_curr > ma3g_curr

    tags: list[str] = []
    if prev_above != curr_above:
        tags.append(BULLISH_SIGNAL if curr_above else BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend: slope of MA3G
    delta = ma3g_curr - ma3g_prev
    if pd.isna(delta) or delta == 0:
        trend_tag = NEUTRAL_TREND
    else:
        trend_tag = BULLISH_TREND if delta > 0 else BEARISH_TREND

    tags.append(trend_tag)
    return tags