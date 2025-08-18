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

def signal_xma_coloured_updated_for_nnfx(df: pd.DataFrame) -> list[str]:
    def classify(row: pd.Series) -> str:
        up = row.get("Up")
        dn = row.get("Dn")
        fl = row.get("Fl")
        up_flag = pd.notna(up)
        dn_flag = pd.notna(dn)
        fl_flag = pd.notna(fl)

        if up_flag and not dn_flag:
            return "up"
        if dn_flag and not up_flag:
            return "dn"
        # If both Up and Dn present (rare tie) or neither present, fall back to flat/neutral
        if up_flag and dn_flag:
            return "fl"
        if fl_flag:
            return "fl"
        return "fl"

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last_state = classify(df.iloc[-1])

    # Determine signal using last two bars
    signal_tag = NO_SIGNAL
    if n >= 2:
        prev_state = classify(df.iloc[-2])
        if last_state == "up" and prev_state in ("dn", "fl"):
            signal_tag = BULLISH_SIGNAL
        elif last_state == "dn" and prev_state in ("up", "fl"):
            signal_tag = BEARISH_SIGNAL

    # Trend from the latest bar
    if last_state == "up":
        trend_tag = BULLISH_TREND
    elif last_state == "dn":
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    return [signal_tag, trend_tag]