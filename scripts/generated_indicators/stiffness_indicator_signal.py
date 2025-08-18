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

def signal_stiffness_indicator(df: pd.DataFrame) -> list[str]:
    # Validate input
    try:
        s = pd.to_numeric(df["Stiffness"], errors="coerce")
        sig = pd.to_numeric(df["Signal"], errors="coerce")
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s1, s2 = s.iloc[-2], s.iloc[-1]
    sig1, sig2 = sig.iloc[-2], sig.iloc[-1]

    if pd.isna(s1) or pd.isna(s2) or pd.isna(sig1) or pd.isna(sig2):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Volatility trend via cross/slope of Stiffness vs its signal
    cross_up = s1 <= sig1 and s2 > sig2
    cross_down = s1 >= sig1 and s2 < sig2
    delta = s2 - s1

    if cross_up:
        vol_tag = DECREASING_VOLATILITY
    elif cross_down:
        vol_tag = INCREASING_VOLATILITY
    else:
        if delta > 0:
            vol_tag = DECREASING_VOLATILITY
        elif delta < 0:
            vol_tag = INCREASING_VOLATILITY
        else:
            vol_tag = STABLE_VOLATILITY

    # Regime/extreme assessment on recent history
    recent = s.dropna()
    if not recent.empty:
        window = min(len(recent), 200)
        recent = recent.iloc[-window:]
        p10 = recent.quantile(0.10)
        p90 = recent.quantile(0.90)

        # Extreme lows may precede volatility expansions -> prioritize INCREASING_VOLATILITY
        if s2 <= p10:
            vol_tag = INCREASING_VOLATILITY
        elif s2 >= p90:
            vol_tag = LOW_VOLATILITY

    return [vol_tag, NEUTRAL_TREND]