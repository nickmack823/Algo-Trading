import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_mcginley_dynamic_2_3(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Expect 'mcg' column from McginleyDynamic23 calculation output
    if "mcg" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = df["mcg"].astype(float)

    n = len(s)
    # Determine signal via slope change (rise/fall flip) using the last three points
    signal_tag = NO_SIGNAL
    if n >= 3:
        cur = s.iloc[-1]
        prev = s.iloc[-2]
        prev2 = s.iloc[-3]
        if pd.notna(cur) and pd.notna(prev) and pd.notna(prev2):
            slope_prev = prev - prev2
            slope_cur = cur - prev
            if slope_prev <= 0 and slope_cur > 0:
                signal_tag = BULLISH_SIGNAL
            elif slope_prev >= 0 and slope_cur < 0:
                signal_tag = BEARISH_SIGNAL

    tags.append(signal_tag)

    # Determine trend from the last two points (rising/falling/flat)
    trend_tag = NEUTRAL_TREND
    if n >= 2:
        cur = s.iloc[-1]
        prev = s.iloc[-2]
        if pd.notna(cur) and pd.notna(prev):
            if cur > prev:
                trend_tag = BULLISH_TREND
            elif cur < prev:
                trend_tag = BEARISH_TREND

    tags.append(trend_tag)
    return tags