import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_precision_trend_histogram(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or df.empty or not {"Up", "Down", "Trend"}.issubset(df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    # Determine trend tag from the latest bar
    last_trend_val = df["Trend"].iloc[-1] if n >= 1 else float("nan")
    if pd.notna(last_trend_val):
        if last_trend_val > 0:
            trend_tag = BULLISH_TREND
        elif last_trend_val < 0:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Need at least two bars for cross/transition decisions
    if n < 2:
        return [NO_SIGNAL, trend_tag]

    last2 = df.iloc[-2:]
    up_prev = pd.notna(last2["Up"].iloc[0]) and float(last2["Up"].iloc[0]) > 0.0
    up_now = pd.notna(last2["Up"].iloc[1]) and float(last2["Up"].iloc[1]) > 0.0
    down_prev = pd.notna(last2["Down"].iloc[0]) and float(last2["Down"].iloc[0]) > 0.0
    down_now = pd.notna(last2["Down"].iloc[1]) and float(last2["Down"].iloc[1]) > 0.0

    # Signal logic: new appearance of Up/Down on the latest bar
    if up_now and not up_prev:
        signal = BULLISH_SIGNAL
    elif down_now and not down_prev:
        signal = BEARISH_SIGNAL
    else:
        signal = NO_SIGNAL

    return [signal, trend_tag]