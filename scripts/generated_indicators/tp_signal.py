import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_tp(df: pd.DataFrame) -> list[str]:
    # Expecting df with columns ['TP','Up','Dn']
    try:
        s = df['TP']
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if s is None or len(s) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last = s.iloc[-1]
    prev = s.iloc[-2] if len(s) >= 2 else pd.NA

    signal_tag = NO_SIGNAL
    if pd.notna(last) and pd.notna(prev):
        if prev <= 0 and last > 0:
            signal_tag = BULLISH_SIGNAL
        elif prev >= 0 and last < 0:
            signal_tag = BEARISH_SIGNAL

    trend_tag = NEUTRAL_TREND
    if pd.notna(last):
        if last > 0:
            trend_tag = BULLISH_TREND
        elif last < 0:
            trend_tag = BEARISH_TREND

    return [signal_tag, trend_tag]