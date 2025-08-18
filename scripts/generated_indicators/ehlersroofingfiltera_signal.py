import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_ehlersroofingfiltera(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or df.empty or 'rfilt' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    rf = pd.to_numeric(df['rfilt'], errors='coerce')

    # Need at least two bars for cross decisions
    if rf.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = rf.iloc[-2]
    last = rf.iloc[-1]

    # Determine signal based on zero-line cross of Roofing Filter
    signal = NO_SIGNAL
    if pd.notna(prev) and pd.notna(last):
        if prev < 0 and last > 0:
            signal = BULLISH_SIGNAL
        elif prev > 0 and last < 0:
            signal = BEARISH_SIGNAL

    # Determine trend from the latest Roofing Filter value
    trend = NEUTRAL_TREND
    if pd.notna(last):
        if last > 0:
            trend = BULLISH_TREND
        elif last < 0:
            trend = BEARISH_TREND

    return [signal, trend]