import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_jposcillator(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty or 'Jp' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    jp = pd.to_numeric(df['Jp'], errors='coerce')

    j0 = jp.iloc[-1] if len(jp) >= 1 else float('nan')
    j1 = jp.iloc[-2] if len(jp) >= 2 else float('nan')

    signal = None
    if pd.notna(j0) and pd.notna(j1):
        if j1 <= 0 and j0 > 0:
            signal = BULLISH_SIGNAL
        elif j1 >= 0 and j0 < 0:
            signal = BEARISH_SIGNAL

    if signal is None:
        tags.append(NO_SIGNAL)
    else:
        tags.append(signal)

    if pd.isna(j0):
        trend = NEUTRAL_TREND
    elif j0 > 0:
        trend = BULLISH_TREND
    elif j0 < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags