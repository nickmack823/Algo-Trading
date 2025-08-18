import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_is_calculation(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty or 'Pente' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    pente = pd.to_numeric(df['Pente'], errors='coerce')

    # Determine signal using last two bars (zero-cross)
    if len(pente) >= 2:
        prev = pente.iloc[-2]
        curr = pente.iloc[-1]
        if pd.notna(prev) and pd.notna(curr):
            if prev <= 0 and curr > 0:
                tags.append(BULLISH_SIGNAL)
            elif prev >= 0 and curr < 0:
                tags.append(BEARISH_SIGNAL)
            else:
                tags.append(NO_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Determine trend from current reading
    curr = pente.iloc[-1] if len(pente) > 0 else float('nan')
    if pd.isna(curr):
        tags.append(NEUTRAL_TREND)
    elif curr > 0:
        tags.append(BULLISH_TREND)
    elif curr < 0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags