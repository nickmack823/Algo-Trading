import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_aso(df: pd.DataFrame) -> list[str]:
    required_cols = {"ASO_Bulls", "ASO_Bears"}
    if not isinstance(df, pd.DataFrame) or not required_cols.issubset(df.columns) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    bulls = df["ASO_Bulls"]
    bears = df["ASO_Bears"]
    diff = bulls - bears

    prev = diff.iloc[-2]
    curr = diff.iloc[-1]

    # Trend determination
    if pd.isna(curr):
        trend = NEUTRAL_TREND
    elif curr > 0:
        trend = BULLISH_TREND
    elif curr < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Signal determination (last two bars crossover of bulls vs bears / zero on diff)
    if pd.isna(prev) or pd.isna(curr):
        signal = NO_SIGNAL
    elif prev <= 0 and curr > 0:
        signal = BULLISH_SIGNAL
    elif prev >= 0 and curr < 0:
        signal = BEARISH_SIGNAL
    else:
        signal = NO_SIGNAL

    return [signal, trend]