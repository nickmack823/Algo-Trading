import pandas as pd
from scripts.config import BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_sinewma(series: pd.Series) -> list[str]:
    if series is None or not isinstance(series, pd.Series) or series.size < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev_val = series.iloc[-2]
    curr_val = series.iloc[-1]

    if pd.isna(prev_val) or pd.isna(curr_val):
        trend = NEUTRAL_TREND
    else:
        if curr_val > prev_val:
            trend = BULLISH_TREND
        elif curr_val < prev_val:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND

    return [NO_SIGNAL, trend]