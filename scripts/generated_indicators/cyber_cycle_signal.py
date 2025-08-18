import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_cyber_cycle(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if not isinstance(df, pd.DataFrame) or 'Cycle' not in df.columns or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    c = df['Cycle']
    c_prev = c.iloc[-2]
    c_curr = c.iloc[-1]

    if pd.isna(c_prev) or pd.isna(c_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    cross_up = (c_prev <= 0) and (c_curr > 0)
    cross_dn = (c_prev >= 0) and (c_curr < 0)
    rising = c_curr > c_prev
    falling = c_curr < c_prev

    if cross_up:
        tags.append(BULLISH_SIGNAL)
    elif cross_dn:
        tags.append(BEARISH_SIGNAL)
    elif rising:
        tags.append(BULLISH_SIGNAL)
    elif falling:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    if (c_curr > 0 and rising) or (c_curr > 0 and not falling):
        trend = BULLISH_TREND
    elif (c_curr < 0 and falling) or (c_curr < 0 and not rising):
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags