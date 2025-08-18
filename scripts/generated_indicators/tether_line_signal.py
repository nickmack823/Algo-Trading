import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_tether_line(df: pd.DataFrame) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if not {'AboveCenter', 'BelowCenter'}.issubset(df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    last2 = df.tail(2)
    prev_above = pd.notna(last2['AboveCenter'].iloc[0])
    prev_below = pd.notna(last2['BelowCenter'].iloc[0])
    curr_above = pd.notna(last2['AboveCenter'].iloc[1])
    curr_below = pd.notna(last2['BelowCenter'].iloc[1])

    cross_up = prev_below and curr_above
    cross_down = prev_above and curr_below

    if cross_up:
        tags = [BULLISH_SIGNAL]
    elif cross_down:
        tags = [BEARISH_SIGNAL]
    else:
        tags = [NO_SIGNAL]

    if curr_above and not curr_below:
        tags.append(BULLISH_TREND)
    elif curr_below and not curr_above:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags