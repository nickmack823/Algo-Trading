import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_momentum_candles_w_atr(df: pd.DataFrame) -> list[str]:
    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    required_cols = ['BullOpen', 'BullClose', 'BearOpen', 'BearClose']
    if not all(col in df.columns for col in required_cols):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Current bar signal
    last = df.iloc[-1]
    is_bullish = pd.notna(last['BullClose'])
    is_bearish = pd.notna(last['BearClose'])

    tags: list[str] = []
    if is_bullish and not is_bearish:
        tags.append(BULLISH_SIGNAL)
    elif is_bearish and not is_bullish:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend from the last two bars (consecutive candles strengthen trend)
    last2 = df.tail(2)
    bull2 = last2['BullClose'].notna()
    bear2 = last2['BearClose'].notna()

    if len(last2) >= 2:
        if bool(bull2.iloc[-2]) and bool(bull2.iloc[-1]):
            trend = BULLISH_TREND
        elif bool(bear2.iloc[-2]) and bool(bear2.iloc[-1]):
            trend = BEARISH_TREND
        else:
            if is_bullish and not is_bearish:
                trend = BULLISH_TREND
            elif is_bearish and not is_bullish:
                trend = BEARISH_TREND
            else:
                trend = NEUTRAL_TREND
    else:
        if is_bullish and not is_bearish:
            trend = BULLISH_TREND
        elif is_bearish and not is_bullish:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND

    tags.append(trend)
    return tags