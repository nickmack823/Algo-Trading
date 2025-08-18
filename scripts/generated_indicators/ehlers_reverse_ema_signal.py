import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_ehlers_reverse_ema(df: pd.DataFrame) -> list[str]:
    # Expect columns: 'Main' and 'EMA'
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if ('Main' not in df.columns) or ('EMA' not in df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    main = df['Main']
    ema = df['EMA']

    # Use only the last two bars
    m_prev, m_curr = main.iloc[-2], main.iloc[-1]
    e_prev, e_curr = ema.iloc[-2], ema.iloc[-1]

    # If any needed value is NaN, return neutral
    if pd.isna(m_prev) or pd.isna(m_curr) or pd.isna(e_prev) or pd.isna(e_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = []

    bullish_cross = (m_prev <= e_prev) and (m_curr > e_curr)
    bearish_cross = (m_prev >= e_prev) and (m_curr < e_curr)

    if bullish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend classification from current relationship
    if m_curr > e_curr:
        trend = BULLISH_TREND
    elif m_curr < e_curr:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags