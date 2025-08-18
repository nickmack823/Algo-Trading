import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_gchannel(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Validate required column
    if not isinstance(df, pd.DataFrame) or 'Middle' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    mid = pd.to_numeric(df['Middle'], errors='coerce')

    # Try to locate a price series (prefer Close)
    price = None
    for col in ('Close', 'close', 'CLOSE', 'Price', 'price'):
        if col in df.columns:
            price = pd.to_numeric(df[col], errors='coerce')
            break

    n = len(df)

    # Signal: price crossing Middle (last two bars)
    long_cross = False
    short_cross = False
    if price is not None and n >= 2:
        p_prev = price.iloc[-2]
        p_last = price.iloc[-1]
        m_prev = mid.iloc[-2]
        m_last = mid.iloc[-1]
        if pd.notna(p_prev) and pd.notna(p_last) and pd.notna(m_prev) and pd.notna(m_last):
            long_cross = (p_prev <= m_prev) and (p_last > m_last)
            short_cross = (p_prev >= m_prev) and (p_last < m_last)

    if long_cross:
        tags.append(BULLISH_SIGNAL)
    elif short_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag: prefer price vs Middle; fallback to Middle slope
    trend = NEUTRAL_TREND
    if price is not None and n >= 1:
        p_last = price.iloc[-1]
        m_last = mid.iloc[-1]
        if pd.notna(p_last) and pd.notna(m_last):
            if p_last > m_last:
                trend = BULLISH_TREND
            elif p_last < m_last:
                trend = BEARISH_TREND
            else:
                trend = NEUTRAL_TREND
        else:
            # Fallback to Middle slope if last price or middle is NaN
            if n >= 2:
                m_prev = mid.iloc[-2]
                if pd.notna(m_prev) and pd.notna(m_last):
                    if m_last > m_prev:
                        trend = BULLISH_TREND
                    elif m_last < m_prev:
                        trend = BEARISH_TREND
                    else:
                        trend = NEUTRAL_TREND
    else:
        # No price available; use Middle slope if possible
        if n >= 2:
            m_prev = mid.iloc[-2]
            m_last = mid.iloc[-1]
            if pd.notna(m_prev) and pd.notna(m_last):
                if m_last > m_prev:
                    trend = BULLISH_TREND
                elif m_last < m_prev:
                    trend = BEARISH_TREND
                else:
                    trend = NEUTRAL_TREND

    tags.append(trend)
    return tags