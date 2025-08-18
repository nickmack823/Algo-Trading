import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_price_momentum_oscillator(df: pd.DataFrame) -> list[str]:
    # Expecting columns: 'PMO', 'Signal'
    try:
        pmo = df['PMO'].astype(float)
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(pmo)
    pmo_curr = pmo.iloc[-1] if n >= 1 else float('nan')
    pmo_prev = pmo.iloc[-2] if n >= 2 else float('nan')

    tags: list[str] = []

    # Zero-line cross logic (TSI-style interpretation)
    has_vals = pd.notna(pmo_curr) and pd.notna(pmo_prev)
    bull_cross = has_vals and (pmo_prev <= 0.0) and (pmo_curr > 0.0)
    bear_cross = has_vals and (pmo_prev >= 0.0) and (pmo_curr < 0.0)

    if bull_cross:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag from current PMO value
    if pd.isna(pmo_curr):
        trend_tag = NEUTRAL_TREND
    elif pmo_curr > 0.0:
        trend_tag = BULLISH_TREND
    elif pmo_curr < 0.0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags