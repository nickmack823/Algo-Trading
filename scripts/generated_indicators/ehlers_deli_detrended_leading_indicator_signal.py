import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_ehlers_deli_detrended_leading_indicator(series: pd.Series) -> list[str]:
    # Ensure float dtype and handle edge cases
    try:
        s = series.astype(float)
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if s.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = s.iloc[-2]
    curr = s.iloc[-1]

    # Signal line (short EMA) for cross confirmation
    sig = s.ewm(span=5, adjust=False).mean()
    sig_prev = sig.iloc[-2] if sig.shape[0] >= 2 else float('nan')
    sig_curr = sig.iloc[-1] if sig.shape[0] >= 1 else float('nan')

    # Cross conditions using last two bars
    valid_vals = pd.notna(prev) and pd.notna(curr)
    valid_sig = pd.notna(sig_prev) and pd.notna(sig_curr)

    cross_up_zero = valid_vals and (prev <= 0.0) and (curr > 0.0)
    cross_down_zero = valid_vals and (prev >= 0.0) and (curr < 0.0)

    cross_up_sig = valid_vals and valid_sig and (prev <= sig_prev) and (curr > sig_curr)
    cross_down_sig = valid_vals and valid_sig and (prev >= sig_prev) and (curr < sig_curr)

    tags: list[str] = []

    if cross_up_zero or cross_up_sig:
        tags.append(BULLISH_SIGNAL)
    elif cross_down_zero or cross_down_sig:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag based on current DELI level
    if pd.isna(curr):
        trend_tag = NEUTRAL_TREND
    elif curr > 0.0:
        trend_tag = BULLISH_TREND
    elif curr < 0.0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags