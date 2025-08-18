import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_ehlers_deli__detrended_leading_indicator(series: pd.Series) -> list[str]:
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.Series(series, dtype="float64")
    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s_prev = s.iloc[-2]
    s_curr = s.iloc[-1]

    sig = s.ewm(span=5, adjust=False, min_periods=1).mean()
    sig_prev = sig.iloc[-2] if len(sig) >= 2 else pd.NA
    sig_curr = sig.iloc[-1] if len(sig) >= 1 else pd.NA

    up_zero = pd.notna(s_prev) and pd.notna(s_curr) and (s_prev <= 0) and (s_curr > 0)
    down_zero = pd.notna(s_prev) and pd.notna(s_curr) and (s_prev >= 0) and (s_curr < 0)

    up_sig = pd.notna(s_prev) and pd.notna(s_curr) and pd.notna(sig_prev) and pd.notna(sig_curr) and (s_prev <= sig_prev) and (s_curr > sig_curr)
    down_sig = pd.notna(s_prev) and pd.notna(s_curr) and pd.notna(sig_prev) and pd.notna(sig_curr) and (s_prev >= sig_prev) and (s_curr < sig_curr)

    up = up_zero or up_sig
    down = down_zero or down_sig

    if up and not down:
        signal_tag = BULLISH_SIGNAL
    elif down and not up:
        signal_tag = BEARISH_SIGNAL
    else:
        signal_tag = NO_SIGNAL

    if pd.notna(s_curr):
        trend_tag = BULLISH_TREND if s_curr > 0 else (BEARISH_TREND if s_curr < 0 else NEUTRAL_TREND)
    else:
        trend_tag = NEUTRAL_TREND

    return [signal_tag, trend_tag]