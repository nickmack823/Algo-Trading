import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_detrended_synthetic_price_goscillators(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame) or "DSP" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    dsp = df["DSP"]
    v_prev = dsp.iloc[-2]
    v_curr = dsp.iloc[-1]

    # Handle NaNs robustly
    if pd.isna(v_prev) or pd.isna(v_curr):
        trend = NEUTRAL_TREND
        if pd.notna(v_curr):
            if v_curr > 0:
                trend = BULLISH_TREND
            elif v_curr < 0:
                trend = BEARISH_TREND
        return [NO_SIGNAL, trend]

    # Signal: zero-line cross in the last two bars
    if v_prev <= 0 and v_curr > 0:
        sig = BULLISH_SIGNAL
    elif v_prev >= 0 and v_curr < 0:
        sig = BEARISH_SIGNAL
    else:
        sig = NO_SIGNAL

    # Trend: sign of current DSP
    if v_curr > 0:
        trend = BULLISH_TREND
    elif v_curr < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    return [sig, trend]