import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_momentum_candles_modified_w_atr(df: pd.DataFrame) -> list[str]:
    # Validate input and required columns
    if df is None or not isinstance(df, pd.DataFrame):
        return [NO_SIGNAL, NEUTRAL_TREND]
    for col in ("Value", "Threshold_Pos", "Threshold_Neg"):
        if col not in df.columns:
            return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    v = df["Value"].astype(float)
    tp = df["Threshold_Pos"].astype(float)
    tn = df["Threshold_Neg"].astype(float)

    # Current bar
    v_curr = v.iloc[-1]
    tp_curr = tp.iloc[-1]
    tn_curr = tn.iloc[-1]

    # Previous bar (for cross logic)
    v_prev = v.iloc[-2] if n >= 2 else pd.NA
    tp_prev = tp.iloc[-2] if n >= 2 else pd.NA
    tn_prev = tn.iloc[-2] if n >= 2 else pd.NA

    tags: list[str] = []

    # Signal via threshold cross has priority, then zero-line cross
    signal_added = False
    if n >= 2:
        # Threshold cross checks (require non-NaN)
        if pd.notna(v_prev) and pd.notna(v_curr) and pd.notna(tp_prev) and pd.notna(tp_curr):
            if (v_prev < tp_prev) and (v_curr >= tp_curr):
                tags.append(BULLISH_SIGNAL)
                signal_added = True
            elif (v_prev > tn_prev) and (v_curr <= tn_curr):
                tags.append(BEARISH_SIGNAL)
                signal_added = True

        # Zero-line cross checks if no threshold cross
        if not signal_added and pd.notna(v_prev) and pd.notna(v_curr):
            if (v_prev <= 0.0) and (v_curr > 0.0):
                tags.append(BULLISH_SIGNAL)
                signal_added = True
            elif (v_prev >= 0.0) and (v_curr < 0.0):
                tags.append(BEARISH_SIGNAL)
                signal_added = True

    if not signal_added:
        tags.append(NO_SIGNAL)

    # Trend determination per interpretation: above zero bullish, below zero bearish
    if pd.isna(v_curr):
        trend_tag = NEUTRAL_TREND
    elif v_curr > 0.0:
        trend_tag = BULLISH_TREND
    elif v_curr < 0.0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags