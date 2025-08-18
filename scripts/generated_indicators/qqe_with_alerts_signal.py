import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_qqe_with_alerts(df: pd.DataFrame) -> list[str]:
    # Validate inputs and required columns
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]
    required_cols = ['QQE_RSI_MA', 'QQE_TrendLevel']
    if not all(col in df.columns for col in required_cols):
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    rsi = pd.to_numeric(df['QQE_RSI_MA'], errors='coerce')
    tr = pd.to_numeric(df['QQE_TrendLevel'], errors='coerce')

    # Last two bars
    rsi_prev = rsi.iat[-2]
    rsi_cur = rsi.iat[-1]
    tr_prev = tr.iat[-2]
    tr_cur = tr.iat[-1]

    # Initialize tags
    tags: list[str] = []

    # Cross logic between Fast (RSI_MA) and Slow (TrendLevel)
    has_pair_prev = pd.notna(rsi_prev) and pd.notna(tr_prev)
    has_pair_cur = pd.notna(rsi_cur) and pd.notna(tr_cur)
    cross_up = False
    cross_down = False

    if has_pair_prev and has_pair_cur:
        cross_up = (rsi_prev <= tr_prev) and (rsi_cur > tr_cur)
        cross_down = (rsi_prev >= tr_prev) and (rsi_cur < tr_cur)

    # Fallback to midline (50) crosses if TrendLevel is unavailable
    mid_up = False
    mid_down = False
    if not (cross_up or cross_down) and pd.notna(rsi_prev) and pd.notna(rsi_cur):
        mid_up = (rsi_prev <= 50) and (rsi_cur > 50)
        mid_down = (rsi_prev >= 50) and (rsi_cur < 50)

    # Signal tagging
    if cross_up or mid_up:
        tags.append(BULLISH_SIGNAL)
    elif cross_down or mid_down:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Overbought/Oversold context
    if pd.notna(rsi_cur):
        if rsi_cur >= 70:
            tags.append(OVERBOUGHT)
        elif rsi_cur <= 30:
            tags.append(OVERSOLD)

    # Trend tagging (exactly one)
    if pd.notna(rsi_cur) and pd.notna(tr_cur):
        if rsi_cur > tr_cur:
            tags.append(BULLISH_TREND)
        elif rsi_cur < tr_cur:
            tags.append(BEARISH_TREND)
        else:
            # Equal -> use midline as tie-breaker
            if rsi_cur > 50:
                tags.append(BULLISH_TREND)
            elif rsi_cur < 50:
                tags.append(BEARISH_TREND)
            else:
                tags.append(NEUTRAL_TREND)
    else:
        # Fallback to RSI vs midline if TrendLevel missing
        if pd.notna(rsi_cur):
            if rsi_cur > 50:
                tags.append(BULLISH_TREND)
            elif rsi_cur < 50:
                tags.append(BEARISH_TREND)
            else:
                tags.append(NEUTRAL_TREND)
        else:
            tags.append(NEUTRAL_TREND)

    return tags