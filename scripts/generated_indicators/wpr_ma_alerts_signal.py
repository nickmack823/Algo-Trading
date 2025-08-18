import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_wpr_ma_alerts(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) == 0 or not {"WPR", "Signal", "Cross"}.issubset(df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tail = df[["WPR", "Signal", "Cross"]].tail(2).astype(float)

    wpr_vals = tail["WPR"].to_numpy()
    cross_vals = tail["Cross"].to_numpy()

    bullish_cross = False
    bearish_cross = False
    bullish_exit_oversold = False
    bearish_exit_overbought = False

    if len(tail) == 2:
        prev_wpr, curr_wpr = wpr_vals[0], wpr_vals[1]
        prev_cross, curr_cross = cross_vals[0], cross_vals[1]

        # Cross event detection (WPR vs Signal)
        if not pd.isna(prev_cross) and not pd.isna(curr_cross) and prev_cross != curr_cross:
            if curr_cross == 1:
                bullish_cross = True
            elif curr_cross == -1:
                bearish_cross = True

        # Threshold exits
        if not pd.isna(prev_wpr) and not pd.isna(curr_wpr):
            if prev_wpr <= -80 and curr_wpr > -80:
                bullish_exit_oversold = True
            if prev_wpr >= -20 and curr_wpr < -20:
                bearish_exit_overbought = True

    # Primary signal
    if bullish_cross or bullish_exit_oversold:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross or bearish_exit_overbought:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Zone tags (current bar)
    curr_wpr = wpr_vals[-1] if len(wpr_vals) > 0 else float("nan")
    if not pd.isna(curr_wpr):
        if curr_wpr <= -80:
            tags.append(OVERSOLD)
        elif curr_wpr >= -20:
            tags.append(OVERBOUGHT)

    # Trend tag from latest valid Cross state (prefer current, fall back to previous)
    curr_cross_val = cross_vals[-1] if len(cross_vals) > 0 else float("nan")
    if pd.isna(curr_cross_val) or curr_cross_val == 0:
        if len(cross_vals) == 2 and not pd.isna(cross_vals[0]) and cross_vals[0] != 0:
            curr_cross_val = cross_vals[0]

    if curr_cross_val == 1:
        trend_tag = BULLISH_TREND
    elif curr_cross_val == -1:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags