import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_william_vix_fix(series: pd.Series) -> list[str]:
    tags: list[str] = []

    if series is None or series.size == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    valid = series.dropna()
    if valid.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev_idx = valid.index[-2]
    curr_idx = valid.index[-1]
    prev = valid.iloc[-2]
    curr = valid.iloc[-1]

    # Dynamic percentile bands for extremes (fear/complacency)
    window = min(50, max(5, valid.shape[0]))
    upper_q = series.rolling(window=window, min_periods=window).quantile(0.9)
    lower_q = series.rolling(window=window, min_periods=window).quantile(0.1)

    up_prev = upper_q.loc[prev_idx] if prev_idx in upper_q.index else float("nan")
    up_curr = upper_q.loc[curr_idx] if curr_idx in upper_q.index else float("nan")
    low_prev = lower_q.loc[prev_idx] if prev_idx in lower_q.index else float("nan")
    low_curr = lower_q.loc[curr_idx] if curr_idx in lower_q.index else float("nan")

    # Extremes tags
    if pd.notna(up_curr) and curr > up_curr:
        tags.append(OVERSOLD)  # Panic/fear extreme
    elif pd.notna(low_curr) and curr < low_curr:
        tags.append(OVERBOUGHT)  # Complacency extreme

    # Volatility level tags
    if pd.notna(up_curr) and curr > up_curr:
        tags.append(HIGH_VOLATILITY)
    elif pd.notna(low_curr) and curr < low_curr:
        tags.append(LOW_VOLATILITY)

    # Volatility direction tags (based on last two bars)
    if curr > prev:
        tags.append(INCREASING_VOLATILITY)
    elif curr < prev:
        tags.append(DECREASING_VOLATILITY)
    else:
        tags.append(STABLE_VOLATILITY)

    # Signal logic (NNFX-style, use last two bars)
    # Bullish when fear spike begins to fade: either crossing back below the upper band,
    # or turning down after being above the band on the prior bar.
    bullish = False
    if pd.notna(up_prev) and pd.notna(up_curr):
        if (prev > up_prev and curr <= up_curr) or (prev > up_prev and curr < prev):
            bullish = True

    if bullish:
        tags.append(BULLISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Final trend tag (volatility tool => neutral trend context)
    tags.append(NEUTRAL_TREND)

    return tags