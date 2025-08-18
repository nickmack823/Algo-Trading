import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE


def signal_zerolag_macd_mq4(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Validate required columns
    try:
        macd = df["ZL_MACD"]
        sig = df["ZL_Signal"]
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if len(macd) < 2 or len(sig) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    macd_prev = macd.iloc[-2]
    macd_last = macd.iloc[-1]
    sig_prev = sig.iloc[-2]
    sig_last = sig.iloc[-1]

    if pd.isna(macd_prev) or pd.isna(macd_last) or pd.isna(sig_prev) or pd.isna(sig_last):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Cross signals (last two bars)
    bull_cross_sig = macd_prev <= sig_prev and macd_last > sig_last
    bear_cross_sig = macd_prev >= sig_prev and macd_last < sig_last
    bull_cross_zero = macd_prev <= 0.0 and macd_last > 0.0
    bear_cross_zero = macd_prev >= 0.0 and macd_last < 0.0

    if bull_cross_sig or bull_cross_zero:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross_sig or bear_cross_zero:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag (exactly one)
    if macd_last > sig_last and macd_last > 0.0:
        tags.append(BULLISH_TREND)
    elif macd_last < sig_last and macd_last < 0.0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags