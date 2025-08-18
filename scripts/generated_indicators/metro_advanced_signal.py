import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_metro_advanced(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    r_last = df.iloc[-1]

    def is_num(x) -> bool:
        return x is not None and pd.notna(x)

    # If fewer than 2 bars, no cross decisions; try to infer trend, else neutral
    if n < 2:
        trend_tag = NEUTRAL_TREND
        tval = r_last.get("Trend")
        if is_num(tval):
            if tval > 0:
                trend_tag = BULLISH_TREND
            elif tval < 0:
                trend_tag = BEARISH_TREND
        return [NO_SIGNAL, trend_tag]

    r_prev = df.iloc[-2]

    sfast0 = r_prev.get("StepRSI_fast")
    sfast1 = r_last.get("StepRSI_fast")
    sslow0 = r_prev.get("StepRSI_slow")
    sslow1 = r_last.get("StepRSI_slow")

    rsi0 = r_prev.get("RSI")
    rsi1 = r_last.get("RSI")
    lmid0 = r_prev.get("Level_Mid")
    lmid1 = r_last.get("Level_Mid")

    lup0 = r_prev.get("Level_Up")
    lup1 = r_last.get("Level_Up")
    ldn0 = r_prev.get("Level_Dn")
    ldn1 = r_last.get("Level_Dn")

    bull_cross = False
    bear_cross = False

    # Signal-line cross: StepRSI_fast vs StepRSI_slow
    if is_num(sfast0) and is_num(sslow0) and is_num(sfast1) and is_num(sslow1):
        if sfast0 <= sslow0 and sfast1 > sslow1:
            bull_cross = True
        elif sfast0 >= sslow0 and sfast1 < sslow1:
            bear_cross = True

    # Centre-line cross: RSI vs Level_Mid
    if is_num(rsi0) and is_num(lmid0) and is_num(rsi1) and is_num(lmid1):
        if rsi0 <= lmid0 and rsi1 > lmid1:
            bull_cross = True
        elif rsi0 >= lmid0 and rsi1 < lmid1:
            bear_cross = True

    if bull_cross and not bear_cross:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross and not bull_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Overbought / Oversold threshold crosses (last two bars)
    if is_num(rsi0) and is_num(lup0) and is_num(rsi1) and is_num(lup1) and rsi0 <= lup0 and rsi1 > lup1:
        tags.append(OVERBOUGHT)
    elif is_num(rsi0) and is_num(ldn0) and is_num(rsi1) and is_num(ldn1) and rsi0 >= ldn0 and rsi1 < ldn1:
        tags.append(OVERSOLD)

    # Final trend tag (exactly one)
    trend_tag = NEUTRAL_TREND
    tval = r_last.get("Trend")
    if is_num(tval):
        if tval > 0:
            trend_tag = BULLISH_TREND
        elif tval < 0:
            trend_tag = BEARISH_TREND
    else:
        if is_num(sfast1) and is_num(sslow1):
            if sfast1 > sslow1:
                trend_tag = BULLISH_TREND
            elif sfast1 < sslow1:
                trend_tag = BEARISH_TREND
        elif is_num(rsi1) and is_num(lmid1):
            if rsi1 > lmid1:
                trend_tag = BULLISH_TREND
            elif rsi1 < lmid1:
                trend_tag = BEARISH_TREND

    tags.append(trend_tag)
    return tags