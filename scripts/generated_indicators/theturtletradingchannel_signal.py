import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_theturtletradingchannel(df: pd.DataFrame) -> list[str]:
    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    bull = df["BullChange"] if "BullChange" in df.columns else pd.Series([pd.NA] * n, index=df.index)
    bear = df["BearChange"] if "BearChange" in df.columns else pd.Series([pd.NA] * n, index=df.index)
    upper = df["UpperLine"] if "UpperLine" in df.columns else pd.Series([pd.NA] * n, index=df.index)
    lower = df["LowerLine"] if "LowerLine" in df.columns else pd.Series([pd.NA] * n, index=df.index)

    # Last two bars logic
    bull_now = pd.notna(bull.iloc[-1])
    bear_now = pd.notna(bear.iloc[-1])
    bull_prev = pd.notna(bull.iloc[-2]) if n >= 2 else False
    bear_prev = pd.notna(bear.iloc[-2]) if n >= 2 else False

    bull_signal = bool(bull_now and not bull_prev)
    bear_signal = bool(bear_now and not bear_prev)

    tags: list[str] = []
    if bull_signal:
        tags.append(BULLISH_SIGNAL)
    elif bear_signal:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend assessment from current regime lines
    if pd.notna(upper.iloc[-1]):
        trend_tag = BULLISH_TREND
    elif pd.notna(lower.iloc[-1]):
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags