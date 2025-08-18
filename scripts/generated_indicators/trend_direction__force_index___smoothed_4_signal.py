import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_trend_direction__force_index___smoothed_4(df: pd.DataFrame) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2 or "TDF" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    tdf = df["TDF"]
    tu = df["TriggerUp"] if "TriggerUp" in df.columns else pd.Series([0.0] * len(df), index=df.index)
    td = df["TriggerDown"] if "TriggerDown" in df.columns else pd.Series([0.0] * len(df), index=df.index)

    prev_tdf = tdf.iloc[-2]
    curr_tdf = tdf.iloc[-1]
    prev_tu = tu.iloc[-2] if len(tu) >= 2 else tu.iloc[-1]
    curr_tu = tu.iloc[-1]
    prev_td = td.iloc[-2] if len(td) >= 2 else td.iloc[-1]
    curr_td = td.iloc[-1]

    def is_num(x) -> bool:
        return x is not None and pd.notna(x)

    tags: list[str] = []

    bullish_cross = False
    bearish_cross = False
    if all(is_num(x) for x in [prev_tdf, curr_tdf, prev_tu, curr_tu]):
        bullish_cross = (prev_tdf <= prev_tu) and (curr_tdf > curr_tu)
    if all(is_num(x) for x in [prev_tdf, curr_tdf, prev_td, curr_td]):
        bearish_cross = (prev_tdf >= prev_td) and (curr_tdf < curr_td)

    if bullish_cross and not bearish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross and not bullish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    trend_tag = NEUTRAL_TREND
    if "Trend" in df.columns and is_num(df["Trend"].iloc[-1]):
        t = df["Trend"].iloc[-1]
        if t > 0:
            trend_tag = BULLISH_TREND
        elif t < 0:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        if is_num(curr_tdf) and is_num(curr_tu) and is_num(curr_td):
            if curr_tdf > curr_tu:
                trend_tag = BULLISH_TREND
            elif curr_tdf < curr_td:
                trend_tag = BEARISH_TREND

    tags.append(trend_tag)
    return tags