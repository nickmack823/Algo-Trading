import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_ehlers_early_onset_trend(df: pd.DataFrame) -> list[str]:
    # Robustness to insufficient data
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Extract last two bars
    prev = df.iloc[-2]
    last = df.iloc[-1]

    q1_prev = prev.get("EEOT_Q1", float("nan"))
    q1_last = last.get("EEOT_Q1", float("nan"))
    q2_prev = prev.get("EEOT_Q2", float("nan"))
    q2_last = last.get("EEOT_Q2", float("nan"))

    def is_num(x) -> bool:
        return pd.notna(x)

    def is_pos(x) -> bool:
        return is_num(x) and x > 0

    def is_neg(x) -> bool:
        return is_num(x) and x < 0

    # Slopes (last - prev)
    q1_slope = q1_last - q1_prev if is_num(q1_last) and is_num(q1_prev) else float("nan")
    q2_slope = q2_last - q2_prev if is_num(q2_last) and is_num(q2_prev) else float("nan")

    def rising(x) -> bool:
        return is_num(x) and x > 0

    def falling(x) -> bool:
        return is_num(x) and x < 0

    # Crosses around zero
    q1_cross_up = is_neg(q1_prev) and is_num(q1_last) and q1_last >= 0
    q2_cross_up = is_neg(q2_prev) and is_num(q2_last) and q2_last >= 0
    q1_cross_down = is_pos(q1_prev) and is_num(q1_last) and q1_last <= 0
    q2_cross_down = is_pos(q2_prev) and is_num(q2_last) and q2_last <= 0

    cross_up = q1_cross_up or q2_cross_up
    cross_down = q1_cross_down or q2_cross_down

    both_rising = rising(q1_slope) and rising(q2_slope)
    both_falling = falling(q1_slope) and falling(q2_slope)
    any_above = is_pos(q1_last) or is_pos(q2_last)
    any_below = is_neg(q1_last) or is_neg(q2_last)

    tags: list[str] = []

    if cross_up or (both_rising and any_above):
        tags.append(BULLISH_SIGNAL)
    elif cross_down or (both_falling and any_below):
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend assessment using last values and slopes
    score = 0
    if is_pos(q1_last):
        score += 1
    elif is_neg(q1_last):
        score -= 1

    if is_pos(q2_last):
        score += 1
    elif is_neg(q2_last):
        score -= 1

    if rising(q1_slope):
        score += 1
    elif falling(q1_slope):
        score -= 1

    if rising(q2_slope):
        score += 1
    elif falling(q2_slope):
        score -= 1

    if score >= 2:
        tags.append(BULLISH_TREND)
    elif score <= -2:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags