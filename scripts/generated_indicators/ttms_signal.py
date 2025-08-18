import pandas as pd
from scripts.config import (
    BULLISH_SIGNAL,
    BEARISH_SIGNAL,
    BULLISH_TREND,
    BEARISH_TREND,
    NEUTRAL_TREND,
    OVERBOUGHT,
    OVERSOLD,
    NO_SIGNAL,
    HIGH_VOLUME,
    LOW_VOLUME,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    INCREASING_VOLATILITY,
    DECREASING_VOLATILITY,
    STABLE_VOLATILITY,
    INCONCLUSIVE,
)


def signal_ttms(df: pd.DataFrame) -> list[str]:
    # Validate input
    try:
        up = df["TTMS_Up"]
        dn = df["TTMS_Dn"]
        alert = df["Alert"]
        noalert = df["NoAlert"]
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    up2 = up.iloc[-2:]
    dn2 = dn.iloc[-2:]
    alert2 = alert.iloc[-2:]
    noalert2 = noalert.iloc[-2:]

    def sum_ignore_nan(a, b):
        a_valid = pd.notna(a)
        b_valid = pd.notna(b)
        if not a_valid and not b_valid:
            return float("nan")
        s = 0.0
        if a_valid:
            s += float(a)
        if b_valid:
            s += float(b)
        return s

    t_prev = sum_ignore_nan(up2.iloc[0], dn2.iloc[0])
    t_now = sum_ignore_nan(up2.iloc[1], dn2.iloc[1])

    prev_on = pd.notna(alert2.iloc[0])
    prev_off = pd.notna(noalert2.iloc[0])
    now_on = pd.notna(alert2.iloc[1])
    now_off = pd.notna(noalert2.iloc[1])

    # If latest bar has no valid squeeze state, return neutral
    if not (now_on or now_off):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = [NO_SIGNAL]

    # Current volatility state
    if now_on:
        tags.append(LOW_VOLATILITY)
    elif now_off:
        tags.append(HIGH_VOLATILITY)

    # Transition-based volatility change
    transition = None
    if prev_on or prev_off:
        if now_off and prev_on:
            transition = "inc"  # squeeze released -> increasing vol
        elif now_on and prev_off:
            transition = "dec"  # squeeze engaged -> decreasing vol

    eps = 1e-12
    if transition == "inc":
        tags.append(INCREASING_VOLATILITY)
    elif transition == "dec":
        tags.append(DECREASING_VOLATILITY)
    else:
        # Slope-based volatility change if both values are valid
        if pd.notna(t_prev) and pd.notna(t_now):
            delta = t_now - t_prev
            if abs(delta) <= eps:
                tags.append(STABLE_VOLATILITY)
            else:
                if now_on:
                    tags.append(DECREASING_VOLATILITY if delta > 0 else INCREASING_VOLATILITY)
                else:
                    tags.append(INCREASING_VOLATILITY if delta < 0 else DECREASING_VOLATILITY)

    # This indicator is volatility-focused; no directional bias
    tags.append(NEUTRAL_TREND)
    return tags