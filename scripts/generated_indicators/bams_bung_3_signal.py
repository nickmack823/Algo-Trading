import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_bams_bung_3(df: pd.DataFrame) -> list[str]:
    # Robustness checks
    try:
        up_sig = df["UpTrendSignal"]
        dn_sig = df["DownTrendSignal"]
        up_stop = df["UpTrendStop"]
        dn_stop = df["DownTrendStop"]
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    if n < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    def is_active_signal(x) -> bool:
        return pd.notna(x) and x not in (float("inf"), float("-inf")) and x != -1.0

    def is_active_stop(x) -> bool:
        # Active stop when it's defined and not the inactive sentinel -1.0
        return pd.notna(x) and x not in (float("inf"), float("-inf")) and x != -1.0

    # Signal detection (stars fire only on the activation bar)
    last_up_sig = up_sig.iat[-1]
    prev_up_sig = up_sig.iat[-2]
    last_dn_sig = dn_sig.iat[-1]
    prev_dn_sig = dn_sig.iat[-2]

    star_up = is_active_signal(last_up_sig) and not is_active_signal(prev_up_sig)
    star_dn = is_active_signal(last_dn_sig) and not is_active_signal(prev_dn_sig)

    tags: list[str] = []
    if star_up and not star_dn:
        tags.append(BULLISH_SIGNAL)
    elif star_dn and not star_up:
        tags.append(BEARISH_SIGNAL)
    elif star_up and star_dn:
        tags.append(INCONCLUSIVE)
    else:
        tags.append(NO_SIGNAL)

    # Trend determination from stops
    last_up_stop = up_stop.iat[-1]
    last_dn_stop = dn_stop.iat[-1]
    up_active = is_active_stop(last_up_stop)
    dn_active = is_active_stop(last_dn_stop)

    if up_active and not dn_active:
        trend_tag = BULLISH_TREND
    elif dn_active and not up_active:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags