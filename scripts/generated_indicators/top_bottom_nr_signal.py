import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_top_bottom_nr(df: pd.DataFrame) -> list[str]:
    # Robustness checks
    if df is None or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    ls = df.get('LongSignal')
    ss = df.get('ShortSignal')
    if ls is None or ss is None or len(ls) < 2 or len(ss) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    l_prev, l_curr = ls.iloc[-2], ls.iloc[-1]
    s_prev, s_curr = ss.iloc[-2], ss.iloc[-1]

    valid_l = pd.notna(l_prev) and pd.notna(l_curr)
    valid_s = pd.notna(s_prev) and pd.notna(s_curr)

    if not (valid_l or valid_s):
        return [NO_SIGNAL, NEUTRAL_TREND]

    reset_long = bool(valid_l and (l_curr < l_prev))
    reset_short = bool(valid_s and (s_curr < s_prev))

    tags: list[str] = []
    if reset_long or reset_short:
        tags.append(INCREASING_VOLATILITY)
    else:
        tags.append(DECREASING_VOLATILITY)

    tags.append(NEUTRAL_TREND)
    return tags