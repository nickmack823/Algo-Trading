import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_ergodic_tvi(df: pd.DataFrame) -> list[str]:
    # Expect df to contain columns: "ETVI", "Signal"
    if df is None or df.empty or len(df.index) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    cols = ["ETVI", "Signal"]
    if not all(col in df.columns for col in cols):
        return [NO_SIGNAL, NEUTRAL_TREND]

    last2 = df[cols].tail(2)
    e_prev, e_curr = last2["ETVI"].iloc[0], last2["ETVI"].iloc[1]
    s_prev, s_curr = last2["Signal"].iloc[0], last2["Signal"].iloc[1]

    have_sig = pd.notna(e_prev) and pd.notna(e_curr) and pd.notna(s_prev) and pd.notna(s_curr)
    have_zero = pd.notna(e_prev) and pd.notna(e_curr)

    tags: list[str] = []

    # Primary signal: cross of ETVI vs Signal line; Secondary: cross of ETVI vs zero line
    if have_sig and (e_prev <= s_prev and e_curr > s_curr):
        tags.append(BULLISH_SIGNAL)
    elif have_sig and (e_prev >= s_prev and e_curr < s_curr):
        tags.append(BEARISH_SIGNAL)
    elif have_zero and (e_prev <= 0 and e_curr > 0):
        tags.append(BULLISH_SIGNAL)
    elif have_zero and (e_prev >= 0 and e_curr < 0):
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend determination (use ETVI vs Signal if available; else ETVI vs zero; else neutral)
    if pd.notna(e_curr) and pd.notna(s_curr):
        if e_curr > s_curr:
            tags.append(BULLISH_TREND)
        elif e_curr < s_curr:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    elif pd.notna(e_curr):
        if e_curr > 0:
            tags.append(BULLISH_TREND)
        elif e_curr < 0:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags