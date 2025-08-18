import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_dpo_histogram_indicator(df: pd.DataFrame) -> list[str]:
    # Fallbacks for invalid input
    if not isinstance(df, pd.DataFrame) or df.empty or 'DPO' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(df['DPO'], errors='coerce')

    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Last two values
    v0 = s.iloc[-2]
    v1 = s.iloc[-1]

    # Rising/contracting histogram magnitude -> volatility trend
    abs_s = s.abs()
    abs0 = abs_s.iloc[-2] if pd.notna(v0) else pd.NA
    abs1 = abs_s.iloc[-1] if pd.notna(v1) else pd.NA

    inc = pd.notna(abs0) and pd.notna(abs1) and (abs1 > abs0)
    dec = pd.notna(abs0) and pd.notna(abs1) and (abs1 < abs0)

    # Volatility regime via rolling quantiles of |DPO|
    q_win = 20
    q_low = abs_s.rolling(window=q_win, min_periods=5).quantile(0.2)
    q_high = abs_s.rolling(window=q_win, min_periods=5).quantile(0.8)

    ql = q_low.iloc[-1] if len(q_low) else pd.NA
    qh = q_high.iloc[-1] if len(q_high) else pd.NA

    low_vol = pd.notna(abs1) and pd.notna(ql) and (abs1 <= ql)
    high_vol = pd.notna(abs1) and pd.notna(qh) and (abs1 >= qh)

    tags: list[str] = [NO_SIGNAL]

    if low_vol:
        tags.append(LOW_VOLATILITY)
    elif high_vol:
        tags.append(HIGH_VOLATILITY)

    if inc:
        tags.append(INCREASING_VOLATILITY)
    elif dec:
        tags.append(DECREASING_VOLATILITY)
    else:
        tags.append(STABLE_VOLATILITY)

    tags.append(NEUTRAL_TREND)
    return tags