import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_coral(df: pd.DataFrame) -> list[str]:
    # Ensure 'Coral' exists
    if not isinstance(df, pd.DataFrame) or 'Coral' not in df.columns or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]

    coral = df['Coral'].astype(float)
    n = len(coral)
    if n < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    c_curr = coral.iloc[-1]
    c_prev = coral.iloc[-2]

    # Try price-vs-coral confirmation if Close is available; otherwise fall back to coral slope
    signal_tags: list[str] = []
    trend_tag = NEUTRAL_TREND

    if 'Close' in df.columns:
        close = pd.to_numeric(df['Close'], errors='coerce')
        if len(close) >= 2:
            p_curr = close.iloc[-1]
            p_prev = close.iloc[-2]

            # Determine current trend by price relative to coral
            if pd.notna(p_curr) and pd.notna(c_curr):
                if p_curr > c_curr:
                    trend_tag = BULLISH_TREND
                elif p_curr < c_curr:
                    trend_tag = BEARISH_TREND
                else:
                    trend_tag = NEUTRAL_TREND

            # Signals: detect cross on the last bar
            cond_up = pd.notna(p_prev) and pd.notna(c_prev) and pd.notna(p_curr) and pd.notna(c_curr) and (p_prev <= c_prev) and (p_curr > c_curr)
            cond_down = pd.notna(p_prev) and pd.notna(c_prev) and pd.notna(p_curr) and pd.notna(c_curr) and (p_prev >= c_prev) and (p_curr < c_curr)

            if cond_up:
                signal_tags.append(BULLISH_SIGNAL)
            elif cond_down:
                signal_tags.append(BEARISH_SIGNAL)

    # Fallback to coral slope if no price-based signal/trend was determined
    if not signal_tags and trend_tag == NEUTRAL_TREND:
        up_now = pd.notna(c_curr) and pd.notna(c_prev) and (c_curr > c_prev)
        down_now = pd.notna(c_curr) and pd.notna(c_prev) and (c_curr < c_prev)

        if up_now:
            trend_tag = BULLISH_TREND
        elif down_now:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND

        # Optional confirmation signal on slope turn (uses last 3 points if available)
        if n >= 3:
            c_prev2 = coral.iloc[-3]
            if pd.notna(c_prev2) and pd.notna(c_prev):
                up_prev = c_prev > c_prev2
                down_prev = c_prev < c_prev2
                if up_now and not up_prev:
                    signal_tags.append(BULLISH_SIGNAL)
                elif down_now and not down_prev:
                    signal_tags.append(BEARISH_SIGNAL)

    if not signal_tags:
        signal_tags.append(NO_SIGNAL)

    signal_tags.append(trend_tag)
    return signal_tags