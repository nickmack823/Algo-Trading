import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_sherif_hilo(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Ensure required columns exist; if not, treat as inconclusive/neutral
    required_cols = {'LineUp', 'LineDown'}
    if not required_cols.issubset(df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    is_up = df['LineUp'].notna()
    is_down = df['LineDown'].notna()

    n = len(df)
    if n < 2:
        # Not enough bars to detect a cross
        # Trend: derive from most recent line presence if any
        up_last_idx = df['LineUp'].last_valid_index()
        down_last_idx = df['LineDown'].last_valid_index()

        pos_up = df.index.get_loc(up_last_idx) if up_last_idx is not None else -1
        pos_down = df.index.get_loc(down_last_idx) if down_last_idx is not None else -1

        tags.append(NO_SIGNAL)
        if pos_up > pos_down:
            tags.append(BULLISH_TREND)
        elif pos_down > pos_up:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
        return tags

    up_last = bool(is_up.iat[-1])
    up_prev = bool(is_up.iat[-2])
    down_last = bool(is_down.iat[-1])
    down_prev = bool(is_down.iat[-2])

    bullish_cross = up_last and not up_prev
    bearish_cross = down_last and not down_prev

    if bullish_cross and not bearish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross and not bullish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend from most recent non-NaN of LineUp/LineDown across the series
    up_last_idx = df['LineUp'].last_valid_index()
    down_last_idx = df['LineDown'].last_valid_index()

    pos_up = df.index.get_loc(up_last_idx) if up_last_idx is not None else -1
    pos_down = df.index.get_loc(down_last_idx) if down_last_idx is not None else -1

    if pos_up > pos_down:
        tags.append(BULLISH_TREND)
    elif pos_down > pos_up:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags