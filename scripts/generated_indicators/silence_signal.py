import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_silence(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty or 'Volatility' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    vol = pd.to_numeric(df['Volatility'], errors='coerce')

    # Require the last bar to be valid
    if len(vol) == 0 or pd.isna(vol.iloc[-1]):
        return [NO_SIGNAL, NEUTRAL_TREND]

    curr = float(vol.iloc[-1])
    prev = float(vol.iloc[-2]) if len(vol) >= 2 and not pd.isna(vol.iloc[-2]) else None

    # Thresholds on 0..100 (reversed) scale: higher => quieter, lower => more active
    low_vol_threshold = 70.0   # quiet market
    high_vol_threshold = 30.0  # active market
    epsilon = 1.0              # change sensitivity

    # State tags (filter)
    if curr >= low_vol_threshold:
        tags.append(LOW_VOLATILITY)
    elif curr <= high_vol_threshold:
        tags.append(HIGH_VOLATILITY)
    else:
        tags.append(NO_SIGNAL)

    # Change in volatility (consider last two bars)
    if prev is not None:
        if curr < prev - epsilon:
            tags.append(INCREASING_VOLATILITY)   # reversed scale: down => vol expanding
        elif curr > prev + epsilon:
            tags.append(DECREASING_VOLATILITY)   # reversed scale: up => vol contracting
        else:
            tags.append(STABLE_VOLATILITY)
    else:
        tags.append(INCONCLUSIVE)

    # This indicator is a filter; trend is neutral
    tags.append(NEUTRAL_TREND)

    return tags