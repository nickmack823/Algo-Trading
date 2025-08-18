import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_dorsey_inertia(series: pd.Series) -> list[str]:
    # Robustness to insufficient data/NaNs
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]
    last = series.iloc[-1]
    prev = series.iloc[-2]
    if pd.isna(last) or pd.isna(prev):
        return [NO_SIGNAL, NEUTRAL_TREND]

    threshold = 40.0
    trending_now = float(last) >= threshold
    trending_prev = float(prev) >= threshold

    tags: list[str] = []

    # Cross/threshold decisions using last two bars
    if not trending_prev and trending_now:
        # Entering a trending regime, direction to be determined by other tools
        tags.append(INCONCLUSIVE)
    elif trending_prev and not trending_now:
        # Leaving a trending regime
        tags.append(NO_SIGNAL)
    else:
        # Steady state
        if trending_now:
            tags.append(INCONCLUSIVE)  # Trending but direction unknown
        else:
            tags.append(NO_SIGNAL)  # Range-bound

    # Direction is not determined by Inertia alone; always neutral trend tag
    tags.append(NEUTRAL_TREND)
    return tags