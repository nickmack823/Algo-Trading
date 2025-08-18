import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_ttf(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty or "TTF" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    ttf = df["TTF"].astype(float)
    if len(ttf) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    curr = ttf.iloc[-1]
    prev = ttf.iloc[-2]

    # Infer overbought/oversold thresholds from the provided 'Signal' column if available
    upper = None
    lower = None
    if "Signal" in df.columns:
        sig_abs = df["Signal"].dropna().abs()
        if not sig_abs.empty:
            thresh = float(sig_abs.max())
            if pd.notna(thresh) and thresh > 0:
                upper = thresh
                lower = -thresh

    # Zero-line cross signals
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0.0 and curr > 0.0:
            tags.append(BULLISH_SIGNAL)
        elif prev >= 0.0 and curr < 0.0:
            tags.append(BEARISH_SIGNAL)

        # Overbought/Oversold crosses (optional timing tags)
        if upper is not None and prev < upper and curr >= upper:
            tags.append(OVERBOUGHT)
        if lower is not None and prev > lower and curr <= lower:
            tags.append(OVERSOLD)

    # If no primary signal, explicitly add NO_SIGNAL
    if not any(tag in (BULLISH_SIGNAL, BEARISH_SIGNAL) for tag in tags):
        tags.append(NO_SIGNAL)

    # Final trend tag (exactly one)
    if pd.notna(curr):
        if curr > 0.0:
            tags.append(BULLISH_TREND)
        elif curr < 0.0:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags