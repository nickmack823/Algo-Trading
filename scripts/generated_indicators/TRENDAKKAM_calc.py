import numpy as np
import pandas as pd

def TRENDAKKAM(
    df: pd.DataFrame,
    akk_range: int = 100,
    ima_range: int = 1,
    akk_factor: float = 6.0,
    mode: int = 0,
    delta_price: float = 30.0,
    point: float = 1.0,
) -> pd.DataFrame:
    """Compute the TREND AKKAM indicator (TrStop and ATR buffers) using numpy/pandas.
    
    Parameters:
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume']
    - akk_range: ATR period (Wilder's RMA)
    - ima_range: EMA period applied to ATR (period=1 means identity)
    - akk_factor: multiplier for ATR-based stop distance
    - mode: 0 => use ATR*factor; otherwise use delta_price*point constant
    - delta_price: constant price delta (used if mode != 0)
    - point: point size multiplier for delta_price
    
    Returns:
    - DataFrame with columns:
        'TrStop' - primary trailing stop line aligned to df.index
        'ATR'    - Wilder ATR used internally (NaN during warmup)
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame({'TrStop': pd.Series(dtype=float), 'ATR': pd.Series(dtype=float)})

    high = pd.to_numeric(df['High'], errors='coerce')
    low = pd.to_numeric(df['Low'], errors='coerce')
    close = pd.to_numeric(df['Close'], errors='coerce')
    open_ = pd.to_numeric(df['Open'], errors='coerce')

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Wilder's ATR (RMA with alpha = 1/period); NaN until warmup
    atr = tr.ewm(alpha=(1.0 / float(akk_range)), adjust=False, min_periods=akk_range).mean()

    # EMA of ATR on array (iMAOnArray with MODE_EMA). Period 1 => identity.
    if ima_range <= 1:
        ma_atr = atr.copy()
    else:
        ma_atr = atr.ewm(span=ima_range, adjust=False, min_periods=ima_range).mean()

    # DeltaStop series
    if mode == 0:
        delta = ma_atr * float(akk_factor)
    else:
        delta = pd.Series(float(delta_price) * float(point), index=df.index, dtype=float)

    # Recursive computation in reverse to match MQL4 indexing (i from Bars-1 downto 0)
    O = open_.to_numpy(dtype=float)
    D = delta.to_numpy(dtype=float)

    O_rev = O[::-1]
    D_rev = D[::-1]

    tr_rev = np.full(n, np.nan, dtype=float)

    # Seed initial value (no prior state in pure batch context)
    # Use first available Open as neutral seed when both O and D are finite; else NaN.
    if np.isfinite(O_rev[0]) and np.isfinite(D_rev[0]):
        tr_rev[0] = O_rev[0]

    for j in range(1, n):
        prev_T = tr_rev[j - 1]
        prev_O = O_rev[j - 1]
        curr_O = O_rev[j]
        curr_D = D_rev[j]

        if not (np.isfinite(prev_T) and np.isfinite(prev_O) and np.isfinite(curr_O) and np.isfinite(curr_D)):
            tr_rev[j] = np.nan
            continue

        if curr_O == prev_T:
            tr_rev[j] = prev_T
        else:
            if (prev_O < prev_T) and (curr_O < prev_T):
                tr_rev[j] = min(prev_T, curr_O + curr_D)
            elif (prev_O > prev_T) and (curr_O > prev_T):
                tr_rev[j] = max(prev_T, curr_O - curr_D)
            else:
                tr_rev[j] = (curr_O - curr_D) if (curr_O > prev_T) else (curr_O + curr_D)

    trstop = pd.Series(tr_rev[::-1], index=df.index, name='TrStop')

    out = pd.DataFrame({
        'TrStop': trstop,
        'ATR': atr.rename('ATR')
    }, index=df.index)

    return out