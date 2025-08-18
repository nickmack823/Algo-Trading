import numpy as np
import pandas as pd

def StiffnessIndicator(
    df: pd.DataFrame,
    period1: int = 100,
    method1: str | int = "sma",
    period3: int = 60,
    period2: int = 3,
    method2: str | int = "sma",
) -> pd.DataFrame:
    """Return ALL lines as columns ['Stiffness','Signal']; aligned to df.index. Vectorized, handle NaNs.
    
    MQL mapping:
    - period1, method1: MA period/method for primary threshold MA
    - period3: summation period
    - period2, method2: MA on Stiffness (signal)
    
    Methods supported: 'sma', 'ema', 'smma', 'lwma' (or 0,1,2,3 respectively).
    """
    if period1 <= 0 or period2 <= 0 or period3 <= 0:
        raise ValueError("All periods must be positive integers.")
    
    def _norm_method(m):
        if isinstance(m, (int, np.integer)):
            mapping = {0: "sma", 1: "ema", 2: "smma", 3: "lwma"}
            if int(m) not in mapping:
                raise ValueError("Unknown MA method code. Use 0:SMA, 1:EMA, 2:SMMA, 3:LWMA.")
            return mapping[int(m)]
        m = str(m).strip().lower()
        aliases = {
            "sma": "sma",
            "ema": "ema",
            "smma": "smma",
            "rma": "smma",
            "wilder": "smma",
            "lwma": "lwma",
            "wma": "lwma",
        }
        if m not in aliases:
            raise ValueError("Unknown MA method. Use 'sma','ema','smma','lwma' or 0..3.")
        return aliases[m]
    
    def _ma(s: pd.Series, n: int, method: str) -> pd.Series:
        if method == "sma":
            return s.rolling(window=n, min_periods=n).mean()
        elif method == "ema":
            return s.ewm(span=n, adjust=False, min_periods=n).mean()
        elif method == "smma":
            # Welles Wilder smoothing (SMMA/RMA) via EWM alpha=1/n
            return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
        elif method == "lwma":
            w = np.arange(1, n + 1, dtype=float)
            w_sum = w.sum()
            return s.rolling(window=n, min_periods=n).apply(lambda x: np.dot(x, w) / w_sum, raw=True)
        else:
            raise ValueError("Unsupported MA method.")
    
    m1 = _norm_method(method1)
    m2 = _norm_method(method2)
    
    close = pd.to_numeric(df["Close"], errors="coerce")
    
    ma1 = _ma(close, period1, m1)
    stdev = close.rolling(window=period1, min_periods=period1).std(ddof=1)
    temp = ma1 - 0.2 * stdev
    
    # Binary stream: 1 if Close > Temp else 0; NaNs in comparison yield False -> 0
    stream = (close > temp).astype(float)
    
    # Summation over last 'period3' bars (inclusive)
    P = stream.rolling(window=period3, min_periods=period3).sum()
    
    stiffness = P * (period1 / float(period3))
    signal = _ma(stiffness, period2, m2)
    
    out = pd.DataFrame(
        {
            "Stiffness": stiffness.astype(float),
            "Signal": signal.astype(float),
        },
        index=df.index,
    )
    return out