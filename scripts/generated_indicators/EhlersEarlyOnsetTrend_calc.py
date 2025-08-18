import numpy as np
import pandas as pd

def EhlersEarlyOnsetTrend(df: pd.DataFrame, period: int = 20, q1: float = 0.8, q2: float = 0.4) -> pd.DataFrame:
    """Return both Ehlers Early Onset Trend lines (for q1 and q2) as columns aligned to df.index."""
    close = df["Close"].to_numpy(dtype=float)
    n = close.shape[0]

    PI = np.pi
    angle = 0.707 * 2.0 * PI / 100.0
    alpha1 = (np.cos(angle) + np.sin(angle) - 1.0) / np.cos(angle)

    def _quotient(close_arr: np.ndarray, lp_period: int, k: float, alpha: float) -> np.ndarray:
        a1 = np.exp(-1.414 * PI / lp_period)
        b1 = 2.0 * a1 * np.cos(1.414 * PI / lp_period)
        c3 = -a1 * a1
        c1 = 1.0 - b1 - c3

        hp = np.zeros(n, dtype=float)
        filt = np.zeros(n, dtype=float)
        pk = np.zeros(n, dtype=float)
        out = np.full(n, np.nan, dtype=float)

        decay = 0.991
        one_minus_a = (1.0 - alpha)
        coef_hp0 = (1.0 - alpha / 2.0) ** 2
        coef_hp1 = 2.0 * one_minus_a
        coef_hp2 = one_minus_a ** 2

        for t in range(n):
            if t < 2:
                continue
            c0, c1c, c2 = close_arr[t], close_arr[t - 1], close_arr[t - 2]
            if np.isnan(c0) or np.isnan(c1c) or np.isnan(c2):
                # carry states forward with decay on peak, but mark output as NaN
                hp[t] = hp[t - 1]
                filt[t] = filt[t - 1]
                pk[t] = decay * pk[t - 1]
                out[t] = np.nan
                continue

            hp[t] = coef_hp0 * (c0 - 2.0 * c1c + c2) + coef_hp1 * hp[t - 1] - coef_hp2 * hp[t - 2]
            filt[t] = c1 * (hp[t] + hp[t - 1]) / 2.0 + b1 * filt[t - 1] + c3 * filt[t - 2]
            pk[t] = max(abs(filt[t]), decay * pk[t - 1])
            x = 0.0 if pk[t] == 0.0 else (filt[t] / pk[t])
            out[t] = (x + k) / (k * x + 1.0)

        return out

    out_q1 = _quotient(close, period, q1, alpha1)
    out_q2 = _quotient(close, period, q2, alpha1)

    return pd.DataFrame(
        {
            f"EEOT_Q1": out_q1,
            f"EEOT_Q2": out_q2,
        },
        index=df.index,
    )