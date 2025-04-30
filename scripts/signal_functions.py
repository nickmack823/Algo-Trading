import numpy as np
import pandas as pd

from scripts.config import (
    BEARISH_SIGNAL,
    BEARISH_TREND,
    BULLISH_SIGNAL,
    BULLISH_TREND,
    DECREASING_VOLATILITY,
    HIGH_VOLATILITY,
    HIGH_VOLUME,
    INCONCLUSIVE,
    INCREASING_VOLATILITY,
    LOW_VOLATILITY,
    LOW_VOLUME,
    NEUTRAL_TREND,
    NO_SIGNAL,
    OVERBOUGHT,
    OVERSOLD,
    STABLE_VOLATILITY,
)

# --- Baseline Signal Functions ---


def signal_baseline_standard(
    baseline_series: pd.Series, close_series: pd.Series
) -> list:
    signals = []
    prev_close, curr_close = close_series.iloc[-2], close_series.iloc[-1]
    prev_baseline, curr_baseline = baseline_series.iloc[-2], baseline_series.iloc[-1]
    if prev_close < prev_baseline and curr_close > curr_baseline:
        signals.append(BULLISH_SIGNAL)
    elif prev_close > prev_baseline and curr_close < curr_baseline:
        signals.append(BEARISH_SIGNAL)

    if curr_close > curr_baseline:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_baseline:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_recursive_ma(recursive_df: pd.DataFrame, close_series: pd.Series) -> list:
    signals = []
    c_prev, c_curr = close_series.iloc[-2], close_series.iloc[-1]
    x_prev, x_curr = recursive_df["Xema"].iloc[-2], recursive_df["Xema"].iloc[-1]

    if c_prev < x_prev and c_curr > x_curr:
        signals.append(BULLISH_SIGNAL)
    elif c_prev > x_prev and c_curr < x_curr:
        signals.append(BEARISH_SIGNAL)

    if c_curr > x_curr:
        signals.append(BULLISH_TREND)
    elif c_curr < x_curr:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_lsma(lsma_series: pd.Series, close_series: pd.Series) -> list:
    signals = []
    lsma_now, lsma_prev = lsma_series.iloc[-1], lsma_series.iloc[-2]
    close_now, close_prev = close_series.iloc[-1], close_series.iloc[-2]

    if close_prev < lsma_prev and close_now > lsma_now:
        signals.append(BULLISH_SIGNAL)
    elif close_prev > lsma_prev and close_now < lsma_now:
        signals.append(BEARISH_SIGNAL)

    if close_now > lsma_now:
        signals.append(BULLISH_TREND)
    elif close_now < lsma_now:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_gen3ma(gen3_df: pd.DataFrame, close_series: pd.Series) -> list:
    signals = []
    c_prev, c_curr = close_series.iloc[-2], close_series.iloc[-1]
    g_prev, g_curr = gen3_df["MA3G"].iloc[-2], gen3_df["MA3G"].iloc[-1]

    if c_prev < g_prev and c_curr > g_curr:
        signals.append(BULLISH_SIGNAL)
    elif c_prev > g_prev and c_curr < g_curr:
        signals.append(BEARISH_SIGNAL)

    if c_curr > g_curr:
        signals.append(BULLISH_TREND)
    elif c_curr < g_curr:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_trendlord(trendlord_df: pd.DataFrame) -> list:
    signals = []
    main = trendlord_df["Main"]
    signal = trendlord_df["Signal"]

    if main.iloc[-2] < signal.iloc[-2] and main.iloc[-1] > signal.iloc[-1]:
        signals.append(BULLISH_SIGNAL)
    elif main.iloc[-2] > signal.iloc[-2] and main.iloc[-1] < signal.iloc[-1]:
        signals.append(BEARISH_SIGNAL)

    if main.iloc[-1] > signal.iloc[-1]:
        signals.append(BULLISH_TREND)
    elif main.iloc[-1] < signal.iloc[-1]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_braidfilter(df: pd.DataFrame) -> list:
    signals = []

    if df["CrossUp"].iloc[-1] == 1:
        signals.append(BULLISH_SIGNAL)
        signals.append(BULLISH_TREND)
    elif df["CrossDown"].iloc[-1] == -1:
        signals.append(BEARISH_SIGNAL)
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_accelerator_lsma(df: pd.DataFrame) -> list:
    signals = []
    current = df["ExtBuffer0"].iloc[-1]
    prev = df["ExtBuffer0"].iloc[-2]

    if prev < 0 and current > 0:
        signals.append(BULLISH_SIGNAL)
        signals.append(BULLISH_TREND)
    elif prev > 0 and current < 0:
        signals.append(BEARISH_SIGNAL)
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


# ---  Momentum Signal Functions ---


def signal_apo(apo_series: pd.Series) -> list:
    signals = []
    if apo_series.iloc[-2] < 0 and apo_series.iloc[-1] > 0:
        signals.append(BULLISH_SIGNAL)
    elif apo_series.iloc[-2] > 0 and apo_series.iloc[-1] < 0:
        signals.append(BEARISH_SIGNAL)

    if apo_series.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif apo_series.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_macd(macd_df: pd.DataFrame) -> list:
    """
    Generate signals from MACD, Signal Line, and Histogram.

    Args:
        macd_df (pd.DataFrame): DataFrame containing MACD, Signal Line, and Histogram

    Returns:
        list: [BULLISH_CROSSOVER, BULLISH_TREND] etc.
    """
    macd_line, signal_line, hist = (
        macd_df.iloc[:, 0],
        macd_df.iloc[:, 1],
        macd_df.iloc[:, 2],
    )
    signals = []

    # Crossover entry signals
    if (
        macd_line.iloc[-2] < signal_line.iloc[-2]
        and macd_line[-1] > signal_line.iloc[-1]
    ):
        signals.append(BULLISH_SIGNAL)
    elif (
        macd_line.iloc[-2] > signal_line.iloc[-2]
        and macd_line[-1] < signal_line.iloc[-1]
    ):
        signals.append(BEARISH_SIGNAL)

    # Trend bias
    if hist.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif hist.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_macdext(macdext_df: pd.DataFrame) -> list:
    macd_line, signal_line, hist = (
        macdext_df.iloc[:, 0],
        macdext_df.iloc[:, 1],
        macdext_df.iloc[:, 2],
    )
    signals = []

    if (
        macd_line.iloc[-2] < signal_line.iloc[-2]
        and macd_line.iloc[-1] > signal_line.iloc[-1]
    ):
        signals.append(BULLISH_SIGNAL)
    elif (
        macd_line.iloc[-2] > signal_line.iloc[-2]
        and macd_line.iloc[-1] < signal_line.iloc[-1]
    ):
        signals.append(BEARISH_SIGNAL)

    if hist.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif hist.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_macdfix(macdfix_df: pd.DataFrame) -> list:
    macd_line, signal_line, hist = (
        macdfix_df.iloc[:, 0],
        macdfix_df.iloc[:, 1],
        macdfix_df.iloc[:, 2],
    )
    signals = []

    if (
        macd_line.iloc[-2] < signal_line.iloc[-2]
        and macd_line.iloc[-1] > signal_line.iloc[-1]
    ):
        signals.append(BULLISH_SIGNAL)
    elif (
        macd_line.iloc[-2] > signal_line.iloc[-2]
        and macd_line.iloc[-1] < signal_line.iloc[-1]
    ):
        signals.append(BEARISH_SIGNAL)

    if hist.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif hist.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_mama(mama_df: pd.DataFrame) -> list:
    mama, fama = mama_df.iloc[:, 0], mama_df.iloc[:, 1]
    signals = []

    if mama.iloc[-2] < fama.iloc[-2] and mama.iloc[-1] > fama.iloc[-1]:
        signals.append(BULLISH_SIGNAL)
    elif mama.iloc[-2] > fama.iloc[-2] and mama.iloc[-1] < fama.iloc[-1]:
        signals.append(BEARISH_SIGNAL)

    if mama.iloc[-1] > fama.iloc[-1]:
        signals.append(BULLISH_TREND)
    elif mama.iloc[-1] < fama.iloc[-1]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_ppo(ppo_series: pd.Series) -> list:
    signals = []

    if ppo_series.iloc[-2] < 0 and ppo_series.iloc[-1] > 0:
        signals.append(BULLISH_SIGNAL)
    elif ppo_series.iloc[-2] > 0 and ppo_series.iloc[-1] < 0:
        signals.append(BEARISH_SIGNAL)

    if ppo_series.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif ppo_series.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_roc(roc_series: pd.Series) -> list:
    signals = []

    if roc_series.iloc[-2] < 0 and roc_series.iloc[-1] > 0:
        signals.append(BULLISH_SIGNAL)
    elif roc_series.iloc[-2] > 0 and roc_series.iloc[-1] < 0:
        signals.append(BEARISH_SIGNAL)

    if roc_series.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif roc_series.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_stoch(stoch_df: pd.DataFrame, overbought=80, oversold=20) -> list:
    k, d = stoch_df.iloc[:, 0], stoch_df.iloc[:, 1]
    signals = []

    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1]:
        signals.append(BULLISH_SIGNAL)
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1]:
        signals.append(BEARISH_SIGNAL)

    if k.iloc[-1] > overbought:
        signals.append(OVERBOUGHT)
    elif k.iloc[-1] < oversold:
        signals.append(OVERSOLD)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_rsi(rsi_series: pd.Series) -> list:
    signals = []

    if rsi_series.iloc[-2] < 50 and rsi_series.iloc[-1] > 50:
        signals.append(BULLISH_SIGNAL)
    elif rsi_series.iloc[-2] > 50 and rsi_series.iloc[-1] < 50:
        signals.append(BEARISH_SIGNAL)

    if rsi_series.iloc[-1] > 50:
        signals.append(BULLISH_TREND)
    elif rsi_series.iloc[-1] < 50:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_cci(cci_series: pd.Series) -> list:
    signals = []
    if cci_series.iloc[-2] < 0 and cci_series.iloc[-1] > 0:
        signals.append(BULLISH_SIGNAL)
    elif cci_series.iloc[-2] > 0 and cci_series.iloc[-1] < 0:
        signals.append(BEARISH_SIGNAL)

    if cci_series.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif cci_series.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_rocp(rocp_series: pd.Series) -> list:
    signals = []
    if rocp_series.iloc[-2] < 0 and rocp_series.iloc[-1] > 0:
        signals.append(BULLISH_SIGNAL)
    elif rocp_series.iloc[-2] > 0 and rocp_series.iloc[-1] < 0:
        signals.append(BEARISH_SIGNAL)

    if rocp_series.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif rocp_series.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_rocr(rocr_series: pd.Series) -> list:
    signals = []
    if rocr_series.iloc[-2] < 1 and rocr_series.iloc[-1] > 1:
        signals.append(BULLISH_SIGNAL)
    elif rocr_series.iloc[-2] > 1 and rocr_series.iloc[-1] < 1:
        signals.append(BEARISH_SIGNAL)

    if rocr_series.iloc[-1] > 1:
        signals.append(BULLISH_TREND)
    elif rocr_series.iloc[-1] < 1:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_rocr100(rocr100_series: pd.Series) -> list:
    signals = []
    if rocr100_series.iloc[-2] < 100 and rocr100_series.iloc[-1] > 100:
        signals.append(BULLISH_SIGNAL)
    elif rocr100_series.iloc[-2] > 100 and rocr100_series.iloc[-1] < 100:
        signals.append(BEARISH_SIGNAL)

    if rocr100_series.iloc[-1] > 100:
        signals.append(BULLISH_TREND)
    elif rocr100_series.iloc[-1] < 100:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_rsi(rsi_series: pd.Series, overbought=70, oversold=30) -> list:
    signals = []

    if rsi_series.iloc[-2] < oversold and rsi_series.iloc[-1] > oversold:
        signals.append(BULLISH_SIGNAL)
    elif rsi_series.iloc[-2] > overbought and rsi_series.iloc[-1] < overbought:
        signals.append(BEARISH_SIGNAL)

    if rsi_series.iloc[-1] > overbought:
        signals.append(OVERBOUGHT)
    elif rsi_series.iloc[-1] < oversold:
        signals.append(OVERSOLD)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_tsf(tsf_series: pd.Series) -> list:
    signals = []
    if tsf_series.iloc[-1] > tsf_series.iloc[-2]:
        signals.append(BULLISH_TREND)
    elif tsf_series.iloc[-1] < tsf_series.iloc[-2]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_ultosc(ultosc_series: pd.Series, overbought=70, oversold=30) -> list:
    signals = []

    if ultosc_series.iloc[-2] < oversold and ultosc_series.iloc[-1] > oversold:
        signals.append(BULLISH_SIGNAL)
    elif ultosc_series.iloc[-2] > overbought and ultosc_series.iloc[-1] < overbought:
        signals.append(BEARISH_SIGNAL)

    if ultosc_series.iloc[-1] > overbought:
        signals.append(OVERBOUGHT)
    elif ultosc_series.iloc[-1] < oversold:
        signals.append(OVERSOLD)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_macd_zero_lag(macd_df: pd.DataFrame) -> list:
    signals = []
    macd, signal = macd_df["MACD"], macd_df["SIGNAL"]

    if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        signals.append(BULLISH_SIGNAL)
    elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        signals.append(BEARISH_SIGNAL)

    if macd.iloc[-1] > signal.iloc[-1]:
        signals.append(BULLISH_TREND)
    elif macd.iloc[-1] < signal.iloc[-1]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_fisher(fisher_df: pd.DataFrame) -> list:
    signals = []
    f0, f1 = fisher_df["Fisher"].iloc[-2], fisher_df["Fisher"].iloc[-1]

    if f0 < 0 and f1 > 0:
        signals.append(BULLISH_SIGNAL)
    elif f0 > 0 and f1 < 0:
        signals.append(BEARISH_SIGNAL)

    if f1 > 0:
        signals.append(BULLISH_TREND)
    elif f1 < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_bulls_bears_impulse(impulse_df: pd.DataFrame) -> list:
    signals = []
    impulse = impulse_df["Impulse"]

    if impulse.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif impulse.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_j_tpo(jtpo_series: pd.Series) -> list:
    signals = []

    if jtpo_series.iloc[-2] < 0 and jtpo_series.iloc[-1] > 0:
        signals.append(BULLISH_SIGNAL)
    elif jtpo_series.iloc[-2] > 0 and jtpo_series.iloc[-1] < 0:
        signals.append(BEARISH_SIGNAL)

    if jtpo_series.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif jtpo_series.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_laguerre(laguerre_series: pd.Series) -> list:
    signals = []

    if laguerre_series.iloc[-2] < 0.5 and laguerre_series.iloc[-1] > 0.5:
        signals.append(BULLISH_SIGNAL)
    elif laguerre_series.iloc[-2] > 0.5 and laguerre_series.iloc[-1] < 0.5:
        signals.append(BEARISH_SIGNAL)

    if laguerre_series.iloc[-1] > 0.5:
        signals.append(BULLISH_TREND)
    elif laguerre_series.iloc[-1] < 0.5:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_schaff_trend_cycle(
    stc_series: pd.Series, overbought=75, oversold=25
) -> list:
    signals = []

    if stc_series.iloc[-2] < oversold and stc_series.iloc[-1] > oversold:
        signals.append(BULLISH_SIGNAL)
    elif stc_series.iloc[-2] > overbought and stc_series.iloc[-1] < overbought:
        signals.append(BEARISH_SIGNAL)

    if stc_series.iloc[-1] > overbought:
        signals.append(OVERBOUGHT)
    elif stc_series.iloc[-1] < oversold:
        signals.append(OVERSOLD)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_tdfi(td_series: pd.Series) -> list:
    signals = []

    if td_series.iloc[-2] < 0 and td_series.iloc[-1] > 0:
        signals.append(BULLISH_SIGNAL)
    elif td_series.iloc[-2] > 0 and td_series.iloc[-1] < 0:
        signals.append(BEARISH_SIGNAL)

    if td_series.iloc[-1] > 0:
        signals.append(BULLISH_TREND)
    elif td_series.iloc[-1] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_ttf(ttf_series: pd.Series, upper=75, lower=-75) -> list:
    signals = []

    if ttf_series.iloc[-2] < lower and ttf_series.iloc[-1] > lower:
        signals.append(BULLISH_SIGNAL)
    elif ttf_series.iloc[-2] > upper and ttf_series.iloc[-1] < upper:
        signals.append(BEARISH_SIGNAL)

    if ttf_series.iloc[-1] > upper:
        signals.append(BULLISH_TREND)
    elif ttf_series.iloc[-1] < lower:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


# --- Trend Signal Functions ---


def signal_aroon(aroon_df: pd.DataFrame, threshold: float = 70.0) -> list:
    signals = []
    aroon_up, aroon_down = aroon_df.iloc[:, 0], aroon_df.iloc[:, 1]
    up_prev, up_curr = aroon_up.iloc[-2], aroon_up.iloc[-1]
    down_prev, down_curr = aroon_down.iloc[-2], aroon_down.iloc[-1]

    # Entry signal (crossover)
    if up_prev < down_prev and up_curr > down_curr:
        signals.append(BULLISH_SIGNAL)
    elif down_prev < up_prev and down_curr > up_curr:
        signals.append(BEARISH_SIGNAL)

    # Trend direction
    if up_curr > threshold and down_curr < 30:
        signals.append(BULLISH_TREND)
    elif down_curr > threshold and up_curr < 30:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_aroonosc(osc_series: pd.Series) -> list:
    signals = []
    val = osc_series.iloc[-1]

    if val > 0:
        signals.append(BULLISH_TREND)
    elif val < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_kijunsen(kijun_series: pd.Series, close_series: pd.Series) -> list:
    signals = []
    curr_close = close_series.iloc[-1]
    prev_close = close_series.iloc[-2]
    curr_kijun = kijun_series.iloc[-1]
    prev_kijun = kijun_series.iloc[-2]

    if prev_close < prev_kijun and curr_close > curr_kijun:
        signals.append(BULLISH_SIGNAL)
    elif prev_close > prev_kijun and curr_close < curr_kijun:
        signals.append(BEARISH_SIGNAL)

    if curr_close > curr_kijun:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_kijun:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_ssl(ssl_df: pd.DataFrame) -> list:
    signals = []
    ssl_up, ssl_down = ssl_df["SSL_Up"], ssl_df["SSL_Down"]
    if ssl_down.iloc[-2] < ssl_up.iloc[-2] and ssl_down.iloc[-1] > ssl_up.iloc[-1]:
        signals.append(BEARISH_SIGNAL)
    elif ssl_down.iloc[-2] > ssl_up.iloc[-2] and ssl_down.iloc[-1] < ssl_up.iloc[-1]:
        signals.append(BULLISH_SIGNAL)

    if ssl_down.iloc[-1] > ssl_up.iloc[-1]:
        signals.append(BEARISH_TREND)
    elif ssl_down.iloc[-1] < ssl_up.iloc[-1]:
        signals.append(BULLISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_supertrend(supertrend_df: pd.DataFrame) -> list:
    signals = []
    trend = supertrend_df["Trend"].iloc[-1]

    if trend == 1:
        signals.append(BULLISH_TREND)
    elif trend == -1:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_halftrend(ht_df: pd.DataFrame) -> list:
    signals = []

    up_prev, up_curr = ht_df["Up"].iloc[-2], ht_df["Up"].iloc[-1]
    down_prev, down_curr = ht_df["Down"].iloc[-2], ht_df["Down"].iloc[-1]

    # Entry signals only when new reversal occurs
    if up_prev == 0 and up_curr > 0:
        signals.append(BULLISH_SIGNAL)
    elif down_prev == 0 and down_curr > 0:
        signals.append(BEARISH_SIGNAL)

    # Trend bias (ongoing)
    if up_curr > 0:
        signals.append(BULLISH_TREND)
    elif down_curr > 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_lsma(lsma_series: pd.Series, close_series: pd.Series) -> list:
    signals = []

    if (
        close_series.iloc[-2] < lsma_series.iloc[-2]
        and close_series.iloc[-1] > lsma_series.iloc[-1]
    ):
        signals.append(BULLISH_SIGNAL)
    elif (
        close_series.iloc[-2] > lsma_series.iloc[-2]
        and close_series.iloc[-1] < lsma_series.iloc[-1]
    ):
        signals.append(BEARISH_SIGNAL)

    if close_series.iloc[-1] > lsma_series.iloc[-1]:
        signals.append(BULLISH_TREND)
    elif close_series.iloc[-1] < lsma_series.iloc[-1]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_kalman_filter(kalman_df: pd.DataFrame) -> list:
    signals = []
    up_curr = kalman_df["Up"].iloc[-1]
    down_curr = kalman_df["Down"].iloc[-1]

    if not pd.isna(up_curr):
        signals.append(BULLISH_TREND)
    elif not pd.isna(down_curr):
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_uf2018(uf_df: pd.DataFrame) -> list:
    signals = []
    prev = uf_df.iloc[-2]
    curr = uf_df.iloc[-1]

    if prev["Value"] < 0 and curr["Value"] > 0:
        signals.append(BULLISH_SIGNAL)
    elif prev["Value"] > 0 and curr["Value"] < 0:
        signals.append(BEARISH_SIGNAL)

    if curr["Value"] > 0:
        signals.append(BULLISH_TREND)
    elif curr["Value"] < 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_center_of_gravity(cog_series: pd.Series) -> list:
    signals = []
    signal_line = cog_series.rolling(window=3).mean()

    if (
        cog_series.iloc[-2] < signal_line.iloc[-2]
        and cog_series.iloc[-1] > signal_line.iloc[-1]
    ):
        signals.append(BULLISH_SIGNAL)
    elif (
        cog_series.iloc[-2] > signal_line.iloc[-2]
        and cog_series.iloc[-1] < signal_line.iloc[-1]
    ):
        signals.append(BEARISH_SIGNAL)

    if cog_series.iloc[-1] > signal_line.iloc[-1]:
        signals.append(BULLISH_TREND)
    elif cog_series.iloc[-1] < signal_line.iloc[-1]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_grucha_index(grucha_df: pd.DataFrame) -> list:
    signals = []
    idx_prev = grucha_df["Grucha Index"].iloc[-2]
    idx_curr = grucha_df["Grucha Index"].iloc[-1]
    ma_prev = grucha_df["MA of Grucha Index"].iloc[-2]
    ma_curr = grucha_df["MA of Grucha Index"].iloc[-1]

    if idx_prev < ma_prev and idx_curr > ma_curr:
        signals.append(BULLISH_SIGNAL)
    elif idx_prev > ma_prev and idx_curr < ma_curr:
        signals.append(BEARISH_SIGNAL)

    if idx_curr > ma_curr:
        signals.append(BULLISH_TREND)
    elif idx_curr < ma_curr:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_top_trend(toptrend_df: pd.DataFrame) -> list:
    signals = []
    trend_val = toptrend_df["TopTrend"].iloc[-1]

    if trend_val == 1:
        signals.append(BULLISH_TREND)
    elif trend_val == -1:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_kase(kase_df: pd.DataFrame) -> list:
    """
    Generate trading signals based on the Kase Permission Stochastic Smoothed (KPSS) indicator.

    Parameters:
    kase_df (pd.DataFrame): DataFrame containing 'KPSS_BUY' and 'KPSS_SELL' columns from the KPSS indicator.

    Returns:
    list: A list of signals indicating potential trade actions.
    """
    signals = []

    # Ensure there are at least two data points to compare
    if len(kase_df) < 2:
        return signals

    # Get the last two values of the KPSS_BUY and KPSS_SELL lines
    prev_buy = kase_df["KPSS_BUY"].iloc[-2]
    curr_buy = kase_df["KPSS_BUY"].iloc[-1]
    prev_sell = kase_df["KPSS_SELL"].iloc[-2]
    curr_sell = kase_df["KPSS_SELL"].iloc[-1]

    # Determine crossover signals
    if prev_buy < prev_sell and curr_buy > curr_sell:
        signals.append("BULLISH_CROSSOVER")
    elif prev_buy > prev_sell and curr_buy < curr_sell:
        signals.append("BEARISH_CROSSOVER")

    # Determine trend direction
    if curr_buy > curr_sell:
        signals.append("BULLISH_TREND")
    elif curr_buy < curr_sell:
        signals.append("BEARISH_TREND")
    else:
        signals.append("NEUTRAL_TREND")

    return signals


# --- Volume Signal Functions (Returning List of Signal Constants) ---


def signal_ad(ad_series: pd.Series) -> list:
    signals = []
    if ad_series.iloc[-1] > ad_series.iloc[-2]:
        signals.append(HIGH_VOLUME)
    elif ad_series.iloc[-1] < ad_series.iloc[-2]:
        signals.append(LOW_VOLUME)

    return signals


# TA-Lib ADX and ADXR show only strength, not direction
def signal_adx(adx_series: pd.Series, threshold: float = 25.0) -> list:
    signals = []
    if adx_series.iloc[-1] >= threshold:
        signals.append(HIGH_VOLUME)
    else:
        signals.append(LOW_VOLUME)
    return signals


def signal_adosc(adosc_series: pd.Series) -> list:
    signals = []
    if adosc_series.iloc[-1] > adosc_series.iloc[-2]:
        signals.append(HIGH_VOLUME)
    elif adosc_series.iloc[-1] < adosc_series.iloc[-2]:
        signals.append(LOW_VOLUME)

    return signals


def signal_obv(obv_series: pd.Series) -> list:
    signals = []
    if obv_series.iloc[-1] > obv_series.iloc[-2]:
        signals.append(HIGH_VOLUME)
    elif obv_series.iloc[-1] < obv_series.iloc[-2]:
        signals.append(LOW_VOLUME)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_fantail_vma(fantail_df: pd.DataFrame) -> list:
    signals = []

    # Use first column for signal calculation
    vma_series = fantail_df["Fantail_VMA"]

    if vma_series.iloc[-1] > vma_series.iloc[-2]:
        signals.append(HIGH_VOLUME)
    elif vma_series.iloc[-1] < vma_series.iloc[-2]:
        signals.append(LOW_VOLUME)

    return signals


def signal_kvo(kvo_df: pd.DataFrame) -> list:
    signals = []
    if kvo_df["KVO"].iloc[-1] > kvo_df["KVO_Signal"].iloc[-1]:
        signals.append(HIGH_VOLUME)
    elif kvo_df["KVO"].iloc[-1] < kvo_df["KVO_Signal"].iloc[-1]:
        signals.append(LOW_VOLUME)

    return signals


def signal_lwpi(lwpi_df: pd.DataFrame) -> list:
    signals = []
    val = lwpi_df["LWPI"].iloc[-1]
    if val > 55:
        signals.append(HIGH_VOLUME)
    elif val < 45:
        signals.append(LOW_VOLUME)

    return signals


def signal_normalized_volume(vol_df: pd.DataFrame, threshold: float = 100.0) -> list:
    signals = []
    vol = vol_df["Vol"].iloc[-1]
    if vol > threshold:
        signals.append(HIGH_VOLUME)
    else:
        signals.append(LOW_VOLUME)

    return signals


def signal_twiggs_mf(tmf_df: pd.DataFrame, threshold: float = 0) -> list:
    signals = []
    tmf = tmf_df["TMF"].iloc[-1]
    if tmf > threshold:
        signals.append(HIGH_VOLUME)
    elif tmf < threshold:
        signals.append(LOW_VOLUME)

    return signals


# --- Volatility Signal Functions (Return List of Signal Constants) ---


def signal_stddev(std_series: pd.Series, threshold: float = None) -> list:
    signals = []

    if threshold and std_series.iloc[-1] > threshold:
        signals.append(HIGH_VOLATILITY)
    elif threshold and std_series.iloc[-1] < threshold:
        signals.append(LOW_VOLATILITY)

    if std_series.iloc[-1] > std_series.iloc[-2]:
        signals.append(INCREASING_VOLATILITY)
    elif std_series.iloc[-1] < std_series.iloc[-2]:
        signals.append(DECREASING_VOLATILITY)
    else:
        signals.append(STABLE_VOLATILITY)

    return signals


def signal_volatility_line(vol_series: pd.Series, threshold: float = None) -> list:
    """
    Signal generator for single-line volatility indicators (ATR, NATR, TRANGE).
    Optionally include a spike detection threshold.
    """
    signals = []

    # Optional threshold logic
    if threshold and vol_series.iloc[-1] > threshold:
        signals.append(HIGH_VOLATILITY)
    elif threshold and vol_series.iloc[-1] < threshold:
        signals.append(LOW_VOLATILITY)

    # Momentum-based trendiness
    if vol_series.iloc[-1] > vol_series.iloc[-2]:
        signals.append(INCREASING_VOLATILITY)
    elif vol_series.iloc[-1] < vol_series.iloc[-2]:
        signals.append(DECREASING_VOLATILITY)
    else:
        signals.append(STABLE_VOLATILITY)

    return signals


def signal_bollinger(bbands_df: pd.DataFrame, close_series: pd.Series) -> list:
    """
    Signal logic for Bollinger Bands. Uses crossover of price with upper/lower bands.
    """
    # First three columns are Bollinger Bands
    upper, middle, lower = (
        bbands_df.iloc[:, 0],
        bbands_df.iloc[:, 1],
        bbands_df.iloc[:, 2],
    )
    signals = []

    if (
        close_series.iloc[-2] < lower.iloc[-2]
        and close_series.iloc[-1] > lower.iloc[-1]
    ):
        signals.append(BULLISH_SIGNAL)
    elif (
        close_series.iloc[-2] > upper.iloc[-2]
        and close_series.iloc[-1] < upper.iloc[-1]
    ):
        signals.append(BEARISH_SIGNAL)

    if close_series.iloc[-1] > upper.iloc[-1]:
        signals.append(OVERBOUGHT)
    elif close_series.iloc[-1] < lower.iloc[-1]:
        signals.append(OVERSOLD)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_var(var_series: pd.Series, threshold: float = None) -> list:
    signals = []

    if threshold and var_series.iloc[-1] > threshold:
        signals.append(HIGH_VOLATILITY)
    elif threshold and var_series.iloc[-1] < threshold:
        signals.append(LOW_VOLATILITY)

    if var_series.iloc[-1] > var_series.iloc[-2]:
        signals.append(INCREASING_VOLATILITY)
    elif var_series.iloc[-1] < var_series.iloc[-2]:
        signals.append(DECREASING_VOLATILITY)
    else:
        signals.append(STABLE_VOLATILITY)

    return signals


def signal_volatility_ratio(vr_series: pd.Series, threshold: float = 1.0) -> list:
    signals = []
    if vr_series.iloc[-1] > threshold:
        signals.append(HIGH_VOLATILITY)
    else:
        signals.append(LOW_VOLATILITY)
    return signals


def signal_wae(wae_df: pd.DataFrame) -> list:
    signals = []
    # Assume 'green_histogram' and 'red_histogram' are common WAE buffer names
    hist_green = (
        wae_df["green_histogram"].iloc[-1] if "green_histogram" in wae_df else 0
    )
    hist_red = wae_df["red_histogram"].iloc[-1] if "red_histogram" in wae_df else 0
    if abs(hist_green) > 0 or abs(hist_red) > 0:
        signals.append(HIGH_VOLATILITY)
    else:
        signals.append(LOW_VOLATILITY)
    return signals


def signal_filtered_atr(
    filtered_atr_series: pd.Series, threshold: float = None
) -> list:
    signals = []

    if threshold and filtered_atr_series.iloc[-1] > threshold:
        signals.append(HIGH_VOLATILITY)
    elif threshold and filtered_atr_series.iloc[-1] < threshold:
        signals.append(LOW_VOLATILITY)

    if filtered_atr_series.iloc[-1] > filtered_atr_series.iloc[-2]:
        signals.append(INCREASING_VOLATILITY)
    elif filtered_atr_series.iloc[-1] < filtered_atr_series.iloc[-2]:
        signals.append(DECREASING_VOLATILITY)
    else:
        signals.append(STABLE_VOLATILITY)

    return signals


# def signal_volatility_ratio(vr_df: pd.DataFrame) -> list:
#     signals = []
#     series = vr_df.iloc[:, 0]  # Assume first column is the ratio

#     if series.iloc[-1] > series.iloc[-2]:
#         signals.append(INCREASING_VOLATILITY)
#     elif series.iloc[-1] < series.iloc[-2]:
#         signals.append(DECREASING_VOLATILITY)
#     else:
#         signals.append(STABLE_VOLATILITY)

#     return signals


# --- Price/Statistical Signal Functions (Special Use Cases) ---


def signal_avgprice(avg_series: pd.Series, close_series: pd.Series) -> list:
    """If price crosses average price (OHLC/4), it suggests directional bias."""
    signals = []
    prev_close, curr_close = close_series.iloc[-2], close_series.iloc[-1]
    prev_avg, curr_avg = avg_series.iloc[-2], avg_series.iloc[-1]

    if prev_close < prev_avg and curr_close > curr_avg:
        signals.append(BULLISH_SIGNAL)
    elif prev_close > prev_avg and curr_close < curr_avg:
        signals.append(BEARISH_SIGNAL)

    if curr_close > curr_avg:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_avg:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_max(max_series: pd.Series, close_series: pd.Series) -> list:
    """Breakout above recent max close may suggest trend initiation."""
    signals = []
    if (
        close_series.iloc[-2] <= max_series.iloc[-2]
        and close_series.iloc[-1] > max_series.iloc[-1]
    ):
        signals.append(BULLISH_SIGNAL)
        signals.append(BULLISH_SIGNAL)
    elif close_series.iloc[-1] < max_series.iloc[-1]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_maxindex(maxindex_series: pd.Series) -> list:
    """If max index is 0, price just made a recent high — potential bullish bias."""
    signals = []
    if maxindex_series.iloc[-1] == 0:
        signals.append(BULLISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_minindex(minindex_series: pd.Series) -> list:
    """If min index is 0, price just made a recent low — potential bearish bias."""
    signals = []
    if minindex_series.iloc[-1] == 0:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_midpoint(midpoint_series: pd.Series, close_series: pd.Series) -> list:
    """Dynamic midpoint of closing prices over time — useful as a trend bias."""
    signals = []
    prev_close, curr_close = close_series.iloc[-2], close_series.iloc[-1]
    prev_mid, curr_mid = midpoint_series.iloc[-2], midpoint_series.iloc[-1]

    if prev_close < prev_mid and curr_close > curr_mid:
        signals.append(BULLISH_SIGNAL)
    elif prev_close > prev_mid and curr_close < curr_mid:
        signals.append(BEARISH_SIGNAL)

    if curr_close > curr_mid:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_mid:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_min(min_series: pd.Series, close_series: pd.Series) -> list:
    """Break below recent minimum may suggest momentum shift downward."""
    signals = []
    if (
        close_series.iloc[-2] >= min_series.iloc[-2]
        and close_series.iloc[-1] < min_series.iloc[-1]
    ):
        signals.append(BEARISH_SIGNAL)
        signals.append(BEARISH_SIGNAL)
    elif close_series.iloc[-1] > min_series.iloc[-1]:
        signals.append(BULLISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_minmax(
    min_series: pd.Series, max_series: pd.Series, close_series: pd.Series
) -> list:
    """Detect range-bound vs breakout behavior using recent min and max bounds."""
    signals = []
    close = close_series.iloc[-1]

    if close > max_series.iloc[-1]:
        signals.append(BULLISH_SIGNAL)
        signals.append(BULLISH_SIGNAL)
    elif close < min_series.iloc[-1]:
        signals.append(BEARISH_SIGNAL)
        signals.append(BEARISH_SIGNAL)
    elif min_series.iloc[-1] < close < max_series.iloc[-1]:
        signals.append(NEUTRAL_TREND)
    else:
        signals.append(INCONCLUSIVE)

    return signals


def signal_medprice(med_series: pd.Series, close_series: pd.Series) -> list:
    """Median price is (High + Low)/2 — acts as a central reference for trend bias."""
    signals = []
    curr_close = close_series.iloc[-1]
    curr_med = med_series.iloc[-1]

    if curr_close > curr_med:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_med:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_midprice(mid_series: pd.Series, close_series: pd.Series) -> list:
    """Midprice (avg of High and Low over time) used as short-term bias line."""
    signals = []
    prev_close, curr_close = close_series.iloc[-2], close_series.iloc[-1]
    prev_mid, curr_mid = mid_series.iloc[-2], mid_series.iloc[-1]

    if prev_close < prev_mid and curr_close > curr_mid:
        signals.append(BULLISH_SIGNAL)
    elif prev_close > prev_mid and curr_close < curr_mid:
        signals.append(BEARISH_SIGNAL)

    if curr_close > curr_mid:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_mid:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_sum(sum_series: pd.Series) -> list:
    """Rising sum suggests bullish momentum; falling sum suggests bearish momentum."""
    signals = []
    if sum_series.iloc[-1] > sum_series.iloc[-2]:
        signals.append(BULLISH_TREND)
    elif sum_series.iloc[-1] < sum_series.iloc[-2]:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)
    return signals


def signal_typprice(typ_series: pd.Series, close_series: pd.Series) -> list:
    """Typical price acts as a smoothed central bias — crossover can suggest directional momentum."""
    signals = []
    prev_close, curr_close = close_series.iloc[-2], close_series.iloc[-1]
    prev_typ, curr_typ = typ_series.iloc[-2], typ_series.iloc[-1]

    if prev_close < prev_typ and curr_close > curr_typ:
        signals.append(BULLISH_SIGNAL)
    elif prev_close > prev_typ and curr_close < curr_typ:
        signals.append(BEARISH_SIGNAL)

    if curr_close > curr_typ:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_typ:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


def signal_wclprice(wcl_series: pd.Series, close_series: pd.Series) -> list:
    """Weighted close price crossover suggests directional momentum and price positioning."""
    signals = []
    prev_close, curr_close = close_series.iloc[-2], close_series.iloc[-1]
    prev_wcl, curr_wcl = wcl_series.iloc[-2], wcl_series.iloc[-1]

    if prev_close < prev_wcl and curr_close > curr_wcl:
        signals.append(BULLISH_SIGNAL)
    elif prev_close > prev_wcl and curr_close < curr_wcl:
        signals.append(BEARISH_SIGNAL)

    if curr_close > curr_wcl:
        signals.append(BULLISH_TREND)
    elif curr_close < curr_wcl:
        signals.append(BEARISH_TREND)
    else:
        signals.append(NEUTRAL_TREND)

    return signals


# --- Generic Candlestick Signal Function ---
# TA-Lib returns:
#  100 for bullish pattern
# -100 for bearish pattern
#    0 for no pattern


def interpret_candlestick(cdl_series: pd.Series) -> list:
    signals = []
    value = cdl_series.iloc[-1]

    if value == 100:
        signals.append(BULLISH_SIGNAL)
    elif value == -100:
        signals.append(BEARISH_SIGNAL)
    else:
        signals.append(NO_SIGNAL)

    return signals
