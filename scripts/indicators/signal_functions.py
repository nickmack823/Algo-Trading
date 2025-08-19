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
        macd_df["0"],
        macd_df["1"],
        macd_df["2"],
    )
    signals = []

    # Crossover entry signals
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


def signal_fisher(fisher_series: pd.Series) -> list:
    signals = []
    f0, f1 = fisher_series.iloc[-2], fisher_series.iloc[-1]

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
    bulls = impulse_df["Bulls"].iloc[-1]
    bears = impulse_df["Bears"].iloc[-1]

    if bulls > 0:
        signals.append("BULLISH_TREND")
    elif bears > 0:
        signals.append("BEARISH_TREND")
    else:
        signals.append("NEUTRAL_TREND")

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


def signal_schaff_trend_cycle(stc_df: pd.DataFrame, overbought=75, oversold=25) -> list:
    signals = []
    stc_series = stc_df["STC"]

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


def signal_ttf(ttf_df: pd.DataFrame, upper=75, lower=-75) -> list:
    signals = []
    signal_series = ttf_df["Signal"]

    if signal_series.iloc[-2] <= lower and signal_series.iloc[-1] > lower:
        signals.append(BULLISH_SIGNAL)
    elif signal_series.iloc[-2] >= upper and signal_series.iloc[-1] < upper:
        signals.append(BEARISH_SIGNAL)

    if signal_series.iloc[-1] >= upper:
        signals.append(BULLISH_TREND)
    elif signal_series.iloc[-1] <= lower:
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

    if prev["BUY"] < 0 and curr["BUY"] > 0:
        signals.append(BULLISH_SIGNAL)
    elif prev["SELL"] > 0 and curr["SELL"] < 0:
        signals.append(BEARISH_SIGNAL)

    if curr["BUY"] > 0:
        signals.append(BULLISH_TREND)
    elif curr["SELL"] < 0:
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


def signal_top_trend(toptrend_series: pd.Series) -> list:
    signals = []
    trend_val = toptrend_series.iloc[-1]

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


def signal_lwpi(lwpi_series: pd.Series) -> list:
    signals = []
    val = lwpi_series.iloc[-1]
    if val > 55:
        signals.append(HIGH_VOLUME)
    elif val < 45:
        signals.append(LOW_VOLUME)

    return signals


def signal_normalized_volume(vol_series: pd.Series, threshold: float = 100.0) -> list:
    signals = []
    vol = vol_series.iloc[-1]
    if vol > threshold:
        signals.append(HIGH_VOLUME)
    else:
        signals.append(LOW_VOLUME)

    return signals


def signal_twiggs_mf(tmf_series: pd.Series, threshold: float = 0) -> list:
    signals = []
    tmf = tmf_series.iloc[-1]
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


# Auto-generated by merge_indicators.py
# Combined indicator signal functions


# === BEGIN 2nd_order_gaussian_high_pass_filter_mtf_zones_signal.py ===
def signal_2nd_order_gaussian_high_pass_filter_mtf_zones(
    series: pd.Series,
) -> list[str]:
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = series.iloc[-2]
    curr = series.iloc[-1]

    # Determine trend from current value
    if pd.notna(curr):
        if curr > 0:
            trend = BULLISH_TREND
        elif curr < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    # Signal from zero-line cross using last two bars
    signal = NO_SIGNAL
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            signal = BULLISH_SIGNAL
        elif prev >= 0 and curr < 0:
            signal = BEARISH_SIGNAL

    return [signal, trend]


# === END 2nd_order_gaussian_high_pass_filter_mtf_zones_signal.py ===


# === BEGIN 3rdgenma_signal.py ===
def signal_3rdgenma(df: pd.DataFrame) -> list[str]:
    # Expect df to contain columns: "MA3G" (the 3rd gen MA) and "MA1" (first-pass MA as price proxy)
    # try:
    ma3g = df["MA3G"]
    ma1 = df["MA1"]
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]
    # if ma3g is None or ma1 is None or len(df) < 2:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    # Use last two bars
    ma3g_prev = ma3g.iloc[-2] if len(ma3g) >= 2 else pd.NA
    ma3g_curr = ma3g.iloc[-1] if len(ma3g) >= 1 else pd.NA
    ma1_prev = ma1.iloc[-2] if len(ma1) >= 2 else pd.NA
    ma1_curr = ma1.iloc[-1] if len(ma1) >= 1 else pd.NA

    # Robust to NaNs: if any needed value is NaN, return neutral
    vals = [ma3g_prev, ma3g_curr, ma1_prev, ma1_curr]
    if any(pd.isna(v) for v in vals):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Signal: cross of MA1 (price proxy) vs MA3G
    prev_above = ma1_prev > ma3g_prev
    curr_above = ma1_curr > ma3g_curr

    tags: list[str] = []
    if prev_above != curr_above:
        tags.append(BULLISH_SIGNAL if curr_above else BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend: slope of MA3G
    delta = ma3g_curr - ma3g_prev
    if pd.isna(delta) or delta == 0:
        trend_tag = NEUTRAL_TREND
    else:
        trend_tag = BULLISH_TREND if delta > 0 else BEARISH_TREND

    tags.append(trend_tag)
    return tags


# === END 3rdgenma_signal.py ===

# === BEGIN adaptive_smoother_triggerlines_mtf_alerts_nmc_signal.py ===


def signal_adaptive_smoother_triggerlines_mtf_alerts_nmc(df: pd.DataFrame) -> list[str]:
    # Basic guards
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Ensure required columns exist
    if "lstrend" not in df.columns and "lwtrend" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Try to use lstrend; fall back to lwtrend for NaNs on the last two bars
    lstrend = (
        df["lstrend"]
        if "lstrend" in df.columns
        else pd.Series([float("nan")] * len(df), index=df.index)
    )
    lwtrend = (
        df["lwtrend"]
        if "lwtrend" in df.columns
        else pd.Series([float("nan")] * len(df), index=df.index)
    )

    prev = lstrend.iloc[-2]
    cur = lstrend.iloc[-1]

    if pd.isna(prev):
        prev = lwtrend.iloc[-2]
    if pd.isna(cur):
        cur = lwtrend.iloc[-1]

    tags: list[str] = []

    # Signal on change in trend slope (+1 to -1 or -1 to +1)
    if pd.notna(prev) and pd.notna(cur) and prev != cur:
        if cur > 0:
            tags.append(BULLISH_SIGNAL)
        elif cur < 0:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag from current slope
    if pd.notna(cur):
        if cur > 0:
            tags.append(BULLISH_TREND)
        elif cur < 0:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END adaptive_smoother_triggerlines_mtf_alerts_nmc_signal.py ===


# === BEGIN aso_signal.py ===
def signal_aso(df: pd.DataFrame) -> list[str]:
    required_cols = {"ASO_Bulls", "ASO_Bears"}
    if (
        not isinstance(df, pd.DataFrame)
        or not required_cols.issubset(df.columns)
        or len(df) < 2
    ):
        return [NO_SIGNAL, NEUTRAL_TREND]

    bulls = df["ASO_Bulls"]
    bears = df["ASO_Bears"]
    diff = bulls - bears

    prev = diff.iloc[-2]
    curr = diff.iloc[-1]

    # Trend determination
    if pd.isna(curr):
        trend = NEUTRAL_TREND
    elif curr > 0:
        trend = BULLISH_TREND
    elif curr < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Signal determination (last two bars crossover of bulls vs bears / zero on diff)
    if pd.isna(prev) or pd.isna(curr):
        signal = NO_SIGNAL
    elif prev <= 0 and curr > 0:
        signal = BULLISH_SIGNAL
    elif prev >= 0 and curr < 0:
        signal = BEARISH_SIGNAL
    else:
        signal = NO_SIGNAL

    return [signal, trend]


# === END aso_signal.py ===


# === BEGIN atr_based_ema_variant_1_signal.py ===
def signal_atr_based_ema_variant_1(df: pd.DataFrame) -> list[str]:
    # Expecting columns: ['EMA_ATR_var1', 'EMA_Equivalent']
    # try:
    ema = pd.to_numeric(df["EMA_ATR_var1"], errors="coerce")
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    ema_nonan = ema.dropna()
    n = len(ema_nonan)

    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if n == 1:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last = float(ema_nonan.iloc[-1])
    prev = float(ema_nonan.iloc[-2])

    # Trend based on slope of baseline over the last two bars
    if last > prev:
        trend_tag = BULLISH_TREND
    elif last < prev:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Signal on slope direction change (uses last three valid points if available)
    signal_tag = NO_SIGNAL
    if n >= 3:
        prev2 = float(ema_nonan.iloc[-3])
        delta_prev = prev - prev2
        delta_now = last - prev

        sign_prev = 1 if delta_prev > 0 else (-1 if delta_prev < 0 else 0)
        sign_now = 1 if delta_now > 0 else (-1 if delta_now < 0 else 0)

        if sign_now != 0 and sign_now != sign_prev:
            signal_tag = BULLISH_SIGNAL if sign_now > 0 else BEARISH_SIGNAL

    return [signal_tag, trend_tag]


# === END atr_based_ema_variant_1_signal.py ===


# === BEGIN bams_bung_3_signal.py ===
def signal_bams_bung_3(df: pd.DataFrame) -> list[str]:
    # Robustness checks
    # try:
    up_sig = df["UpTrendSignal"]
    dn_sig = df["DownTrendSignal"]
    up_stop = df["UpTrendStop"]
    dn_stop = df["DownTrendStop"]
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    if n < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    def is_active_signal(x) -> bool:
        return pd.notna(x) and x not in (float("inf"), float("-inf")) and x != -1.0

    def is_active_stop(x) -> bool:
        # Active stop when it's defined and not the inactive sentinel -1.0
        return pd.notna(x) and x not in (float("inf"), float("-inf")) and x != -1.0

    # Signal detection (stars fire only on the activation bar)
    last_up_sig = up_sig.iat[-1]
    prev_up_sig = up_sig.iat[-2]
    last_dn_sig = dn_sig.iat[-1]
    prev_dn_sig = dn_sig.iat[-2]

    star_up = is_active_signal(last_up_sig) and not is_active_signal(prev_up_sig)
    star_dn = is_active_signal(last_dn_sig) and not is_active_signal(prev_dn_sig)

    tags: list[str] = []
    if star_up and not star_dn:
        tags.append(BULLISH_SIGNAL)
    elif star_dn and not star_up:
        tags.append(BEARISH_SIGNAL)
    elif star_up and star_dn:
        tags.append(INCONCLUSIVE)
    else:
        tags.append(NO_SIGNAL)

    # Trend determination from stops
    last_up_stop = up_stop.iat[-1]
    last_dn_stop = dn_stop.iat[-1]
    up_active = is_active_stop(last_up_stop)
    dn_active = is_active_stop(last_dn_stop)

    if up_active and not dn_active:
        trend_tag = BULLISH_TREND
    elif dn_active and not up_active:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END bams_bung_3_signal.py ===


# === BEGIN band_pass_filter_signal.py ===
def signal_band_pass_filter(df: pd.DataFrame) -> list[str]:
    # Validate input
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or "BP" not in df.columns
    ):
        return [NO_SIGNAL, NEUTRAL_TREND]

    bp = df["BP"]
    n = len(bp)

    # Determine trend from latest available value (prefer current, then previous)
    def trend_from_value(v):
        if pd.isna(v):
            return NEUTRAL_TREND
        if v > 0:
            return BULLISH_TREND
        if v < 0:
            return BEARISH_TREND
        return NEUTRAL_TREND

    if n < 2:
        last_val = bp.iloc[-1] if n == 1 else float("nan")
        return [NO_SIGNAL, trend_from_value(last_val)]

    prev = bp.iloc[-2]
    curr = bp.iloc[-1]

    # Signal logic: zero-cross over the last two bars
    signal = NO_SIGNAL
    if not pd.isna(prev) and not pd.isna(curr):
        if prev <= 0 and curr > 0:
            signal = BULLISH_SIGNAL
        elif prev >= 0 and curr < 0:
            signal = BEARISH_SIGNAL

    # Trend tag based on latest available data (current preferred, else previous)
    if not pd.isna(curr):
        trend_tag = trend_from_value(curr)
    elif not pd.isna(prev):
        trend_tag = trend_from_value(prev)
    else:
        trend_tag = NEUTRAL_TREND

    return [signal, trend_tag]


# === END band_pass_filter_signal.py ===


# === BEGIN chandelierexit_signal.py ===
def signal_chandelierexit(df: pd.DataFrame) -> list[str]:
    # Validate input and coerce to numeric
    # try:
    long_line = pd.to_numeric(df["Chandelier_Long"], errors="coerce")
    short_line = pd.to_numeric(df["Chandelier_Short"], errors="coerce")
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Current state
    last_long = long_line.iloc[-1] if n >= 1 else float("nan")
    last_short = short_line.iloc[-1] if n >= 1 else float("nan")
    curr_long_active = pd.notna(last_long) and pd.isna(last_short)
    curr_short_active = pd.notna(last_short) and pd.isna(last_long)

    # Trend tag
    if curr_long_active:
        trend = BULLISH_TREND
    elif curr_short_active:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Signal based on flip between last two bars
    signal = NO_SIGNAL
    if n >= 2:
        prev_long = long_line.iloc[-2]
        prev_short = short_line.iloc[-2]
        prev_long_active = pd.notna(prev_long) and pd.isna(prev_short)
        prev_short_active = pd.notna(prev_short) and pd.isna(prev_long)

        if prev_long_active and curr_short_active:
            signal = BEARISH_SIGNAL  # Exit longs
        elif prev_short_active and curr_long_active:
            signal = BULLISH_SIGNAL  # Exit shorts

    return [signal, trend]


# === END chandelierexit_signal.py ===


# === BEGIN cmo_signal.py ===
def signal_cmo(series: pd.Series) -> list[str]:
    tags: list[str] = []

    if series is None or len(series) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    c0 = series.iloc[-1] if len(series) >= 1 else float("nan")
    c1 = series.iloc[-2] if len(series) >= 2 else float("nan")

    # Overbought / Oversold tags based on current value
    if pd.notna(c0):
        if c0 >= 50:
            tags.append(OVERBOUGHT)
        elif c0 <= -50:
            tags.append(OVERSOLD)

    # Signal generation using last two bars
    signaled = False
    if pd.notna(c0) and pd.notna(c1):
        bullish_cross_zero = c1 <= 0 and c0 > 0
        bearish_cross_zero = c1 >= 0 and c0 < 0
        bullish_exit_oversold = c1 <= -50 and c0 > -50
        bearish_exit_overbought = c1 >= 50 and c0 < 50

        if bullish_cross_zero or bullish_exit_oversold:
            tags.append(BULLISH_SIGNAL)
            signaled = True
        elif bearish_cross_zero or bearish_exit_overbought:
            tags.append(BEARISH_SIGNAL)
            signaled = True

    if not signaled:
        tags.append(NO_SIGNAL)

    # Trend tag (final tag)
    if pd.notna(c0):
        if c0 > 0:
            trend = BULLISH_TREND
        elif c0 < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END cmo_signal.py ===


# === BEGIN coral_signal.py ===
def signal_coral(df: pd.DataFrame) -> list[str]:
    # Ensure 'Coral' exists
    if not isinstance(df, pd.DataFrame) or "Coral" not in df.columns or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]

    coral = df["Coral"].astype(float)
    n = len(coral)
    if n < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    c_curr = coral.iloc[-1]
    c_prev = coral.iloc[-2]

    # Try price-vs-coral confirmation if Close is available; otherwise fall back to coral slope
    signal_tags: list[str] = []
    trend_tag = NEUTRAL_TREND

    if "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors="coerce")
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
            cond_up = (
                pd.notna(p_prev)
                and pd.notna(c_prev)
                and pd.notna(p_curr)
                and pd.notna(c_curr)
                and (p_prev <= c_prev)
                and (p_curr > c_curr)
            )
            cond_down = (
                pd.notna(p_prev)
                and pd.notna(c_prev)
                and pd.notna(p_curr)
                and pd.notna(c_curr)
                and (p_prev >= c_prev)
                and (p_curr < c_curr)
            )

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


# === END coral_signal.py ===


# === BEGIN cvi_multi_signal.py ===
def signal_cvi_multi(series: pd.Series) -> list[str]:
    # Expecting a CVI Series where sign indicates price relative to baseline
    # try:
    s = series.astype(float)
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    if s.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = s.iloc[-2]
    curr = s.iloc[-1]

    if pd.isna(prev) or pd.isna(curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = []

    # Cross of zero line implies baseline cross
    if prev <= 0.0 and curr > 0.0:
        tags.append(BULLISH_SIGNAL)
    elif prev >= 0.0 and curr < 0.0:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Optional OB/OS based on normalized magnitude (ATR units)
    ob_level = 2.0
    if curr >= ob_level:
        tags.append(OVERBOUGHT)
    elif curr <= -ob_level:
        tags.append(OVERSOLD)

    # Trend tag (exactly one, appended last)
    if curr > 0.0:
        tags.append(BULLISH_TREND)
    elif curr < 0.0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END cvi_multi_signal.py ===


# === BEGIN cyber_cycle_signal.py ===
def signal_cyber_cycle(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if not isinstance(df, pd.DataFrame) or "Cycle" not in df.columns or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    c = df["Cycle"]
    c_prev = c.iloc[-2]
    c_curr = c.iloc[-1]

    if pd.isna(c_prev) or pd.isna(c_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    cross_up = (c_prev <= 0) and (c_curr > 0)
    cross_dn = (c_prev >= 0) and (c_curr < 0)
    rising = c_curr > c_prev
    falling = c_curr < c_prev

    if cross_up:
        tags.append(BULLISH_SIGNAL)
    elif cross_dn:
        tags.append(BEARISH_SIGNAL)
    elif rising:
        tags.append(BULLISH_SIGNAL)
    elif falling:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    if (c_curr > 0 and rising) or (c_curr > 0 and not falling):
        trend = BULLISH_TREND
    elif (c_curr < 0 and falling) or (c_curr < 0 and not rising):
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END cyber_cycle_signal.py ===

# === BEGIN decycler_oscillator_signal.py ===


def signal_decycler_oscillator(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Extract series safely
    deo = (
        df["DEO"]
        if "DEO" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    )
    deo2 = (
        df["DEO2"]
        if "DEO2" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    )

    # Signals: use zero-line cross of DEO2 (faster line) over the last two bars
    prev = deo2.iloc[-2]
    curr = deo2.iloc[-1]
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            tags.append(BULLISH_SIGNAL)
        elif prev >= 0 and curr < 0:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend: use current DEO (slower line). If NaN, fall back to DEO2. Zero implies neutral.
    curr_deo = deo.iloc[-1]
    trend_tag = NEUTRAL_TREND
    if pd.notna(curr_deo):
        if curr_deo > 0:
            trend_tag = BULLISH_TREND
        elif curr_deo < 0:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        curr_deo2 = deo2.iloc[-1]
        if pd.notna(curr_deo2):
            if curr_deo2 > 0:
                trend_tag = BULLISH_TREND
            elif curr_deo2 < 0:
                trend_tag = BEARISH_TREND
            else:
                trend_tag = NEUTRAL_TREND
        else:
            trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END decycler_oscillator_signal.py ===


# === BEGIN detrended_synthetic_price_goscillators_signal.py ===
def signal_detrended_synthetic_price_goscillators(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame) or "DSP" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    dsp = df["DSP"]
    v_prev = dsp.iloc[-2]
    v_curr = dsp.iloc[-1]

    # Handle NaNs robustly
    if pd.isna(v_prev) or pd.isna(v_curr):
        trend = NEUTRAL_TREND
        if pd.notna(v_curr):
            if v_curr > 0:
                trend = BULLISH_TREND
            elif v_curr < 0:
                trend = BEARISH_TREND
        return [NO_SIGNAL, trend]

    # Signal: zero-line cross in the last two bars
    if v_prev <= 0 and v_curr > 0:
        sig = BULLISH_SIGNAL
    elif v_prev >= 0 and v_curr < 0:
        sig = BEARISH_SIGNAL
    else:
        sig = NO_SIGNAL

    # Trend: sign of current DSP
    if v_curr > 0:
        trend = BULLISH_TREND
    elif v_curr < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    return [sig, trend]


# === END detrended_synthetic_price_goscillators_signal.py ===


# === BEGIN doda_stochastic_modified_signal.py ===
def signal_doda_stochastic_modified(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Validate inputs
    if df is None or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Extract lines
    # try:
    k = df["DodaStoch"]
    d = df["DodaSignal"]
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    if len(k) < 2 or len(d) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    k_prev, k_curr = k.iloc[-2], k.iloc[-1]
    d_prev, d_curr = d.iloc[-2], d.iloc[-1]

    if pd.isna(k_prev) or pd.isna(k_curr) or pd.isna(d_prev) or pd.isna(d_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Overbought/Oversold states based on current values
    if (k_curr <= 20) or (d_curr <= 20):
        tags.append(OVERSOLD)
    elif (k_curr >= 80) or (d_curr >= 80):
        tags.append(OVERBOUGHT)

    # Cross detections using last two bars
    bullish_cross = (k_prev <= d_prev) and (k_curr > d_curr)
    bearish_cross = (k_prev >= d_prev) and (k_curr < d_curr)

    # Signal only if cross occurs in appropriate territory (~20 for long, ~80 for short)
    if bullish_cross and (
        (k_curr <= 20) or (d_curr <= 20) or (k_prev <= 20) or (d_prev <= 20)
    ):
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross and (
        (k_curr >= 80) or (d_curr >= 80) or (k_prev >= 80) or (d_prev >= 80)
    ):
        tags.append(BEARISH_SIGNAL)

    # If no explicit signal, mark NO_SIGNAL
    if BULLISH_SIGNAL not in tags and BEARISH_SIGNAL not in tags:
        tags.append(NO_SIGNAL)

    # Trend determination from current relation of K vs D
    if k_curr > d_curr:
        trend_tag = BULLISH_TREND
    elif k_curr < d_curr:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END doda_stochastic_modified_signal.py ===


# === BEGIN dorsey_inertia_signal.py ===
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


# === END dorsey_inertia_signal.py ===


# === BEGIN dpo_histogram_indicator_signal.py ===
def signal_dpo_histogram_indicator(df: pd.DataFrame) -> list[str]:
    # Fallbacks for invalid input
    if not isinstance(df, pd.DataFrame) or df.empty or "DPO" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(df["DPO"], errors="coerce")

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


# === END dpo_histogram_indicator_signal.py ===


# === BEGIN ehlers_deli__detrended_leading_indicator_signal.py ===
def signal_ehlers_deli__detrended_leading_indicator(series: pd.Series) -> list[str]:
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.Series(series, dtype="float64")
    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s_prev = s.iloc[-2]
    s_curr = s.iloc[-1]

    sig = s.ewm(span=5, adjust=False, min_periods=1).mean()
    sig_prev = sig.iloc[-2] if len(sig) >= 2 else pd.NA
    sig_curr = sig.iloc[-1] if len(sig) >= 1 else pd.NA

    up_zero = pd.notna(s_prev) and pd.notna(s_curr) and (s_prev <= 0) and (s_curr > 0)
    down_zero = pd.notna(s_prev) and pd.notna(s_curr) and (s_prev >= 0) and (s_curr < 0)

    up_sig = (
        pd.notna(s_prev)
        and pd.notna(s_curr)
        and pd.notna(sig_prev)
        and pd.notna(sig_curr)
        and (s_prev <= sig_prev)
        and (s_curr > sig_curr)
    )
    down_sig = (
        pd.notna(s_prev)
        and pd.notna(s_curr)
        and pd.notna(sig_prev)
        and pd.notna(sig_curr)
        and (s_prev >= sig_prev)
        and (s_curr < sig_curr)
    )

    up = up_zero or up_sig
    down = down_zero or down_sig

    if up and not down:
        signal_tag = BULLISH_SIGNAL
    elif down and not up:
        signal_tag = BEARISH_SIGNAL
    else:
        signal_tag = NO_SIGNAL

    if pd.notna(s_curr):
        trend_tag = (
            BULLISH_TREND
            if s_curr > 0
            else (BEARISH_TREND if s_curr < 0 else NEUTRAL_TREND)
        )
    else:
        trend_tag = NEUTRAL_TREND

    return [signal_tag, trend_tag]


# === END ehlers_deli__detrended_leading_indicator_signal.py ===


# === BEGIN ehlers_deli_detrended_leading_indicator_signal.py ===
def signal_ehlers_deli_detrended_leading_indicator(series: pd.Series) -> list[str]:
    # Ensure float dtype and handle edge cases
    # try:
    s = series.astype(float)
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    if s.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = s.iloc[-2]
    curr = s.iloc[-1]

    # Signal line (short EMA) for cross confirmation
    sig = s.ewm(span=5, adjust=False).mean()
    sig_prev = sig.iloc[-2] if sig.shape[0] >= 2 else float("nan")
    sig_curr = sig.iloc[-1] if sig.shape[0] >= 1 else float("nan")

    # Cross conditions using last two bars
    valid_vals = pd.notna(prev) and pd.notna(curr)
    valid_sig = pd.notna(sig_prev) and pd.notna(sig_curr)

    cross_up_zero = valid_vals and (prev <= 0.0) and (curr > 0.0)
    cross_down_zero = valid_vals and (prev >= 0.0) and (curr < 0.0)

    cross_up_sig = valid_vals and valid_sig and (prev <= sig_prev) and (curr > sig_curr)
    cross_down_sig = (
        valid_vals and valid_sig and (prev >= sig_prev) and (curr < sig_curr)
    )

    tags: list[str] = []

    if cross_up_zero or cross_up_sig:
        tags.append(BULLISH_SIGNAL)
    elif cross_down_zero or cross_down_sig:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag based on current DELI level
    if pd.isna(curr):
        trend_tag = NEUTRAL_TREND
    elif curr > 0.0:
        trend_tag = BULLISH_TREND
    elif curr < 0.0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END ehlers_deli_detrended_leading_indicator_signal.py ===


# === BEGIN ehlers_early_onset_trend_signal.py ===
def signal_ehlers_early_onset_trend(df: pd.DataFrame) -> list[str]:
    # Robustness to insufficient data
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Extract last two bars
    prev = df.iloc[-2]
    last = df.iloc[-1]

    q1_prev = prev.get("EEOT_Q1", float("nan"))
    q1_last = last.get("EEOT_Q1", float("nan"))
    q2_prev = prev.get("EEOT_Q2", float("nan"))
    q2_last = last.get("EEOT_Q2", float("nan"))

    def is_num(x) -> bool:
        return pd.notna(x)

    def is_pos(x) -> bool:
        return is_num(x) and x > 0

    def is_neg(x) -> bool:
        return is_num(x) and x < 0

    # Slopes (last - prev)
    q1_slope = (
        q1_last - q1_prev if is_num(q1_last) and is_num(q1_prev) else float("nan")
    )
    q2_slope = (
        q2_last - q2_prev if is_num(q2_last) and is_num(q2_prev) else float("nan")
    )

    def rising(x) -> bool:
        return is_num(x) and x > 0

    def falling(x) -> bool:
        return is_num(x) and x < 0

    # Crosses around zero
    q1_cross_up = is_neg(q1_prev) and is_num(q1_last) and q1_last >= 0
    q2_cross_up = is_neg(q2_prev) and is_num(q2_last) and q2_last >= 0
    q1_cross_down = is_pos(q1_prev) and is_num(q1_last) and q1_last <= 0
    q2_cross_down = is_pos(q2_prev) and is_num(q2_last) and q2_last <= 0

    cross_up = q1_cross_up or q2_cross_up
    cross_down = q1_cross_down or q2_cross_down

    both_rising = rising(q1_slope) and rising(q2_slope)
    both_falling = falling(q1_slope) and falling(q2_slope)
    any_above = is_pos(q1_last) or is_pos(q2_last)
    any_below = is_neg(q1_last) or is_neg(q2_last)

    tags: list[str] = []

    if cross_up or (both_rising and any_above):
        tags.append(BULLISH_SIGNAL)
    elif cross_down or (both_falling and any_below):
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend assessment using last values and slopes
    score = 0
    if is_pos(q1_last):
        score += 1
    elif is_neg(q1_last):
        score -= 1

    if is_pos(q2_last):
        score += 1
    elif is_neg(q2_last):
        score -= 1

    if rising(q1_slope):
        score += 1
    elif falling(q1_slope):
        score -= 1

    if rising(q2_slope):
        score += 1
    elif falling(q2_slope):
        score -= 1

    if score >= 2:
        tags.append(BULLISH_TREND)
    elif score <= -2:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END ehlers_early_onset_trend_signal.py ===


# === BEGIN ehlers_reverse_ema_signal.py ===
def signal_ehlers_reverse_ema(df: pd.DataFrame) -> list[str]:
    # Expect columns: 'Main' and 'EMA'
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if ("Main" not in df.columns) or ("EMA" not in df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    main = df["Main"]
    ema = df["EMA"]

    # Use only the last two bars
    m_prev, m_curr = main.iloc[-2], main.iloc[-1]
    e_prev, e_curr = ema.iloc[-2], ema.iloc[-1]

    # If any needed value is NaN, return neutral
    if pd.isna(m_prev) or pd.isna(m_curr) or pd.isna(e_prev) or pd.isna(e_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = []

    bullish_cross = (m_prev <= e_prev) and (m_curr > e_curr)
    bearish_cross = (m_prev >= e_prev) and (m_curr < e_curr)

    if bullish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend classification from current relationship
    if m_curr > e_curr:
        trend = BULLISH_TREND
    elif m_curr < e_curr:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END ehlers_reverse_ema_signal.py ===


# === BEGIN ehlers_two_pole_super_smoother_filter_signal.py ===
def signal_ehlers_two_pole_super_smoother_filter(series: pd.Series) -> list[str]:
    # Robust to empty/short/NaN series
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = series.iloc[-2]
    curr = series.iloc[-1]

    if pd.isna(prev) or pd.isna(curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Trend from last two bars (slope of the baseline)
    if curr > prev:
        trend = BULLISH_TREND
    elif curr < prev:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Without price context, do not emit entry signals; provide trend only
    return [NO_SIGNAL, trend]


# === END ehlers_two_pole_super_smoother_filter_signal.py ===


# === BEGIN ehlersroofingfiltera_signal.py ===
def signal_ehlersroofingfiltera(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or df.empty or "rfilt" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    rf = pd.to_numeric(df["rfilt"], errors="coerce")

    # Need at least two bars for cross decisions
    if rf.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = rf.iloc[-2]
    last = rf.iloc[-1]

    # Determine signal based on zero-line cross of Roofing Filter
    signal = NO_SIGNAL
    if pd.notna(prev) and pd.notna(last):
        if prev < 0 and last > 0:
            signal = BULLISH_SIGNAL
        elif prev > 0 and last < 0:
            signal = BEARISH_SIGNAL

    # Determine trend from the latest Roofing Filter value
    trend = NEUTRAL_TREND
    if pd.notna(last):
        if last > 0:
            trend = BULLISH_TREND
        elif last < 0:
            trend = BEARISH_TREND

    return [signal, trend]


# === END ehlersroofingfiltera_signal.py ===


# === BEGIN ergodic_tvi_signal.py ===
def signal_ergodic_tvi(df: pd.DataFrame) -> list[str]:
    # Expect df to contain columns: "ETVI", "Signal"
    if df is None or df.empty or len(df.index) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    cols = ["ETVI", "Signal"]
    if not all(col in df.columns for col in cols):
        return [NO_SIGNAL, NEUTRAL_TREND]

    last2 = df[cols].tail(2)
    e_prev, e_curr = last2["ETVI"].iloc[0], last2["ETVI"].iloc[1]
    s_prev, s_curr = last2["Signal"].iloc[0], last2["Signal"].iloc[1]

    have_sig = (
        pd.notna(e_prev) and pd.notna(e_curr) and pd.notna(s_prev) and pd.notna(s_curr)
    )
    have_zero = pd.notna(e_prev) and pd.notna(e_curr)

    tags: list[str] = []

    # Primary signal: cross of ETVI vs Signal line; Secondary: cross of ETVI vs zero line
    if have_sig and (e_prev <= s_prev and e_curr > s_curr):
        tags.append(BULLISH_SIGNAL)
    elif have_sig and (e_prev >= s_prev and e_curr < s_curr):
        tags.append(BEARISH_SIGNAL)
    elif have_zero and (e_prev <= 0 and e_curr > 0):
        tags.append(BULLISH_SIGNAL)
    elif have_zero and (e_prev >= 0 and e_curr < 0):
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend determination (use ETVI vs Signal if available; else ETVI vs zero; else neutral)
    if pd.notna(e_curr) and pd.notna(s_curr):
        if e_curr > s_curr:
            tags.append(BULLISH_TREND)
        elif e_curr < s_curr:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    elif pd.notna(e_curr):
        if e_curr > 0:
            tags.append(BULLISH_TREND)
        elif e_curr < 0:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END ergodic_tvi_signal.py ===

# === BEGIN forecast_signal.py ===


def signal_forecast(series: pd.Series) -> list[str]:
    tags: list[str] = []

    n = len(series) if series is not None else 0
    prev = float(series.iloc[-2]) if n >= 2 else float("nan")
    curr = float(series.iloc[-1]) if n >= 1 else float("nan")

    cross_up = (prev <= 0) and (curr > 0)
    cross_down = (prev >= 0) and (curr < 0)

    if cross_up:
        tags.append(BULLISH_SIGNAL)
    elif cross_down:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    if curr > 0:
        tags.append(BULLISH_TREND)
    elif curr < 0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END forecast_signal.py ===


# === BEGIN frama_indicator_signal.py ===
def signal_frama_indicator(series: pd.Series) -> list[str]:
    # Ensure numeric and handle NaNs gracefully
    if series is None or len(series) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(series, errors="coerce")
    n = len(s)

    # Determine trend based on FRAMA slope (last two bars)
    trend = NEUTRAL_TREND
    s0 = s.iloc[-1] if n >= 1 else float("nan")
    s1 = s.iloc[-2] if n >= 2 else float("nan")
    if pd.notna(s0) and pd.notna(s1):
        if s0 > s1:
            trend = BULLISH_TREND
        elif s0 < s1:
            trend = BEARISH_TREND

    # Determine signal based on slope crossing zero (uses last two bars' slope comparison)
    signal = NO_SIGNAL
    if n >= 3:
        s2 = s.iloc[-3]
        if pd.notna(s0) and pd.notna(s1) and pd.notna(s2):
            diff_prev = s1 - s2
            diff_curr = s0 - s1
            if diff_prev <= 0 and diff_curr > 0:
                signal = BULLISH_SIGNAL
            elif diff_prev >= 0 and diff_curr < 0:
                signal = BEARISH_SIGNAL

    return [signal, trend]


# === END frama_indicator_signal.py ===


# === BEGIN gchannel_signal.py ===
def signal_gchannel(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Validate required column
    if not isinstance(df, pd.DataFrame) or "Middle" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    mid = pd.to_numeric(df["Middle"], errors="coerce")

    # Try to locate a price series (prefer Close)
    price = None
    for col in ("Close", "close", "CLOSE", "Price", "price"):
        if col in df.columns:
            price = pd.to_numeric(df[col], errors="coerce")
            break

    n = len(df)

    # Signal: price crossing Middle (last two bars)
    long_cross = False
    short_cross = False
    if price is not None and n >= 2:
        p_prev = price.iloc[-2]
        p_last = price.iloc[-1]
        m_prev = mid.iloc[-2]
        m_last = mid.iloc[-1]
        if (
            pd.notna(p_prev)
            and pd.notna(p_last)
            and pd.notna(m_prev)
            and pd.notna(m_last)
        ):
            long_cross = (p_prev <= m_prev) and (p_last > m_last)
            short_cross = (p_prev >= m_prev) and (p_last < m_last)

    if long_cross:
        tags.append(BULLISH_SIGNAL)
    elif short_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag: prefer price vs Middle; fallback to Middle slope
    trend = NEUTRAL_TREND
    if price is not None and n >= 1:
        p_last = price.iloc[-1]
        m_last = mid.iloc[-1]
        if pd.notna(p_last) and pd.notna(m_last):
            if p_last > m_last:
                trend = BULLISH_TREND
            elif p_last < m_last:
                trend = BEARISH_TREND
            else:
                trend = NEUTRAL_TREND
        else:
            # Fallback to Middle slope if last price or middle is NaN
            if n >= 2:
                m_prev = mid.iloc[-2]
                if pd.notna(m_prev) and pd.notna(m_last):
                    if m_last > m_prev:
                        trend = BULLISH_TREND
                    elif m_last < m_prev:
                        trend = BEARISH_TREND
                    else:
                        trend = NEUTRAL_TREND
    else:
        # No price available; use Middle slope if possible
        if n >= 2:
            m_prev = mid.iloc[-2]
            m_last = mid.iloc[-1]
            if pd.notna(m_prev) and pd.notna(m_last):
                if m_last > m_prev:
                    trend = BULLISH_TREND
                elif m_last < m_prev:
                    trend = BEARISH_TREND
                else:
                    trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END gchannel_signal.py ===


# === BEGIN gd_signal.py ===
def signal_gd(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if "GD" not in df.columns or "EMA" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    ema = df["EMA"]
    gd = df["GD"]

    curr_ema = ema.iloc[-1]
    prev_ema = ema.iloc[-2]
    curr_gd = gd.iloc[-1]
    prev_gd = gd.iloc[-2]

    # Robust to NaNs
    if pd.isna(curr_ema) or pd.isna(prev_ema) or pd.isna(curr_gd) or pd.isna(prev_gd):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Cross detection using last two bars
    bullish_cross = (prev_ema <= prev_gd) and (curr_ema > curr_gd)
    bearish_cross = (prev_ema >= prev_gd) and (curr_ema < curr_gd)

    # Trend determination from current relationship
    if curr_ema > curr_gd:
        trend = BULLISH_TREND
    elif curr_ema < curr_gd:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    if bullish_cross:
        return [BULLISH_SIGNAL, trend]
    if bearish_cross:
        return [BEARISH_SIGNAL, trend]
    return [NO_SIGNAL, trend]


# === END gd_signal.py ===


# === BEGIN geomin_ma_signal.py ===
def signal_geomin_ma(series: pd.Series) -> list[str]:
    # Robustness: need at least 3 values to detect a turn (baseline turning up/down)
    if series is None or len(series) < 3:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s1 = series.iloc[-1]
    s2 = series.iloc[-2]
    s3 = series.iloc[-3]

    # Handle NaNs in the last three bars
    if pd.isna(s1) or pd.isna(s2) or pd.isna(s3):
        return [NO_SIGNAL, NEUTRAL_TREND]

    d_now = s1 - s2
    d_prev = s2 - s3

    tags: list[str] = []

    # Signal on baseline turning points
    if d_prev <= 0 and d_now > 0:
        tags.append(BULLISH_SIGNAL)
    elif d_prev >= 0 and d_now < 0:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend from current slope
    if d_now > 0:
        tags.append(BULLISH_TREND)
    elif d_now < 0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END geomin_ma_signal.py ===


# === BEGIN glitch_index_fixed_signal.py ===
def signal_glitch_index_fixed(df: pd.DataFrame) -> list[str]:
    # Robustness checks
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if "gli" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s_gli = df["gli"]
    s_state = (
        df["state"]
        if "state" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )

    # Last two bars
    g1 = s_gli.iloc[-2]
    g0 = s_gli.iloc[-1]
    st1 = s_state.iloc[-2]
    st0 = s_state.iloc[-1]

    # Helper: finite check
    def _is_num(x) -> bool:
        return pd.notna(x)

    # Zero-line crosses using last two bars
    bull_cross = _is_num(g1) and _is_num(g0) and (g1 <= 0) and (g0 > 0)
    bear_cross = _is_num(g1) and _is_num(g0) and (g1 >= 0) and (g0 < 0)

    # State-based shift (last two bars)
    bull_shift = _is_num(st1) and _is_num(st0) and (st1 <= 0) and (st0 > 0)
    bear_shift = _is_num(st1) and _is_num(st0) and (st1 >= 0) and (st0 < 0)

    tags: list[str] = []

    if bull_cross or bull_shift:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross or bear_shift:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend: based on current gli (fallback to state if needed)
    if _is_num(g0):
        if g0 > 0:
            trend = BULLISH_TREND
        elif g0 < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    elif _is_num(st0):
        if st0 > 0:
            trend = BULLISH_TREND
        elif st0 < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END glitch_index_fixed_signal.py ===


# === BEGIN hacolt_2_02_lines_signal.py ===
def signal_hacolt_2_02_lines(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = (
        df["HACOLT"]
        if "HACOLT" in df.columns
        else pd.Series(index=df.index, dtype=float)
    )

    # Use last two bars only
    # try:
    prev_val = float(s.iloc[-2])
    curr_val = float(s.iloc[-1])
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    # Determine signal based on transition across zero line
    if pd.notna(prev_val) and pd.notna(curr_val):
        if (curr_val > 0) and (prev_val <= 0):
            tags.append(BULLISH_SIGNAL)
        elif (curr_val < 0) and (prev_val >= 0):
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Determine trend from the latest value
    if pd.notna(curr_val):
        if curr_val > 0:
            trend = BULLISH_TREND
        elif curr_val < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END hacolt_2_02_lines_signal.py ===


# === BEGIN hlctrend_signal.py ===
def signal_hlctrend(df: pd.DataFrame) -> list[str]:
    # Expecting df with columns ['first','second'] from Hlctrend(...)
    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if "first" in df.columns:
        s_first = pd.to_numeric(df["first"], errors="coerce")
    else:
        s_first = pd.Series(float("nan"), index=df.index, dtype="float64")

    if "second" in df.columns:
        s_second = pd.to_numeric(df["second"], errors="coerce")
    else:
        s_second = pd.Series(float("nan"), index=df.index, dtype="float64")

    # Composite: EMAC - HLC_baseline ∝ (first - second)
    comp = s_first - s_second

    # Last two bars
    curr = comp.iloc[-1] if len(comp) >= 1 else float("nan")
    prev = comp.iloc[-2] if len(comp) >= 2 else float("nan")

    tags: list[str] = []

    # Signal on zero-cross of composite
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            tags.append(BULLISH_SIGNAL)
        elif prev >= 0 and curr < 0:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend by current composite sign
    if pd.notna(curr):
        if curr > 0:
            tags.append(BULLISH_TREND)
        elif curr < 0:
            tags.append(BEARISH_TREND)
        else:
            tags.append(NEUTRAL_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END hlctrend_signal.py ===


# === BEGIN is_calculation_signal.py ===
def signal_is_calculation(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty or "Pente" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    pente = pd.to_numeric(df["Pente"], errors="coerce")

    # Determine signal using last two bars (zero-cross)
    if len(pente) >= 2:
        prev = pente.iloc[-2]
        curr = pente.iloc[-1]
        if pd.notna(prev) and pd.notna(curr):
            if prev <= 0 and curr > 0:
                tags.append(BULLISH_SIGNAL)
            elif prev >= 0 and curr < 0:
                tags.append(BEARISH_SIGNAL)
            else:
                tags.append(NO_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Determine trend from current reading
    curr = pente.iloc[-1] if len(pente) > 0 else float("nan")
    if pd.isna(curr):
        tags.append(NEUTRAL_TREND)
    elif curr > 0:
        tags.append(BULLISH_TREND)
    elif curr < 0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END is_calculation_signal.py ===


# === BEGIN jposcillator_signal.py ===
def signal_jposcillator(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty or "Jp" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    jp = pd.to_numeric(df["Jp"], errors="coerce")

    j0 = jp.iloc[-1] if len(jp) >= 1 else float("nan")
    j1 = jp.iloc[-2] if len(jp) >= 2 else float("nan")

    signal = None
    if pd.notna(j0) and pd.notna(j1):
        if j1 <= 0 and j0 > 0:
            signal = BULLISH_SIGNAL
        elif j1 >= 0 and j0 < 0:
            signal = BEARISH_SIGNAL

    if signal is None:
        tags.append(NO_SIGNAL)
    else:
        tags.append(signal)

    if pd.isna(j0):
        trend = NEUTRAL_TREND
    elif j0 > 0:
        trend = BULLISH_TREND
    elif j0 < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END jposcillator_signal.py ===


# === BEGIN mcginley_dynamic_2_3_signal.py ===
def signal_mcginley_dynamic_2_3(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Expect 'mcg' column from McginleyDynamic23 calculation output
    if "mcg" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = df["mcg"].astype(float)

    n = len(s)
    # Determine signal via slope change (rise/fall flip) using the last three points
    signal_tag = NO_SIGNAL
    if n >= 3:
        cur = s.iloc[-1]
        prev = s.iloc[-2]
        prev2 = s.iloc[-3]
        if pd.notna(cur) and pd.notna(prev) and pd.notna(prev2):
            slope_prev = prev - prev2
            slope_cur = cur - prev
            if slope_prev <= 0 and slope_cur > 0:
                signal_tag = BULLISH_SIGNAL
            elif slope_prev >= 0 and slope_cur < 0:
                signal_tag = BEARISH_SIGNAL

    tags.append(signal_tag)

    # Determine trend from the last two points (rising/falling/flat)
    trend_tag = NEUTRAL_TREND
    if n >= 2:
        cur = s.iloc[-1]
        prev = s.iloc[-2]
        if pd.notna(cur) and pd.notna(prev):
            if cur > prev:
                trend_tag = BULLISH_TREND
            elif cur < prev:
                trend_tag = BEARISH_TREND

    tags.append(trend_tag)
    return tags


# === END mcginley_dynamic_2_3_signal.py ===


# === BEGIN metro_advanced_signal.py ===
def signal_metro_advanced(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    r_last = df.iloc[-1]

    def is_num(x) -> bool:
        return x is not None and pd.notna(x)

    # If fewer than 2 bars, no cross decisions; try to infer trend, else neutral
    if n < 2:
        trend_tag = NEUTRAL_TREND
        tval = r_last.get("Trend")
        if is_num(tval):
            if tval > 0:
                trend_tag = BULLISH_TREND
            elif tval < 0:
                trend_tag = BEARISH_TREND
        return [NO_SIGNAL, trend_tag]

    r_prev = df.iloc[-2]

    sfast0 = r_prev.get("StepRSI_fast")
    sfast1 = r_last.get("StepRSI_fast")
    sslow0 = r_prev.get("StepRSI_slow")
    sslow1 = r_last.get("StepRSI_slow")

    rsi0 = r_prev.get("RSI")
    rsi1 = r_last.get("RSI")
    lmid0 = r_prev.get("Level_Mid")
    lmid1 = r_last.get("Level_Mid")

    lup0 = r_prev.get("Level_Up")
    lup1 = r_last.get("Level_Up")
    ldn0 = r_prev.get("Level_Dn")
    ldn1 = r_last.get("Level_Dn")

    bull_cross = False
    bear_cross = False

    # Signal-line cross: StepRSI_fast vs StepRSI_slow
    if is_num(sfast0) and is_num(sslow0) and is_num(sfast1) and is_num(sslow1):
        if sfast0 <= sslow0 and sfast1 > sslow1:
            bull_cross = True
        elif sfast0 >= sslow0 and sfast1 < sslow1:
            bear_cross = True

    # Centre-line cross: RSI vs Level_Mid
    if is_num(rsi0) and is_num(lmid0) and is_num(rsi1) and is_num(lmid1):
        if rsi0 <= lmid0 and rsi1 > lmid1:
            bull_cross = True
        elif rsi0 >= lmid0 and rsi1 < lmid1:
            bear_cross = True

    if bull_cross and not bear_cross:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross and not bull_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Overbought / Oversold threshold crosses (last two bars)
    if (
        is_num(rsi0)
        and is_num(lup0)
        and is_num(rsi1)
        and is_num(lup1)
        and rsi0 <= lup0
        and rsi1 > lup1
    ):
        tags.append(OVERBOUGHT)
    elif (
        is_num(rsi0)
        and is_num(ldn0)
        and is_num(rsi1)
        and is_num(ldn1)
        and rsi0 >= ldn0
        and rsi1 < ldn1
    ):
        tags.append(OVERSOLD)

    # Final trend tag (exactly one)
    trend_tag = NEUTRAL_TREND
    tval = r_last.get("Trend")
    if is_num(tval):
        if tval > 0:
            trend_tag = BULLISH_TREND
        elif tval < 0:
            trend_tag = BEARISH_TREND
    else:
        if is_num(sfast1) and is_num(sslow1):
            if sfast1 > sslow1:
                trend_tag = BULLISH_TREND
            elif sfast1 < sslow1:
                trend_tag = BEARISH_TREND
        elif is_num(rsi1) and is_num(lmid1):
            if rsi1 > lmid1:
                trend_tag = BULLISH_TREND
            elif rsi1 < lmid1:
                trend_tag = BEARISH_TREND

    tags.append(trend_tag)
    return tags


# === END metro_advanced_signal.py ===


# === BEGIN metro_fixed_signal.py ===
def signal_metro_fixed(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Basic validation
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    required_cols = {"RSI", "StepRSI_slow"}
    if not required_cols.issubset(df.columns):
        return [INCONCLUSIVE, NEUTRAL_TREND]

    rsi = df["RSI"]
    slow = df["StepRSI_slow"]

    # Last two bars
    rsi_prev = rsi.iloc[-2]
    rsi_curr = rsi.iloc[-1]
    slow_prev = slow.iloc[-2]
    slow_curr = slow.iloc[-1]

    # Robustness to NaNs
    if (
        pd.isna(rsi_prev)
        or pd.isna(rsi_curr)
        or pd.isna(slow_prev)
        or pd.isna(slow_curr)
    ):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Signal: crossover of RSI above/below StepRSI_slow (FRAMA-style interpretation)
    bullish_cross = (rsi_prev <= slow_prev) and (rsi_curr > slow_curr)
    bearish_cross = (rsi_prev >= slow_prev) and (rsi_curr < slow_curr)

    if bullish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Overbought/Oversold context on current bar
    if rsi_curr >= 70.0:
        tags.append(OVERBOUGHT)
    elif rsi_curr <= 30.0:
        tags.append(OVERSOLD)

    # Trend: current RSI relative to StepRSI_slow
    if rsi_curr > slow_curr:
        trend_tag = BULLISH_TREND
    elif rsi_curr < slow_curr:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END metro_fixed_signal.py ===


# === BEGIN momentum_candles_modified_w_atr_signal.py ===
def signal_momentum_candles_modified_w_atr(df: pd.DataFrame) -> list[str]:
    # Validate input and required columns
    if df is None or not isinstance(df, pd.DataFrame):
        return [NO_SIGNAL, NEUTRAL_TREND]
    for col in ("Value", "Threshold_Pos", "Threshold_Neg"):
        if col not in df.columns:
            return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    v = df["Value"].astype(float)
    tp = df["Threshold_Pos"].astype(float)
    tn = df["Threshold_Neg"].astype(float)

    # Current bar
    v_curr = v.iloc[-1]
    tp_curr = tp.iloc[-1]
    tn_curr = tn.iloc[-1]

    # Previous bar (for cross logic)
    v_prev = v.iloc[-2] if n >= 2 else pd.NA
    tp_prev = tp.iloc[-2] if n >= 2 else pd.NA
    tn_prev = tn.iloc[-2] if n >= 2 else pd.NA

    tags: list[str] = []

    # Signal via threshold cross has priority, then zero-line cross
    signal_added = False
    if n >= 2:
        # Threshold cross checks (require non-NaN)
        if (
            pd.notna(v_prev)
            and pd.notna(v_curr)
            and pd.notna(tp_prev)
            and pd.notna(tp_curr)
        ):
            if (v_prev < tp_prev) and (v_curr >= tp_curr):
                tags.append(BULLISH_SIGNAL)
                signal_added = True
            elif (v_prev > tn_prev) and (v_curr <= tn_curr):
                tags.append(BEARISH_SIGNAL)
                signal_added = True

        # Zero-line cross checks if no threshold cross
        if not signal_added and pd.notna(v_prev) and pd.notna(v_curr):
            if (v_prev <= 0.0) and (v_curr > 0.0):
                tags.append(BULLISH_SIGNAL)
                signal_added = True
            elif (v_prev >= 0.0) and (v_curr < 0.0):
                tags.append(BEARISH_SIGNAL)
                signal_added = True

    if not signal_added:
        tags.append(NO_SIGNAL)

    # Trend determination per interpretation: above zero bullish, below zero bearish
    if pd.isna(v_curr):
        trend_tag = NEUTRAL_TREND
    elif v_curr > 0.0:
        trend_tag = BULLISH_TREND
    elif v_curr < 0.0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END momentum_candles_modified_w_atr_signal.py ===


# === BEGIN momentum_candles_w_atr_signal.py ===
def signal_momentum_candles_w_atr(df: pd.DataFrame) -> list[str]:
    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    required_cols = ["BullOpen", "BullClose", "BearOpen", "BearClose"]
    if not all(col in df.columns for col in required_cols):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Current bar signal
    last = df.iloc[-1]
    is_bullish = pd.notna(last["BullClose"])
    is_bearish = pd.notna(last["BearClose"])

    tags: list[str] = []
    if is_bullish and not is_bearish:
        tags.append(BULLISH_SIGNAL)
    elif is_bearish and not is_bullish:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend from the last two bars (consecutive candles strengthen trend)
    last2 = df.tail(2)
    bull2 = last2["BullClose"].notna()
    bear2 = last2["BearClose"].notna()

    if len(last2) >= 2:
        if bool(bull2.iloc[-2]) and bool(bull2.iloc[-1]):
            trend = BULLISH_TREND
        elif bool(bear2.iloc[-2]) and bool(bear2.iloc[-1]):
            trend = BEARISH_TREND
        else:
            if is_bullish and not is_bearish:
                trend = BULLISH_TREND
            elif is_bearish and not is_bullish:
                trend = BEARISH_TREND
            else:
                trend = NEUTRAL_TREND
    else:
        if is_bullish and not is_bearish:
            trend = BULLISH_TREND
        elif is_bearish and not is_bullish:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND

    tags.append(trend)
    return tags


# === END momentum_candles_w_atr_signal.py ===


# === BEGIN precision_trend_histogram_signal.py ===
def signal_precision_trend_histogram(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or df.empty or not {"Up", "Down", "Trend"}.issubset(df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    # Determine trend tag from the latest bar
    last_trend_val = df["Trend"].iloc[-1] if n >= 1 else float("nan")
    if pd.notna(last_trend_val):
        if last_trend_val > 0:
            trend_tag = BULLISH_TREND
        elif last_trend_val < 0:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Need at least two bars for cross/transition decisions
    if n < 2:
        return [NO_SIGNAL, trend_tag]

    last2 = df.iloc[-2:]
    up_prev = pd.notna(last2["Up"].iloc[0]) and float(last2["Up"].iloc[0]) > 0.0
    up_now = pd.notna(last2["Up"].iloc[1]) and float(last2["Up"].iloc[1]) > 0.0
    down_prev = pd.notna(last2["Down"].iloc[0]) and float(last2["Down"].iloc[0]) > 0.0
    down_now = pd.notna(last2["Down"].iloc[1]) and float(last2["Down"].iloc[1]) > 0.0

    # Signal logic: new appearance of Up/Down on the latest bar
    if up_now and not up_prev:
        signal = BULLISH_SIGNAL
    elif down_now and not down_prev:
        signal = BEARISH_SIGNAL
    else:
        signal = NO_SIGNAL

    return [signal, trend_tag]


# === END precision_trend_histogram_signal.py ===


# === BEGIN price_momentum_oscillator_signal.py ===
def signal_price_momentum_oscillator(df: pd.DataFrame) -> list[str]:
    # Expecting columns: 'PMO', 'Signal'
    # try:
    pmo = df["PMO"].astype(float)
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(pmo)
    pmo_curr = pmo.iloc[-1] if n >= 1 else float("nan")
    pmo_prev = pmo.iloc[-2] if n >= 2 else float("nan")

    tags: list[str] = []

    # Zero-line cross logic (TSI-style interpretation)
    has_vals = pd.notna(pmo_curr) and pd.notna(pmo_prev)
    bull_cross = has_vals and (pmo_prev <= 0.0) and (pmo_curr > 0.0)
    bear_cross = has_vals and (pmo_prev >= 0.0) and (pmo_curr < 0.0)

    if bull_cross:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag from current PMO value
    if pd.isna(pmo_curr):
        trend_tag = NEUTRAL_TREND
    elif pmo_curr > 0.0:
        trend_tag = BULLISH_TREND
    elif pmo_curr < 0.0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END price_momentum_oscillator_signal.py ===


# === BEGIN qqe_with_alerts_signal.py ===
def signal_qqe_with_alerts(df: pd.DataFrame) -> list[str]:
    # Validate inputs and required columns
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]
    required_cols = ["QQE_RSI_MA", "QQE_TrendLevel"]
    if not all(col in df.columns for col in required_cols):
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    rsi = pd.to_numeric(df["QQE_RSI_MA"], errors="coerce")
    tr = pd.to_numeric(df["QQE_TrendLevel"], errors="coerce")

    # Last two bars
    rsi_prev = rsi.iat[-2]
    rsi_cur = rsi.iat[-1]
    tr_prev = tr.iat[-2]
    tr_cur = tr.iat[-1]

    # Initialize tags
    tags: list[str] = []

    # Cross logic between Fast (RSI_MA) and Slow (TrendLevel)
    has_pair_prev = pd.notna(rsi_prev) and pd.notna(tr_prev)
    has_pair_cur = pd.notna(rsi_cur) and pd.notna(tr_cur)
    cross_up = False
    cross_down = False

    if has_pair_prev and has_pair_cur:
        cross_up = (rsi_prev <= tr_prev) and (rsi_cur > tr_cur)
        cross_down = (rsi_prev >= tr_prev) and (rsi_cur < tr_cur)

    # Fallback to midline (50) crosses if TrendLevel is unavailable
    mid_up = False
    mid_down = False
    if not (cross_up or cross_down) and pd.notna(rsi_prev) and pd.notna(rsi_cur):
        mid_up = (rsi_prev <= 50) and (rsi_cur > 50)
        mid_down = (rsi_prev >= 50) and (rsi_cur < 50)

    # Signal tagging
    if cross_up or mid_up:
        tags.append(BULLISH_SIGNAL)
    elif cross_down or mid_down:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Overbought/Oversold context
    if pd.notna(rsi_cur):
        if rsi_cur >= 70:
            tags.append(OVERBOUGHT)
        elif rsi_cur <= 30:
            tags.append(OVERSOLD)

    # Trend tagging (exactly one)
    if pd.notna(rsi_cur) and pd.notna(tr_cur):
        if rsi_cur > tr_cur:
            tags.append(BULLISH_TREND)
        elif rsi_cur < tr_cur:
            tags.append(BEARISH_TREND)
        else:
            # Equal -> use midline as tie-breaker
            if rsi_cur > 50:
                tags.append(BULLISH_TREND)
            elif rsi_cur < 50:
                tags.append(BEARISH_TREND)
            else:
                tags.append(NEUTRAL_TREND)
    else:
        # Fallback to RSI vs midline if TrendLevel missing
        if pd.notna(rsi_cur):
            if rsi_cur > 50:
                tags.append(BULLISH_TREND)
            elif rsi_cur < 50:
                tags.append(BEARISH_TREND)
            else:
                tags.append(NEUTRAL_TREND)
        else:
            tags.append(NEUTRAL_TREND)

    return tags


# === END qqe_with_alerts_signal.py ===


# === BEGIN range_filter_modified_signal.py ===
def signal_range_filter_modified(df: pd.DataFrame) -> list[str]:
    # Robust defaults
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.shape[0] < 2
        or "LineCenter" not in df.columns
    ):
        return [NO_SIGNAL, NEUTRAL_TREND]

    lc = df["LineCenter"]
    if lc.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = lc.iloc[-2]
    curr = lc.iloc[-1]

    signal = NO_SIGNAL
    trend = NEUTRAL_TREND

    if pd.notna(prev) and pd.notna(curr):
        if curr > prev:
            signal = BULLISH_SIGNAL
            trend = BULLISH_TREND
        elif curr < prev:
            signal = BEARISH_SIGNAL
            trend = BEARISH_TREND
        else:
            signal = NO_SIGNAL
            trend = NEUTRAL_TREND
    else:
        signal = NO_SIGNAL
        trend = NEUTRAL_TREND

    return [signal, trend]


# === END range_filter_modified_signal.py ===


# === BEGIN rwi_btf_signal.py ===
def signal_rwi_btf(df: pd.DataFrame) -> list[str]:
    # Extract series safely
    rwi_h = pd.to_numeric(
        df.get("RWIH", pd.Series(index=df.index, dtype=float)), errors="coerce"
    )
    rwi_l = pd.to_numeric(
        df.get("RWIL", pd.Series(index=df.index, dtype=float)), errors="coerce"
    )

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Get last two values (use NaNs when not available)
    if n >= 2:
        h_prev, h_curr = rwi_h.iloc[-2], rwi_h.iloc[-1]
        l_prev, l_curr = rwi_l.iloc[-2], rwi_l.iloc[-1]
    else:
        h_prev, h_curr = pd.NA, rwi_h.iloc[-1] if len(rwi_h) else pd.NA
        l_prev, l_curr = pd.NA, rwi_l.iloc[-1] if len(rwi_l) else pd.NA

    thr = 1.0

    # Helper checks robust to NaNs
    def valid(x):
        return pd.notna(x)

    def crossed_up(a, b, level):
        return valid(a) and valid(b) and a <= level and b > level

    def ge(a, b):
        return valid(a) and valid(b) and a >= b

    def gt(a, b):
        return valid(a) and valid(b) and a > b

    def le(a, b):
        return valid(a) and valid(b) and a <= b

    # Threshold cross conditions (last two bars)
    bull_thr_cross = crossed_up(h_prev, h_curr, thr)
    bear_thr_cross = crossed_up(l_prev, l_curr, thr)

    # Dominance cross conditions (RWIH vs RWIL), confirm with > 1 to filter noise
    bull_dom_cross = (
        valid(h_prev)
        and valid(l_prev)
        and valid(h_curr)
        and valid(l_curr)
        and le(h_prev, l_prev)
        and gt(h_curr, l_curr)
        and gt(h_curr, thr)
    )
    bear_dom_cross = (
        valid(h_prev)
        and valid(l_prev)
        and valid(h_curr)
        and valid(l_curr)
        and le(l_prev, h_prev)
        and gt(l_curr, h_curr)
        and gt(l_curr, thr)
    )

    # Decide signal
    signal = NO_SIGNAL
    if bull_thr_cross and (not valid(l_curr) or ge(h_curr, l_curr)):
        signal = BULLISH_SIGNAL
    elif bear_thr_cross and (not valid(h_curr) or ge(l_curr, h_curr)):
        signal = BEARISH_SIGNAL
    elif bull_dom_cross:
        signal = BULLISH_SIGNAL
    elif bear_dom_cross:
        signal = BEARISH_SIGNAL

    # Trend determination:
    # - If both components are <= 1 or NaN, trend is neutral.
    # - Otherwise, whichever component is larger above 1 sets the trend direction.
    trend = NEUTRAL_TREND
    if valid(h_curr) and valid(l_curr):
        if max(h_curr, l_curr) > thr:
            trend = BULLISH_TREND if h_curr > l_curr else BEARISH_TREND
    elif valid(h_curr):
        trend = BULLISH_TREND if h_curr > thr else NEUTRAL_TREND
    elif valid(l_curr):
        trend = BEARISH_TREND if l_curr > thr else NEUTRAL_TREND

    return [signal, trend]


# === END rwi_btf_signal.py ===


# === BEGIN sherif_hilo_signal.py ===
def signal_sherif_hilo(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Ensure required columns exist; if not, treat as inconclusive/neutral
    required_cols = {"LineUp", "LineDown"}
    if not required_cols.issubset(df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    is_up = df["LineUp"].notna()
    is_down = df["LineDown"].notna()

    n = len(df)
    if n < 2:
        # Not enough bars to detect a cross
        # Trend: derive from most recent line presence if any
        up_last_idx = df["LineUp"].last_valid_index()
        down_last_idx = df["LineDown"].last_valid_index()

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
    up_last_idx = df["LineUp"].last_valid_index()
    down_last_idx = df["LineDown"].last_valid_index()

    pos_up = df.index.get_loc(up_last_idx) if up_last_idx is not None else -1
    pos_down = df.index.get_loc(down_last_idx) if down_last_idx is not None else -1

    if pos_up > pos_down:
        tags.append(BULLISH_TREND)
    elif pos_down > pos_up:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END sherif_hilo_signal.py ===


# === BEGIN silence_signal.py ===
def signal_silence(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or df.empty or "Volatility" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    vol = pd.to_numeric(df["Volatility"], errors="coerce")

    # Require the last bar to be valid
    if len(vol) == 0 or pd.isna(vol.iloc[-1]):
        return [NO_SIGNAL, NEUTRAL_TREND]

    curr = float(vol.iloc[-1])
    prev = float(vol.iloc[-2]) if len(vol) >= 2 and not pd.isna(vol.iloc[-2]) else None

    # Thresholds on 0..100 (reversed) scale: higher => quieter, lower => more active
    low_vol_threshold = 70.0  # quiet market
    high_vol_threshold = 30.0  # active market
    epsilon = 1.0  # change sensitivity

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
            tags.append(INCREASING_VOLATILITY)  # reversed scale: down => vol expanding
        elif curr > prev + epsilon:
            tags.append(DECREASING_VOLATILITY)  # reversed scale: up => vol contracting
        else:
            tags.append(STABLE_VOLATILITY)
    else:
        tags.append(INCONCLUSIVE)

    # This indicator is a filter; trend is neutral
    tags.append(NEUTRAL_TREND)

    return tags


# === END silence_signal.py ===


# === BEGIN sinewma_signal.py ===
def signal_sinewma(series: pd.Series) -> list[str]:
    if series is None or not isinstance(series, pd.Series) or series.size < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev_val = series.iloc[-2]
    curr_val = series.iloc[-1]

    if pd.isna(prev_val) or pd.isna(curr_val):
        trend = NEUTRAL_TREND
    else:
        if curr_val > prev_val:
            trend = BULLISH_TREND
        elif curr_val < prev_val:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND

    return [NO_SIGNAL, trend]


# === END sinewma_signal.py ===


# === BEGIN smoothed_momentum_signal.py ===
def signal_smoothed_momentum(df: pd.DataFrame) -> list[str]:
    # Expecting DataFrame with column 'SM' from SmoothedMomentum(...)
    if df is None or df.empty or "SM" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = df["SM"].astype(float)
    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = s.iloc[-2]
    curr = s.iloc[-1]

    if pd.isna(prev) or pd.isna(curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    baseline = 100.0  # Equivalent to "zero line" for this momentum measure

    tags: list[str] = []

    # Signals: cross of baseline using the last two bars
    if prev <= baseline and curr > baseline:
        tags.append(BULLISH_SIGNAL)
    elif prev >= baseline and curr < baseline:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag: based on current position relative to baseline
    if curr > baseline:
        tags.append(BULLISH_TREND)
    elif curr < baseline:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END smoothed_momentum_signal.py ===


# === BEGIN smoothstep_signal.py ===
def signal_smoothstep(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    s = (
        pd.to_numeric(df.get("SmoothStep"), errors="coerce")
        if isinstance(df, pd.DataFrame)
        else None
    )
    if s is None or s.shape[0] == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    curr = s.iloc[-1] if s.shape[0] >= 1 else float("nan")
    prev = s.iloc[-2] if s.shape[0] >= 2 else float("nan")

    threshold = 0.5

    # Signal: cross of SmoothStep over/under the midline (0.5)
    signal = None
    if pd.notna(prev) and pd.notna(curr):
        crossed_up = (prev <= threshold) and (curr > threshold)
        crossed_down = (prev >= threshold) and (curr < threshold)
        if crossed_up:
            signal = BULLISH_SIGNAL
        elif crossed_down:
            signal = BEARISH_SIGNAL

    if signal is None:
        tags.append(NO_SIGNAL)
    else:
        tags.append(signal)

    # Trend: position relative to midline
    trend = NEUTRAL_TREND
    if pd.notna(curr):
        if curr > threshold:
            trend = BULLISH_TREND
        elif curr < threshold:
            trend = BEARISH_TREND

    tags.append(trend)
    return tags


# === END smoothstep_signal.py ===

# === BEGIN stiffness_indicator_signal.py ===


def signal_stiffness_indicator(df: pd.DataFrame) -> list[str]:
    # Validate input
    # try:
    s = pd.to_numeric(df["Stiffness"], errors="coerce")
    sig = pd.to_numeric(df["Signal"], errors="coerce")
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s1, s2 = s.iloc[-2], s.iloc[-1]
    sig1, sig2 = sig.iloc[-2], sig.iloc[-1]

    if pd.isna(s1) or pd.isna(s2) or pd.isna(sig1) or pd.isna(sig2):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Volatility trend via cross/slope of Stiffness vs its signal
    cross_up = s1 <= sig1 and s2 > sig2
    cross_down = s1 >= sig1 and s2 < sig2
    delta = s2 - s1

    if cross_up:
        vol_tag = DECREASING_VOLATILITY
    elif cross_down:
        vol_tag = INCREASING_VOLATILITY
    else:
        if delta > 0:
            vol_tag = DECREASING_VOLATILITY
        elif delta < 0:
            vol_tag = INCREASING_VOLATILITY
        else:
            vol_tag = STABLE_VOLATILITY

    # Regime/extreme assessment on recent history
    recent = s.dropna()
    if not recent.empty:
        window = min(len(recent), 200)
        recent = recent.iloc[-window:]
        p10 = recent.quantile(0.10)
        p90 = recent.quantile(0.90)

        # Extreme lows may precede volatility expansions -> prioritize INCREASING_VOLATILITY
        if s2 <= p10:
            vol_tag = INCREASING_VOLATILITY
        elif s2 >= p90:
            vol_tag = LOW_VOLATILITY

    return [vol_tag, NEUTRAL_TREND]


# === END stiffness_indicator_signal.py ===


# === BEGIN supertrend_signal.py ===
def signal_supertrend(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "Trend" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    t = df["Trend"]
    n = len(t)

    # Determine current trend tag
    curr = t.iloc[-1] if n >= 1 else float("nan")
    if pd.isna(curr):
        trend_tag = NEUTRAL_TREND
    elif curr > 0:
        trend_tag = BULLISH_TREND
    elif curr < 0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Determine signal based on last two bars (flip detection)
    if n >= 2:
        prev = t.iloc[-2]
        if pd.notna(prev) and pd.notna(curr):
            if prev < 0 and curr > 0:
                signal = BULLISH_SIGNAL
            elif prev > 0 and curr < 0:
                signal = BEARISH_SIGNAL
            else:
                signal = NO_SIGNAL
        else:
            signal = NO_SIGNAL
    else:
        signal = NO_SIGNAL

    return [signal, trend_tag]


# === END supertrend_signal.py ===


# === BEGIN t3_ma_signal.py ===
def signal_t3_ma(series: pd.Series) -> list[str]:
    # Robust to NaNs and short series
    if series is None or not isinstance(series, pd.Series):
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = series.dropna()
    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev_val = s.iloc[-2]
    curr_val = s.iloc[-1]

    # Determine trend from last two bars of T3
    if pd.isna(prev_val) or pd.isna(curr_val):
        trend_tag = NEUTRAL_TREND
    elif curr_val > prev_val:
        trend_tag = BULLISH_TREND
    elif curr_val < prev_val:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Without price series, we cannot confirm price-vs-baseline cross; emit NO_SIGNAL
    return [NO_SIGNAL, trend_tag]


# === END t3_ma_signal.py ===


# === BEGIN tether_line_signal.py ===
def signal_tether_line(df: pd.DataFrame) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if not {"AboveCenter", "BelowCenter"}.issubset(df.columns):
        return [NO_SIGNAL, NEUTRAL_TREND]

    last2 = df.tail(2)
    prev_above = pd.notna(last2["AboveCenter"].iloc[0])
    prev_below = pd.notna(last2["BelowCenter"].iloc[0])
    curr_above = pd.notna(last2["AboveCenter"].iloc[1])
    curr_below = pd.notna(last2["BelowCenter"].iloc[1])

    cross_up = prev_below and curr_above
    cross_down = prev_above and curr_below

    if cross_up:
        tags = [BULLISH_SIGNAL]
    elif cross_down:
        tags = [BEARISH_SIGNAL]
    else:
        tags = [NO_SIGNAL]

    if curr_above and not curr_below:
        tags.append(BULLISH_TREND)
    elif curr_below and not curr_above:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END tether_line_signal.py ===


# === BEGIN theturtletradingchannel_signal.py ===
def signal_theturtletradingchannel(df: pd.DataFrame) -> list[str]:
    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    bull = (
        df["BullChange"]
        if "BullChange" in df.columns
        else pd.Series([pd.NA] * n, index=df.index)
    )
    bear = (
        df["BearChange"]
        if "BearChange" in df.columns
        else pd.Series([pd.NA] * n, index=df.index)
    )
    upper = (
        df["UpperLine"]
        if "UpperLine" in df.columns
        else pd.Series([pd.NA] * n, index=df.index)
    )
    lower = (
        df["LowerLine"]
        if "LowerLine" in df.columns
        else pd.Series([pd.NA] * n, index=df.index)
    )

    # Last two bars logic
    bull_now = pd.notna(bull.iloc[-1])
    bear_now = pd.notna(bear.iloc[-1])
    bull_prev = pd.notna(bull.iloc[-2]) if n >= 2 else False
    bear_prev = pd.notna(bear.iloc[-2]) if n >= 2 else False

    bull_signal = bool(bull_now and not bull_prev)
    bear_signal = bool(bear_now and not bear_prev)

    tags: list[str] = []
    if bull_signal:
        tags.append(BULLISH_SIGNAL)
    elif bear_signal:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend assessment from current regime lines
    if pd.notna(upper.iloc[-1]):
        trend_tag = BULLISH_TREND
    elif pd.notna(lower.iloc[-1]):
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END theturtletradingchannel_signal.py ===


# === BEGIN tii_signal.py ===
def signal_tii(series: pd.Series) -> list[str]:
    up_th = 60.0
    down_th = 40.0

    # Handle insufficient data
    if series is None or len(series) < 1:
        return [NO_SIGNAL, NEUTRAL_TREND]
    last = series.iloc[-1]

    # Determine trend from the latest value
    if pd.isna(last):
        trend_tag = NEUTRAL_TREND
    elif last > up_th:
        trend_tag = BULLISH_TREND
    elif last < down_th:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Need two bars for cross signals
    if len(series) < 2 or pd.isna(last) or pd.isna(series.iloc[-2]):
        return [NO_SIGNAL, trend_tag]

    prev = series.iloc[-2]

    bullish_cross = (prev <= up_th) and (last > up_th)
    bearish_cross = (prev >= down_th) and (last < down_th)

    if bullish_cross:
        return [BULLISH_SIGNAL, trend_tag]
    if bearish_cross:
        return [BEARISH_SIGNAL, trend_tag]

    return [NO_SIGNAL, trend_tag]


# === END tii_signal.py ===


# === BEGIN top_bottom_nr_signal.py ===
def signal_top_bottom_nr(df: pd.DataFrame) -> list[str]:
    # Robustness checks
    if df is None or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    ls = df.get("LongSignal")
    ss = df.get("ShortSignal")
    if ls is None or ss is None or len(ls) < 2 or len(ss) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    l_prev, l_curr = ls.iloc[-2], ls.iloc[-1]
    s_prev, s_curr = ss.iloc[-2], ss.iloc[-1]

    valid_l = pd.notna(l_prev) and pd.notna(l_curr)
    valid_s = pd.notna(s_prev) and pd.notna(s_curr)

    if not (valid_l or valid_s):
        return [NO_SIGNAL, NEUTRAL_TREND]

    reset_long = bool(valid_l and (l_curr < l_prev))
    reset_short = bool(valid_s and (s_curr < s_prev))

    tags: list[str] = []
    if reset_long or reset_short:
        tags.append(INCREASING_VOLATILITY)
    else:
        tags.append(DECREASING_VOLATILITY)

    tags.append(NEUTRAL_TREND)
    return tags


# === END top_bottom_nr_signal.py ===


# === BEGIN tp_signal.py ===
def signal_tp(df: pd.DataFrame) -> list[str]:
    # Expecting df with columns ['TP','Up','Dn']
    # try:
    s = df["TP"]
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    if s is None or len(s) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last = s.iloc[-1]
    prev = s.iloc[-2] if len(s) >= 2 else pd.NA

    signal_tag = NO_SIGNAL
    if pd.notna(last) and pd.notna(prev):
        if prev <= 0 and last > 0:
            signal_tag = BULLISH_SIGNAL
        elif prev >= 0 and last < 0:
            signal_tag = BEARISH_SIGNAL

    trend_tag = NEUTRAL_TREND
    if pd.notna(last):
        if last > 0:
            trend_tag = BULLISH_TREND
        elif last < 0:
            trend_tag = BEARISH_TREND

    return [signal_tag, trend_tag]


# === END tp_signal.py ===

# === BEGIN trend_akkam_signal.py ===


def signal_trend_akkam(df: pd.DataFrame) -> list[str]:
    # Expecting DataFrame with at least 'TrStop' column
    if not isinstance(df, pd.DataFrame) or "TrStop" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(df["TrStop"], errors="coerce")
    n = len(s)

    if n < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    t_prev = s.iloc[-2]
    t_curr = s.iloc[-1]

    if pd.isna(t_prev) or pd.isna(t_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Current slope (line color proxy)
    delta_curr = t_curr - t_prev
    curr_slope = 1 if delta_curr > 0 else (-1 if delta_curr < 0 else 0)

    # Previous slope to detect "turns green/red" events
    prev_slope = None
    if n >= 3:
        t_prev2 = s.iloc[-3]
        if pd.notna(t_prev2):
            delta_prev = t_prev - t_prev2
            prev_slope = 1 if delta_prev > 0 else (-1 if delta_prev < 0 else 0)

    # Determine signal on slope change
    signal = NO_SIGNAL
    if prev_slope is not None:
        if curr_slope > 0 and prev_slope <= 0:
            signal = BULLISH_SIGNAL
        elif curr_slope < 0 and prev_slope >= 0:
            signal = BEARISH_SIGNAL

    # Trend tag from current slope
    if curr_slope > 0:
        trend = BULLISH_TREND
    elif curr_slope < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    return [signal, trend]


# === END trend_akkam_signal.py ===


# === BEGIN trend_direction__force_index___smoothed_4_signal.py ===
def signal_trend_direction_force_index_smoothed_4(df: pd.DataFrame) -> list[str]:
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or len(df) < 2
        or "TDF" not in df.columns
    ):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tdf = df["TDF"]
    tu = (
        df["TriggerUp"]
        if "TriggerUp" in df.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )
    td = (
        df["TriggerDown"]
        if "TriggerDown" in df.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )

    prev_tdf = tdf.iloc[-2]
    curr_tdf = tdf.iloc[-1]
    prev_tu = tu.iloc[-2] if len(tu) >= 2 else tu.iloc[-1]
    curr_tu = tu.iloc[-1]
    prev_td = td.iloc[-2] if len(td) >= 2 else td.iloc[-1]
    curr_td = td.iloc[-1]

    def is_num(x) -> bool:
        return x is not None and pd.notna(x)

    tags: list[str] = []

    bullish_cross = False
    bearish_cross = False
    if all(is_num(x) for x in [prev_tdf, curr_tdf, prev_tu, curr_tu]):
        bullish_cross = (prev_tdf <= prev_tu) and (curr_tdf > curr_tu)
    if all(is_num(x) for x in [prev_tdf, curr_tdf, prev_td, curr_td]):
        bearish_cross = (prev_tdf >= prev_td) and (curr_tdf < curr_td)

    if bullish_cross and not bearish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross and not bullish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    trend_tag = NEUTRAL_TREND
    if "Trend" in df.columns and is_num(df["Trend"].iloc[-1]):
        t = df["Trend"].iloc[-1]
        if t > 0:
            trend_tag = BULLISH_TREND
        elif t < 0:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        if is_num(curr_tdf) and is_num(curr_tu) and is_num(curr_td):
            if curr_tdf > curr_tu:
                trend_tag = BULLISH_TREND
            elif curr_tdf < curr_td:
                trend_tag = BEARISH_TREND

    tags.append(trend_tag)
    return tags


# === END trend_direction__force_index___smoothed_4_signal.py ===

# === BEGIN trend_lord_nrp_indicator_signal.py ===


def signal_trend_lord_nrp_indicator(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if "Buy" not in df.columns or "Sell" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    b_prev = df["Buy"].iloc[-2]
    b_curr = df["Buy"].iloc[-1]
    s_prev = df["Sell"].iloc[-2]
    s_curr = df["Sell"].iloc[-1]

    # Signal: crossover between Buy (ma1/slot) and Sell (ma2) on last two bars
    signal_tag = NO_SIGNAL
    if pd.notna(b_prev) and pd.notna(b_curr) and pd.notna(s_prev) and pd.notna(s_curr):
        crossed_up = b_prev <= s_prev and b_curr > s_curr
        crossed_down = b_prev >= s_prev and b_curr < s_curr
        if crossed_up:
            signal_tag = BULLISH_SIGNAL
        elif crossed_down:
            signal_tag = BEARISH_SIGNAL

    # Trend: slope of Sell (ma2) on last two bars
    if pd.notna(s_prev) and pd.notna(s_curr):
        if s_curr > s_prev:
            trend_tag = BULLISH_TREND
        elif s_curr < s_prev:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        trend_tag = NEUTRAL_TREND

    return [signal_tag, trend_tag]


# === END trend_lord_nrp_indicator_signal.py ===


# === BEGIN trendcontinuation2_signal.py ===
def signal_trendcontinuation2(df: pd.DataFrame) -> list[str]:
    # Validate input
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    pos = df.get("TrendContinuation2_Pos")
    neg = df.get("TrendContinuation2_Neg")
    if pos is None or neg is None:
        return [NO_SIGNAL, NEUTRAL_TREND]

    reading = pos - neg

    prev = reading.iloc[-2]
    curr = reading.iloc[-1]

    # Determine trend tag from current value
    if pd.notna(curr) and curr > 0:
        trend_tag = BULLISH_TREND
    elif pd.notna(curr) and curr < 0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Signal on zero-cross using last two bars
    signal_tag = NO_SIGNAL
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            signal_tag = BULLISH_SIGNAL
        elif prev >= 0 and curr < 0:
            signal_tag = BEARISH_SIGNAL

    return [signal_tag, trend_tag]


# === END trendcontinuation2_signal.py ===


# === BEGIN trimagen_signal.py ===
def signal_trimagen(series: pd.Series) -> list[str]:
    tags: list[str] = []
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = series.astype(float)

    # Determine signal based on slope cross (using last two slopes -> needs last 3 values)
    signal_emitted = False
    if len(s) >= 3:
        v3 = s.iloc[-3]
        v2 = s.iloc[-2]
        v1 = s.iloc[-1]
        if pd.notna(v3) and pd.notna(v2) and pd.notna(v1):
            delta_prev = v2 - v3
            delta_curr = v1 - v2
            if delta_prev <= 0 and delta_curr > 0:
                tags.append(BULLISH_SIGNAL)
                signal_emitted = True
            elif delta_prev >= 0 and delta_curr < 0:
                tags.append(BEARISH_SIGNAL)
                signal_emitted = True

    if not signal_emitted:
        tags.append(NO_SIGNAL)

    # Trend tag from latest slope (last two bars)
    v2 = s.iloc[-2]
    v1 = s.iloc[-1]
    trend = NEUTRAL_TREND
    if pd.notna(v2) and pd.notna(v1):
        d = v1 - v2
        if d > 0:
            trend = BULLISH_TREND
        elif d < 0:
            trend = BEARISH_TREND

    tags.append(trend)
    return tags


# === END trimagen_signal.py ===


# === BEGIN true_strength_index_signal.py ===
def signal_true_strength_index(series: pd.Series) -> list[str]:
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    curr = series.iloc[-1]
    prev = series.iloc[-2]

    if pd.isna(curr) or pd.isna(prev):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = []

    bullish_cross = (prev <= 0) and (curr > 0)
    bearish_cross = (prev >= 0) and (curr < 0)

    if bullish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    if curr > 0:
        tags.append(BULLISH_TREND)
    elif curr < 0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END true_strength_index_signal.py ===


# === BEGIN ttf_signal.py ===
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


# === END ttf_signal.py ===

# === BEGIN ttms_signal.py ===


def signal_ttms(df: pd.DataFrame) -> list[str]:
    # Validate input
    # try:
    up = df["TTMS_Up"]
    dn = df["TTMS_Dn"]
    alert = df["Alert"]
    noalert = df["NoAlert"]
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    up2 = up.iloc[-2:]
    dn2 = dn.iloc[-2:]
    alert2 = alert.iloc[-2:]
    noalert2 = noalert.iloc[-2:]

    def sum_ignore_nan(a, b):
        a_valid = pd.notna(a)
        b_valid = pd.notna(b)
        if not a_valid and not b_valid:
            return float("nan")
        s = 0.0
        if a_valid:
            s += float(a)
        if b_valid:
            s += float(b)
        return s

    t_prev = sum_ignore_nan(up2.iloc[0], dn2.iloc[0])
    t_now = sum_ignore_nan(up2.iloc[1], dn2.iloc[1])

    prev_on = pd.notna(alert2.iloc[0])
    prev_off = pd.notna(noalert2.iloc[0])
    now_on = pd.notna(alert2.iloc[1])
    now_off = pd.notna(noalert2.iloc[1])

    # If latest bar has no valid squeeze state, return neutral
    if not (now_on or now_off):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = [NO_SIGNAL]

    # Current volatility state
    if now_on:
        tags.append(LOW_VOLATILITY)
    elif now_off:
        tags.append(HIGH_VOLATILITY)

    # Transition-based volatility change
    transition = None
    if prev_on or prev_off:
        if now_off and prev_on:
            transition = "inc"  # squeeze released -> increasing vol
        elif now_on and prev_off:
            transition = "dec"  # squeeze engaged -> decreasing vol

    eps = 1e-12
    if transition == "inc":
        tags.append(INCREASING_VOLATILITY)
    elif transition == "dec":
        tags.append(DECREASING_VOLATILITY)
    else:
        # Slope-based volatility change if both values are valid
        if pd.notna(t_prev) and pd.notna(t_now):
            delta = t_now - t_prev
            if abs(delta) <= eps:
                tags.append(STABLE_VOLATILITY)
            else:
                if now_on:
                    tags.append(
                        DECREASING_VOLATILITY if delta > 0 else INCREASING_VOLATILITY
                    )
                else:
                    tags.append(
                        INCREASING_VOLATILITY if delta < 0 else DECREASING_VOLATILITY
                    )

    # This indicator is volatility-focused; no directional bias
    tags.append(NEUTRAL_TREND)
    return tags


# === END ttms_signal.py ===


# === BEGIN vidya_signal.py ===
def signal_vidya(series: pd.Series) -> list[str]:
    # Robustness: handle empty or too-short inputs
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(series, errors="coerce")
    v_prev = s.iloc[-2]
    v_last = s.iloc[-1]

    # If either of the last two bars is NaN, be neutral
    if pd.isna(v_prev) or pd.isna(v_last):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Determine trend from the last two bars (slope of VIDYA)
    scale = max(abs(v_last), abs(v_prev), 1.0)
    eps = 1e-6 * scale

    if v_last > v_prev + eps:
        trend = BULLISH_TREND
    elif v_last < v_prev - eps:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Baseline role: emit trend; no entry signal without price-vs-baseline cross
    return [NO_SIGNAL, trend]


# === END vidya_signal.py ===

# === BEGIN volatility_ratio_signal.py ===


def signal_volatility_ratio(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or len(df) == 0 or "VR" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    vr = df["VR"]
    if vr.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last = vr.iloc[-1]
    prev = vr.iloc[-2] if len(vr) > 1 else pd.NA

    if pd.isna(last):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tol = 0.02

    if abs(last - 1.0) <= tol:
        tags.append(STABLE_VOLATILITY)
    elif last > 1.0:
        tags.append(HIGH_VOLATILITY)
    else:
        tags.append(LOW_VOLATILITY)

    if not pd.isna(prev):
        crossed_up = prev < 1.0 and last >= 1.0
        crossed_down = prev > 1.0 and last <= 1.0
        if crossed_up:
            tags.append(INCREASING_VOLATILITY)
        elif crossed_down:
            tags.append(DECREASING_VOLATILITY)
        else:
            delta = last - prev
            if delta > tol:
                tags.append(INCREASING_VOLATILITY)
            elif delta < -tol:
                tags.append(DECREASING_VOLATILITY)

    if not tags:
        tags.append(NO_SIGNAL)

    tags.append(NEUTRAL_TREND)
    # ensure uniqueness while preserving order
    tags = list(dict.fromkeys(tags))
    # ensure exactly one final trend tag
    if tags.count(NEUTRAL_TREND) > 1:
        tags = [
            t for i, t in enumerate(tags) if t != NEUTRAL_TREND or i == len(tags) - 1
        ]
    return tags


# === END volatility_ratio_signal.py ===


# === BEGIN vortex_indicator_signal.py ===
def signal_vortex_indicator(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Basic validation
    if df is None or df.empty or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Identify VI+ and VI- columns (supports any length suffix)
    cols_lower = {c.lower(): c for c in df.columns}
    plus_candidates = [
        orig for low, orig in cols_lower.items() if low.startswith("vi_plus")
    ]
    minus_candidates = [
        orig for low, orig in cols_lower.items() if low.startswith("vi_minus")
    ]

    if not plus_candidates or not minus_candidates:
        return [NO_SIGNAL, NEUTRAL_TREND]

    plus_col = plus_candidates[0]
    minus_col = minus_candidates[0]

    # Use last two bars for cross decisions
    plus_prev = df[plus_col].iloc[-2]
    plus_curr = df[plus_col].iloc[-1]
    minus_prev = df[minus_col].iloc[-2]
    minus_curr = df[minus_col].iloc[-1]

    have_prev = pd.notna(plus_prev) and pd.notna(minus_prev)
    have_curr = pd.notna(plus_curr) and pd.notna(minus_curr)

    if have_prev and have_curr:
        bull_cross = (plus_prev <= minus_prev) and (plus_curr > minus_curr)
        bear_cross = (plus_prev >= minus_prev) and (plus_curr < minus_curr)

        if bull_cross:
            tags.append(BULLISH_SIGNAL)
        elif bear_cross:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Determine current trend from the latest bar
    if pd.notna(plus_curr) and pd.notna(minus_curr):
        if plus_curr > minus_curr:
            trend_tag = BULLISH_TREND
        elif plus_curr < minus_curr:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END vortex_indicator_signal.py ===


# === BEGIN william_vix_fix_signal.py ===
def signal_william_vix_fix(series: pd.Series) -> list[str]:
    tags: list[str] = []

    if series is None or series.size == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    valid = series.dropna()
    if valid.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev_idx = valid.index[-2]
    curr_idx = valid.index[-1]
    prev = valid.iloc[-2]
    curr = valid.iloc[-1]

    # Dynamic percentile bands for extremes (fear/complacency)
    window = min(50, max(5, valid.shape[0]))
    upper_q = series.rolling(window=window, min_periods=window).quantile(0.9)
    lower_q = series.rolling(window=window, min_periods=window).quantile(0.1)

    up_prev = upper_q.loc[prev_idx] if prev_idx in upper_q.index else float("nan")
    up_curr = upper_q.loc[curr_idx] if curr_idx in upper_q.index else float("nan")
    low_prev = lower_q.loc[prev_idx] if prev_idx in lower_q.index else float("nan")
    low_curr = lower_q.loc[curr_idx] if curr_idx in lower_q.index else float("nan")

    # Extremes tags
    if pd.notna(up_curr) and curr > up_curr:
        tags.append(OVERSOLD)  # Panic/fear extreme
    elif pd.notna(low_curr) and curr < low_curr:
        tags.append(OVERBOUGHT)  # Complacency extreme

    # Volatility level tags
    if pd.notna(up_curr) and curr > up_curr:
        tags.append(HIGH_VOLATILITY)
    elif pd.notna(low_curr) and curr < low_curr:
        tags.append(LOW_VOLATILITY)

    # Volatility direction tags (based on last two bars)
    if curr > prev:
        tags.append(INCREASING_VOLATILITY)
    elif curr < prev:
        tags.append(DECREASING_VOLATILITY)
    else:
        tags.append(STABLE_VOLATILITY)

    # Signal logic (NNFX-style, use last two bars)
    # Bullish when fear spike begins to fade: either crossing back below the upper band,
    # or turning down after being above the band on the prior bar.
    bullish = False
    if pd.notna(up_prev) and pd.notna(up_curr):
        if (prev > up_prev and curr <= up_curr) or (prev > up_prev and curr < prev):
            bullish = True

    if bullish:
        tags.append(BULLISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Final trend tag (volatility tool => neutral trend context)
    tags.append(NEUTRAL_TREND)

    return tags


# === END william_vix_fix_signal.py ===


# === BEGIN wpr_ma_alerts_signal.py ===
def signal_wpr_ma_alerts(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if (
        df is None
        or len(df) == 0
        or not {"WPR", "Signal", "Cross"}.issubset(df.columns)
    ):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tail = df[["WPR", "Signal", "Cross"]].tail(2).astype(float)

    wpr_vals = tail["WPR"].to_numpy()
    cross_vals = tail["Cross"].to_numpy()

    bullish_cross = False
    bearish_cross = False
    bullish_exit_oversold = False
    bearish_exit_overbought = False

    if len(tail) == 2:
        prev_wpr, curr_wpr = wpr_vals[0], wpr_vals[1]
        prev_cross, curr_cross = cross_vals[0], cross_vals[1]

        # Cross event detection (WPR vs Signal)
        if (
            not pd.isna(prev_cross)
            and not pd.isna(curr_cross)
            and prev_cross != curr_cross
        ):
            if curr_cross == 1:
                bullish_cross = True
            elif curr_cross == -1:
                bearish_cross = True

        # Threshold exits
        if not pd.isna(prev_wpr) and not pd.isna(curr_wpr):
            if prev_wpr <= -80 and curr_wpr > -80:
                bullish_exit_oversold = True
            if prev_wpr >= -20 and curr_wpr < -20:
                bearish_exit_overbought = True

    # Primary signal
    if bullish_cross or bullish_exit_oversold:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross or bearish_exit_overbought:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Zone tags (current bar)
    curr_wpr = wpr_vals[-1] if len(wpr_vals) > 0 else float("nan")
    if not pd.isna(curr_wpr):
        if curr_wpr <= -80:
            tags.append(OVERSOLD)
        elif curr_wpr >= -20:
            tags.append(OVERBOUGHT)

    # Trend tag from latest valid Cross state (prefer current, fall back to previous)
    curr_cross_val = cross_vals[-1] if len(cross_vals) > 0 else float("nan")
    if pd.isna(curr_cross_val) or curr_cross_val == 0:
        if len(cross_vals) == 2 and not pd.isna(cross_vals[0]) and cross_vals[0] != 0:
            curr_cross_val = cross_vals[0]

    if curr_cross_val == 1:
        trend_tag = BULLISH_TREND
    elif curr_cross_val == -1:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags


# === END wpr_ma_alerts_signal.py ===

# === BEGIN xma_coloured_updated_for_nnfx_signal.py ===


def signal_xma_coloured_updated_for_nnfx(df: pd.DataFrame) -> list[str]:
    def classify(row: pd.Series) -> str:
        up = row.get("Up")
        dn = row.get("Dn")
        fl = row.get("Fl")
        up_flag = pd.notna(up)
        dn_flag = pd.notna(dn)
        fl_flag = pd.notna(fl)

        if up_flag and not dn_flag:
            return "up"
        if dn_flag and not up_flag:
            return "dn"
        # If both Up and Dn present (rare tie) or neither present, fall back to flat/neutral
        if up_flag and dn_flag:
            return "fl"
        if fl_flag:
            return "fl"
        return "fl"

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last_state = classify(df.iloc[-1])

    # Determine signal using last two bars
    signal_tag = NO_SIGNAL
    if n >= 2:
        prev_state = classify(df.iloc[-2])
        if last_state == "up" and prev_state in ("dn", "fl"):
            signal_tag = BULLISH_SIGNAL
        elif last_state == "dn" and prev_state in ("up", "fl"):
            signal_tag = BEARISH_SIGNAL

    # Trend from the latest bar
    if last_state == "up":
        trend_tag = BULLISH_TREND
    elif last_state == "dn":
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    return [signal_tag, trend_tag]


# === END xma_coloured_updated_for_nnfx_signal.py ===


# === BEGIN zerolag_macd_mq4_signal.py ===
def signal_zerolag_macd_mq4(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Validate required columns
    # try:
    macd = df["ZL_MACD"]
    sig = df["ZL_Signal"]
    # except Exception:
    #     return [NO_SIGNAL, NEUTRAL_TREND]

    if len(macd) < 2 or len(sig) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    macd_prev = macd.iloc[-2]
    macd_last = macd.iloc[-1]
    sig_prev = sig.iloc[-2]
    sig_last = sig.iloc[-1]

    if (
        pd.isna(macd_prev)
        or pd.isna(macd_last)
        or pd.isna(sig_prev)
        or pd.isna(sig_last)
    ):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Cross signals (last two bars)
    bull_cross_sig = macd_prev <= sig_prev and macd_last > sig_last
    bear_cross_sig = macd_prev >= sig_prev and macd_last < sig_last
    bull_cross_zero = macd_prev <= 0.0 and macd_last > 0.0
    bear_cross_zero = macd_prev >= 0.0 and macd_last < 0.0

    if bull_cross_sig or bull_cross_zero:
        tags.append(BULLISH_SIGNAL)
    elif bear_cross_sig or bear_cross_zero:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag (exactly one)
    if macd_last > sig_last and macd_last > 0.0:
        tags.append(BULLISH_TREND)
    elif macd_last < sig_last and macd_last < 0.0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags


# === END zerolag_macd_mq4_signal.py ===
