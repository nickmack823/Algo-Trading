from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Helper Functions

# def STDEV(price_data: pd.DataFrame, baseline: str, window: int, threshold: float, bound: str) -> pd.Series:
#     """ Calculate the Standard Deviation of the given indicator.

#     Args:
#         price_data (pd.DataFrame): A Pandas DataFrame of float values representing price data with
#             columns 'Open', 'High', 'Low', 'Close', 'Volume'.
#         baseline (str): The name of the indicator to calculate the standard deviation of.
#         window (int): The number of periods to use for the rolling window.
#         threshold (float): The number of standard deviations to use as the threshold.
#         bound (str): The bound to use for the standard deviation calculation. Can be 'upper' or 'lower'.
#     Returns:
#         pd.Series: A Pandas Series representing the standard deviation values.
#     """
#     indicator_function = baselines[baseline]['function']
#     indicator_values = indicator_function(price_data, **baselines[baseline]['default_params'])
#     stdev = indicator_values.rolling(window).std() * threshold

#     if bound == 'upper':
#         price_thresholds = indicator_values.add(stdev, axis=0)
#     else:
#         price_thresholds = indicator_values.sub(stdev, axis=0)

#     return price_thresholds


def series_wma(series, period):
    weights = np.arange(1, period + 1)
    wma = series.rolling(window=period).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )
    return wma


def series_ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def zero_lag_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Zero-Lag Exponential Moving Average (ZLEMA) for the given series and period."""
    ema = series.ewm(span=period).mean()
    zlema = (2 * ema) - ema.ewm(span=period).mean()
    return zlema


def j_tpo_value(input_prices, period, shift):
    value = 0
    arr1 = [0] + list(input_prices[shift : shift + period][::-1]) + [0]
    arr2 = [0] + list(range(1, period + 1)) + [0]
    arr3 = arr2.copy()

    for m in range(1, period):
        maxval = arr1[m]
        maxloc = m

        for j in range(m + 1, period + 1):
            if arr1[j] < maxval:
                maxval = arr1[j]
                maxloc = j

        arr1[m], arr1[maxloc] = arr1[maxloc], arr1[m]
        arr2[m], arr2[maxloc] = arr2[maxloc], arr2[m]

    m = 1
    while m < period:
        j = m + 1
        flag = True
        accum = arr3[m]

        while flag:
            if arr1[m] != arr1[j]:
                if (j - m) > 1:
                    accum = accum / (j - m)
                    for n in range(m, j):
                        arr3[n] = accum
                flag = False
            else:
                accum += arr3[j]
                j += 1

        m = j

    normalization = 12.0 / (period * (period - 1) * (period + 1))
    lenp1half = (period + 1) * 0.5

    for m in range(1, period + 1):
        value += (arr3[m] - lenp1half) * (arr2[m] - lenp1half)

    value = normalization * value

    return value


# ATR Functions


def ATR(price_data: pd.DataFrame, period: int) -> pd.Series:
    high_low = price_data["High"] - price_data["Low"]
    high_close = np.abs(price_data["High"] - price_data["Close"].shift())
    low_close = np.abs(price_data["Low"] - price_data["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)

    true_range = np.max(ranges, axis=1)

    atr = true_range.rolling(period).sum() / period
    return atr


def FilteredATR(
    price_data: pd.DataFrame, period: int = 34, ma_period: int = 34, ma_shift: int = 0
) -> pd.Series:
    """
    Calculate the Filtered ATR indicator for the given price data.

    Parameters:
    price_data (pd.DataFrame): OHLCV price data
    period (int, optional): ATR period, default is 34
    ma_period (int, optional): MA period, default is 34
    ma_shift (int, optional): Moving average shift, default is 0

    Returns:
    pd.Series: Filtered ATR values
    """
    # Calculate the True Range
    tr = pd.Series(
        np.maximum(
            price_data["High"] - price_data["Low"],
            np.maximum(
                np.abs(price_data["High"] - price_data["Close"].shift(1)),
                np.abs(price_data["Low"] - price_data["Close"].shift(1)),
            ),
            np.abs(price_data["High"] - price_data["Close"].shift(1)),
        ),
        name="TrueRange",
    )

    # Calculate the ATR
    atr = tr.rolling(window=period).mean()

    # Calculate the moving average on ATR
    atr_ma = atr.ewm(span=ma_period, adjust=False).mean()

    # Shift the moving average (if required)
    if ma_shift != 0:
        atr_ma = atr_ma.shift(ma_shift)

    # Return the calculated Filtered ATR values
    return atr_ma


# Baseline Functions


def SMA(price_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(price_data.Close).rolling(period).mean()


def EMA(price_data: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates Exponential Moving Average (EMA) for a given period and column of a pandas DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing the data to compute EMA for.
    period : int
        The period to use for computing EMA.

    Returns:
    --------
    ema : pandas Series
        A pandas Series containing the computed EMA values.
    """
    ema = price_data["Close"].ewm(span=period, adjust=False).mean()

    return ema


def WMA(price_data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate the Weighted Moving Average (WMA) using a Pandas DataFrame of OHLCV data.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        period (int): The period for the WMA calculation.

    Returns:
        _type_: A Pandas Series containing the WMA values.
    """
    price_series = price_data["Close"]
    weights = np.arange(1, period + 1)

    return price_series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def HMA(price_data: pd.DataFrame, period: int = 13) -> pd.Series:
    """Calculate the Hull Moving Average (HMA) using a Pandas DataFrame of OHLCV data.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        period (int): The period for the HMA calculation.

    Returns:
        pd.Series: A Pandas Series containing the HMA values.
    """
    close_series = price_data["Close"]
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    wma_half = series_wma(close_series, half_length)
    wma_full = series_wma(close_series, period)

    hma_series = 2 * wma_half - wma_full

    return series_wma(hma_series, sqrt_length)


def VIDYA(price_data: pd.DataFrame, period: int = 9, histper: int = 30) -> pd.Series:
    vidya = pd.Series(index=price_data.index, dtype="float64")

    def iStdDev(series, length):
        return series.rolling(window=length).std()

    std_dev_period = iStdDev(price_data["Close"], period)
    std_dev_histper = iStdDev(price_data["Close"], histper)

    i = len(price_data) - 1
    while i >= 0:
        if i < len(price_data) - histper:
            # Avoid divide-by-zero or NaN propagation
            sd_period = std_dev_period.iloc[i]
            sd_histper = std_dev_histper.iloc[i]
            if pd.notna(sd_period) and pd.notna(sd_histper) and sd_histper != 0:
                k = sd_period / sd_histper
                sc = 2.0 / (period + 1)
                prev = (
                    vidya.iloc[i + 1]
                    if pd.notna(vidya.iloc[i + 1])
                    else price_data["Close"].iloc[i]
                )
                vidya.iloc[i] = (
                    k * sc * price_data["Close"].iloc[i] + (1 - k * sc) * prev
                )
            else:
                vidya.iloc[i] = price_data["Close"].iloc[i]
        else:
            vidya.iloc[i] = price_data["Close"].iloc[i]

        i -= 1

    return vidya


def KAMA(
    price_data: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30
) -> pd.Series:
    """Calculate the Kaufman's Adaptive Moving Average (KAMA) using a Pandas DataFrame of OHLCV data.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        period (int, optional): The period to calculate for. Defaults to 10.
        fast (int, optional): The period of the fast EMA. Defaults to 2.
        slow (int, optional): The period of the slow EMA. Defaults to 30.

    Returns:
        pd.Series: A Pandas Series containing the KAMA values.
    """
    close = price_data["Close"]

    # Calculate the absolute price change
    price_change = np.abs(close - close.shift())

    # Calculate the volatility
    volatility = price_change.rolling(window=period).sum()

    # Calculate the Efficiency Ratio (ER)
    er = price_change / volatility

    # Calculate the Smoothing Constant (SC)
    fast_weight = 2 / (fast + 1)
    slow_weight = 2 / (slow + 1)
    sc = np.square(er * (fast_weight - slow_weight) + slow_weight)

    # Calculate KAMA
    kama = pd.Series(index=close.index, dtype="float64")
    kama.iloc[period - 1] = close.iloc[period - 1]

    for i in range(period, len(close)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            close.iloc[i] - kama.iloc[i - 1]
        )

    return kama


def ALMA(
    price_data: pd.DataFrame, period: int = 9, sigma: int = 6, offset: float = 0.85
) -> pd.Series:
    """
    Calculate the Arnaud Legoux Moving Average (ALMA) for the given price data.

    Parameters:
    price_data (DataFrame): A DataFrame containing OHLCV price data with columns 'Open', 'High', 'Low', 'Close', and 'Volume'.
    window (int, optional): The window size to use for the moving average calculation. Defaults to 9.
    sigma (int, optional): The sigma value for Gaussian distribution. Defaults to 6.
    offset (float, optional): The offset value for the Gaussian distribution center. Defaults to 0.85.

    Returns:
    Series: A pandas Series containing the ALMA values.
    """

    # Get the Close prices from the price data
    close = price_data["Close"]

    # Calculate the center of the Gaussian distribution
    m = int(offset * (period - 1))

    # Calculate the standard deviation for the Gaussian distribution
    s = period / sigma

    # Initialize an array of zeros with the size of the window for the ALMA weights
    alma_weights = np.zeros(period)

    # Calculate the weights for the Gaussian distribution function
    for i in range(period):
        xi = i - m
        alma_weights[i] = np.exp(-(xi**2) / (2 * s**2))

    # Normalize the ALMA weights
    alma_weights /= np.sum(alma_weights)

    # Calculate the ALMA values using a centered rolling window and apply the weights
    return close.rolling(window=period, center=False).apply(
        lambda x: np.sum(alma_weights * x), raw=True
    )


def T3(price_data: pd.DataFrame, period: int = 5, vfactor: float = 0.7) -> pd.Series:
    """
    Calculate the T3 moving average indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    period : int, optional
        The number of periods used for the calculation, by default 5.
    vfactor : float, optional
        The volume factor used for smoothing, by default 0.7.

    Returns
    -------
    pd.Series
        A Series containing the T3 values.
    """
    # Calculate the T3 components
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]
    typical_price = (high + low + close) / 3
    e1 = series_ema(typical_price, period)
    e2 = series_ema(e1, period)
    e3 = series_ema(e2, period)
    e4 = series_ema(e3, period)
    e5 = series_ema(e4, period)
    e6 = series_ema(e5, period)

    # Calculate the T3
    c1 = -vfactor * vfactor * vfactor
    c2 = 3 * vfactor * vfactor + 3 * vfactor * vfactor * vfactor
    c3 = -6 * vfactor * vfactor - 3 * vfactor - 3 * vfactor * vfactor * vfactor
    c4 = 1 + 3 * vfactor + vfactor * vfactor * vfactor + 3 * vfactor * vfactor
    T3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    return T3


def FantailVMA(
    price_data: pd.DataFrame,
    adx_length: int = 2,
    weighting: float = 2.0,
    ma_length: int = 1,
) -> pd.DataFrame:
    """
    Calculate the Fantail Volume-Weighted Moving Average (VMA) indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    adx_length : int, optional
        The number of periods used for the ADX calculation, by default 2.
    weighting : float, optional
        The weighting factor, by default 2.0.
    ma_length : int, optional
        The number of periods used for the moving average calculation, by default 1.

    Returns
    -------
    DataFrame
        A DataFrame containing the Fantail VMA values.
    """
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]

    n = len(close)

    # Initialize arrays
    spdi = np.zeros(n)
    smdi = np.zeros(n)
    str_ = np.zeros(n)
    adx = np.zeros(n)
    varma = np.zeros(n)
    ma = np.zeros(n)

    # Calculate the Fantail VMA
    for i in range(n - 2, -1, -1):
        hi = high[i]
        hi1 = high[i + 1]
        lo = low[i]
        lo1 = low[i + 1]
        close1 = close[i + 1]

        bulls = 0.5 * (abs(hi - hi1) + (hi - hi1))
        bears = 0.5 * (abs(lo1 - lo) + (lo1 - lo))

        if bulls > bears:
            bears = 0
        elif bulls < bears:
            bulls = 0
        else:
            bulls = 0
            bears = 0

        spdi[i] = (weighting * spdi[i + 1] + bulls) / (weighting + 1)
        smdi[i] = (weighting * smdi[i + 1] + bears) / (weighting + 1)

        tr = max(hi - lo, hi - close1)
        str_[i] = (weighting * str_[i + 1] + tr) / (weighting + 1)

        if str_[i] > 0:
            pdi = spdi[i] / str_[i]
            mdi = smdi[i] / str_[i]
        else:
            pdi = mdi = 0

        if (pdi + mdi) > 0:
            dx = abs(pdi - mdi) / (pdi + mdi)
        else:
            dx = 0

        adx[i] = (weighting * adx[i + 1] + dx) / (weighting + 1)
        vadx = adx[i]

        adxmin = min(adx[i : i + adx_length])
        adxmax = max(adx[i : i + adx_length])

        diff = adxmax - adxmin
        const = (vadx - adxmin) / diff if diff > 0 else 0

        varma[i] = ((2 - const) * varma[i + 1] + const * close[i]) / 2

    # Calculate the MA
    ma = pd.Series(varma).rolling(window=ma_length).mean()

    # Convert arrays to pandas Series
    varma_series = pd.Series(varma)
    ma_series = pd.Series(ma)

    return pd.DataFrame({"Fantail_VMA": varma_series, "Fantail_MA": ma_series})


def EHLERS(price_data: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Calculate the Ehler's 2 Pole Super Smoother Filter.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    period : int, optional
        The number of periods used for the filter calculation, by default 10.

    Returns
    -------
    pd.Series
        A Series containing the Ehler's 2 Pole Super Smoother Filter values.
    """
    series = price_data["Close"]

    n = len(series)

    # Calculate the Ehler's 2 Pole Super Smoother Filter
    a = np.exp(-1.414 * np.pi / period)
    b = 2 * a * np.cos(1.414 * np.pi / period)
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3

    output = np.zeros(n)
    for i in range(2, n):
        output[i] = (
            c1 * (series[i] + series[i - 1]) / 2
            + c2 * output[i - 1]
            + c3 * output[i - 2]
        )

    return pd.Series(output)


def McGinleyDI(
    price_data: pd.DataFrame, period: int = 12, mcg_constant: float = 5
) -> pd.Series:
    """
    Calculate the McGinley Dynamic Indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    period : int, optional
        The number of periods used for the indicator calculation, by default 12.
    mcg_constant : float, optional
        The McGinley constant, by default 5.

    Returns
    -------
    pd.Series
        A Series containing the McGinley Dynamic Indicator values.
    """
    series = price_data["Close"]

    n = len(series)
    mcg = np.zeros(n)

    # Calculate the Simple Moving Average for the given period
    ma = series.rolling(window=period).mean()

    for i in range(period, n):
        price = series[i]
        mcg[i] = ma[i - 1] + (price - ma[i - 1]) / (
            mcg_constant * period * np.power(price / ma[i - 1], 4)
        )

    return mcg


def DEMA(price_data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Double Exponential Moving Average (DEMA) of the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    period : int, optional, default: 14
        The number of periods to use in the calculation of the DEMA.

    Returns
    -------
    pd.Series
        A Series containing the DEMA values for the specified price data and period.
    """
    # Calculate the Exponential Moving Average (EMA) for the given period
    ema1 = price_data["Close"].ewm(span=period).mean()

    # Calculate the EMA of the first EMA
    ema2 = ema1.ewm(span=period).mean()

    # Calculate the DEMA using the three EMAs
    dema = 2 * ema1 - ema2

    # Return the DEMA
    return dema


def TEMA(price_data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Triple Exponential Moving Average (TEMA) of the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    period : int, optional, default: 14
        The number of periods to use in the calculation of the TEMA.

    Returns
    -------
    pd.Series
        A Series containing the TEMA values for the specified price data and period.
    """
    # Calculate the Exponential Moving Average (EMA) for the given period
    ema1 = price_data["Close"].ewm(span=period).mean()

    # Calculate the EMA of the first EMA
    ema2 = ema1.ewm(span=period).mean()

    # Calculate the EMA of the second EMA
    ema3 = ema2.ewm(span=period).mean()

    # Calculate the TEMA using the three EMAs
    tema = 3 * ema1 - 3 * ema2 + ema3

    # Return the TEMA
    return tema


def KijunSen(df: pd.DataFrame, period: int = 26, shift: int = 9) -> pd.Series:
    """
    Calculate Ichimoku Kinko Hyo Kijun-Sen line

    Parameters:
    df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns
    kijun (int): Kijun period
    kijun_shift (int): Kijun shift

    Returns:
    pd.Series: Series with Kijun-Sen line values
    """
    high = df["High"]
    low = df["Low"]
    kijun_buffer = [np.nan] * len(df)

    # Calculate Kijun-Sen for main part
    for i in range(period, len(df)):
        kijun_buffer[i - shift] = (
            high.iloc[i - period : i].max() + low.iloc[i - period : i].min()
        ) / 2

    # Calculate Kijun-Sen for initial part
    for i in range(shift - 1, -1, -1):
        kijun_buffer[i] = (
            high.iloc[: period - shift + i].max() + low.iloc[: period - shift + i].min()
        ) / 2

    return pd.Series(kijun_buffer, index=df.index)


# Confirmation Functions


def KASE(
    price_data: pd.DataFrame,
    pstLength: int = 9,
    pstX: int = 5,
    pstSmooth: int = 3,
    smoothPeriod: int = 10,
) -> pd.DataFrame:
    """Calculate the Kase Permission Stochastic Smoothed (KPSS) using a Pandas DataFrame.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame containing OHLCV data.
        pstLength (int, optional): The period for the Permission Stochastic calculation. Defaults to 9.
        pstX (int, optional): . Defaults to 5.
        pstSmooth (int, optional): . Defaults to 3.
        smoothPeriod (int, optional): . Defaults to 10.

    Returns:
        _type_: _description_
    """
    lookBackPeriod = pstLength * pstX
    alpha = 2.0 / (1.0 + pstSmooth)

    TripleK, TripleDF, TripleDS, TripleDSs, TripleDFs = 0.0, 0.0, 0.0, 0.0, 0.0

    fmin = price_data["Low"].rolling(window=lookBackPeriod).min()
    fmax = price_data["High"].rolling(window=lookBackPeriod).max() - fmin

    TripleK = 100.0 * (price_data["Close"] - fmin) / fmax
    TripleK = TripleK.replace([np.inf, -np.inf], 0).fillna(0)

    TripleDF = TripleK.ewm(alpha=alpha, adjust=False).mean().shift(pstX)
    TripleDS = (TripleDF.shift(pstX) * 2.0 + TripleDF) / 3.0

    def hma(series, period):
        half_length = period // 2
        sqrt_length = int(np.sqrt(period))

        wma_half = series_wma(series, half_length)
        wma_full = series_wma(series, period)

        hma_series = 2 * wma_half - wma_full

        return series_wma(hma_series, sqrt_length)

    TripleDSs = TripleDS.rolling(window=3).mean()
    pssBuffer = hma(TripleDSs, smoothPeriod)

    TripleDFs = TripleDF.rolling(window=3).mean()
    pstBuffer = hma(TripleDFs, smoothPeriod)

    # pst > pss == buy
    # pst < pss == sell

    # Create a Pandas DataFrame to store the KPSS lines
    result = pd.DataFrame(
        {
            "KPSS_BUY": pstBuffer,
            "KPSS_SELL": pssBuffer,
        }
    )

    return result


def MACDZeroLag(
    price_data: pd.DataFrame,
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate the Zero-Lag MACD and signal line.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing OHLCV price data.
    short_period : int, optional
        The number of periods for the short-term EMA, by default 12.
    long_period : int, optional
        The number of periods for the long-term EMA, by default 26.
    signal_period : int, optional
        The number of periods for the signal line EMA, by default 9.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the MACD and signal line values.
    """
    close = price_data["Close"]

    short_zlema = zero_lag_ema(close, short_period)
    long_zlema = zero_lag_ema(close, long_period)

    macd_line = short_zlema - long_zlema
    signal_line = zero_lag_ema(macd_line, signal_period)

    result = pd.DataFrame(
        {
            "MACD": macd_line,
            "SIGNAL": signal_line,
        }
    )

    return result


def KalmanFilter(
    price_data: pd.DataFrame, k: float = 1, sharpness: float = 1
) -> pd.DataFrame:
    """
    Calculate the Kalman Filter forex indicator based on the provided price data using the 'Close' prices.

    Parameters:
    price_data (pd.DataFrame): A Pandas DataFrame of OHLCV price data.
    k (float): K parameter for the Kalman Filter, default is 1.
    sharpness (float): Sharpness parameter for the Kalman Filter, default is 1.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated Kalman Filter values.
    """

    ext_map_buffer_up = np.empty(len(price_data))
    ext_map_buffer_up[:] = np.nan
    ext_map_buffer_down = np.empty(len(price_data))
    ext_map_buffer_down[:] = np.nan

    velocity = 0
    distance = 0
    error = 0
    value = price_data["Close"].iloc[1]

    for i in range(len(price_data) - 1, -1, -1):
        price = price_data["Close"].iloc[i]
        distance = price - value
        error = value + distance * np.sqrt(sharpness * k / 100)
        velocity = velocity + distance * k / 100
        value = error + velocity

        if velocity > 0:
            ext_map_buffer_up[i] = value
            ext_map_buffer_down[i] = np.nan

            if i < len(price_data) - 1 and np.isnan(ext_map_buffer_up[i + 1]):
                ext_map_buffer_up[i + 1] = ext_map_buffer_down[i + 1]
        else:
            ext_map_buffer_up[i] = np.nan
            ext_map_buffer_down[i] = value

            if i < len(price_data) - 1 and np.isnan(ext_map_buffer_down[i + 1]):
                ext_map_buffer_down[i + 1] = ext_map_buffer_up[i + 1]

    result = pd.DataFrame({"Up": ext_map_buffer_up, "Down": ext_map_buffer_down})
    return result


def Fisher(
    price_data: pd.DataFrame,
    range_periods: int = 10,
    price_smoothing: float = 0.3,
    index_smoothing: float = 0.3,
) -> pd.DataFrame:
    """
    Calculate the Fisher Indicator for the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    range_periods : int, optional, default: 10
        The number of periods to use in the calculation of the Fisher Indicator.
    price_smoothing : float, optional, default: 0.3
        The price smoothing factor.
    index_smoothing : float, optional, default: 0.3
        The index smoothing factor.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Fisher Indicator values for the specified price data and parameters.
    """
    # Calculate mid-price
    mid_price = (price_data["High"] + price_data["Low"]) / 2

    # Calculate the highest high and lowest low for the given range_periods
    highest_high = price_data["High"].rolling(window=range_periods).max()
    lowest_low = price_data["Low"].rolling(window=range_periods).min()

    # Calculate the greatest range and avoid division by zero
    greatest_range = (highest_high - lowest_low).replace(0, 0.1 * 10**-5)

    # Calculate the price location in the current range
    price_location = 2 * ((mid_price - lowest_low) / greatest_range) - 1

    # Apply price smoothing
    smoothed_location = price_location.ewm(alpha=(1 - price_smoothing)).mean()

    # Limit smoothed_location between -0.99 and 0.99 to avoid infinite values in the logarithm
    smoothed_location = smoothed_location.clip(lower=-0.99, upper=0.99)

    # Calculate the Fisher Index
    fisher_index = np.log((1 + smoothed_location) / (1 - smoothed_location))

    # Apply index smoothing
    smoothed_fisher = fisher_index.ewm(alpha=(1 - index_smoothing)).mean()

    # Separate uptrend and downtrend values
    uptrend = smoothed_fisher.where(smoothed_fisher > 0, 0)
    downtrend = smoothed_fisher.where(smoothed_fisher <= 0, 0)

    # Return the Fisher Indicator values as a DataFrame
    return pd.DataFrame({"Fisher": smoothed_fisher})


def BullsBearsImpulse(price_data: pd.DataFrame, ma_period: int = 13) -> pd.DataFrame:
    """
    Calculate the Bulls Bears Impulse indicator for the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    ma_period : int, optional, default: 13
        The number of periods to use in the calculation of the moving average.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Bulls Bears Impulse indicator values for the specified price data and parameters.
    """
    # Calculate moving average of closing prices
    ma = price_data["Close"].rolling(window=ma_period).mean()

    # Calculate the Bulls and Bears difference from the moving average
    bulls = price_data["High"] - ma
    bears = price_data["Low"] - ma

    # Calculate the impulse (bulls minus bears)
    impulse = bulls - bears

    # Create buffers: 1 for bullish, -1 for bearish
    buffer = impulse.apply(lambda x: 1.0 if x > 0 else -1.0)

    return pd.DataFrame({"Bulls": buffer, "Bears": -buffer})


def Gen3MA(
    price_data: pd.DataFrame, period: int = 220, sampling_period: int = 50
) -> pd.DataFrame:
    """
    Calculate the 3rd Generation Moving Average indicator for the given price data.

    Parameters
    ----------
    price_data : pd.DataFrame
        A Pandas DataFrame containing OHLCV price data.
    ma_period : int, optional, default: 220
        The number of periods to use in the calculation of the 3GMA.
    ma_sampling_period : int, optional, default: 50
        The number of periods to use in the calculation of the crossing MA.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the 3rd Generation Moving Average indicator values for the specified price data and parameters.
    """
    ma1 = price_data["Close"].rolling(window=period).mean()
    ma2 = ma1.rolling(window=sampling_period).mean()

    alpha = sampling_period / period

    # 3rd Generation Moving Average line
    ma3g = (1 + alpha) * ma1 - alpha * ma2

    return pd.DataFrame({"MA3G": ma3g, "SignalMA": ma2})


def Aroon(price_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the Aroon indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period (int): The number of periods to use in the calculation. Default is 14.

    Returns:
    pd.DataFrame: A Pandas DataFrame with two columns: 'Aroon Up' and 'Aroon Down'.
    """
    aroon_up = [np.nan] * period
    aroon_down = [np.nan] * period

    for i in range(period, len(price_data)):
        high_period = price_data["High"].iloc[i - period : i].tolist()
        low_period = price_data["Low"].iloc[i - period : i].tolist()
        n_high = high_period.index(max(high_period))
        n_low = low_period.index(min(low_period))

        aroon_up.append(100.0 * (period - n_high) / period)
        aroon_down.append(100.0 * (period - n_low) / period)

    return pd.DataFrame({"Aroon Up": aroon_up, "Aroon Down": aroon_down})


def Coral(price_data: pd.DataFrame, period: int = 34) -> pd.DataFrame:
    """
    Calculates the THV Coral indicator for the given price data and period.
    Returns a Pandas DataFrame containing the calculated indicator values.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period (int): The period used in the indicator calculation. Default is 34.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the THV Coral indicator values.
    """
    gd_88 = 0.4
    g_ibuf_96 = np.empty(len(price_data))
    g_ibuf_100 = np.empty(len(price_data))
    g_ibuf_104 = np.empty(len(price_data))
    g_ibuf_108 = np.empty(len(price_data))
    gda_112 = np.empty(len(price_data))
    gda_116 = np.empty(len(price_data))
    gda_120 = np.empty(len(price_data))
    gda_124 = np.empty(len(price_data))
    gda_128 = np.empty(len(price_data))
    gda_132 = np.empty(len(price_data))
    gd_136 = -(gd_88**3)
    gd_144 = 3.0 * (gd_88**2 + gd_88**3)
    gd_152 = -3.0 * (2.0 * gd_88**2 + gd_88 + gd_88**3)
    gd_160 = 3.0 * gd_88 + 1.0 + gd_88**3 + 3.0 * gd_88**2
    gi_84 = period
    if gi_84 < 1:
        gi_84 = 1
    gi_84 = (gi_84 - 1) // 2 + 1
    gd_176 = 2 / (gi_84 + 1)
    gd_184 = 1 - gd_176

    for i in range(len(price_data)):
        if i == 0:
            gda_112[i] = price_data["Close"][i]
            gda_116[i] = gda_112[i]
            gda_120[i] = gda_116[i]
            gda_124[i] = gda_120[i]
            gda_128[i] = gda_124[i]
            gda_132[i] = gda_128[i]
        else:
            gda_112[i] = gd_176 * price_data["Close"][i] + gd_184 * gda_112[i - 1]
            gda_116[i] = gd_176 * gda_112[i] + gd_184 * gda_116[i - 1]
            gda_120[i] = gd_176 * gda_116[i] + gd_184 * gda_120[i - 1]
            gda_124[i] = gd_176 * gda_120[i] + gd_184 * gda_124[i - 1]
            gda_128[i] = gd_176 * gda_124[i] + gd_184 * gda_128[i - 1]
            gda_132[i] = gd_176 * gda_128[i] + gd_184 * gda_132[i - 1]

        g_ibuf_108[i] = (
            gd_136 * gda_132[i]
            + gd_144 * gda_128[i]
            + gd_152 * gda_124[i]
            + gd_160 * gda_120[i]
        )
        ld_0 = g_ibuf_108[i]
        if i == 0:
            ld_8 = g_ibuf_108[i + 1]
        else:
            ld_8 = g_ibuf_108[i - 1]
        g_ibuf_96[i] = ld_0
        g_ibuf_100[i] = ld_0
        g_ibuf_104[i] = ld_0
        if ld_8 > ld_0:
            g_ibuf_100[i] = np.nan
        elif ld_8 < ld_0:
            g_ibuf_104[i] = np.nan
        else:
            g_ibuf_96[i] = np.nan

    return pd.DataFrame({"THV Coral": g_ibuf_108, "Up": g_ibuf_100, "Down": g_ibuf_104})


def CenterOfGravity(price_data, period=10) -> pd.Series:
    """
    Calculates Ehler's Center of Gravity oscillator.

    Args:
        price_data: Pandas DataFrame with 'High' and 'Low'
        period: int, number of periods to calculate

    Returns:
        pd.Series with same length as price_data
    """

    def p(index: int) -> float:
        return (price_data["High"].iloc[index] + price_data["Low"].iloc[index]) / 2.0

    cg = [float("nan")] * (period - 1)  # prepend NaNs

    for s in range(period - 1, len(price_data)):
        num = 0.0
        denom = 0.0
        for count in range(period):
            idx = s - period + 1 + count
            p_val = p(idx)
            num += (1.0 + count) * p_val
            denom += p_val

        if denom != 0:
            cg_val = -num / denom + (period + 1.0) / 2.0
        else:
            cg_val = 0.0

        cg.append(cg_val)

    return pd.Series(cg, index=price_data.index)


def GruchaIndex(
    price_data: pd.DataFrame, period: int = 10, ma_period: int = 10
) -> pd.DataFrame:
    """
    Calculates the Grucha Index and its moving average.

    Parameters:
    price_data (pd.DataFrame): DataFrame of OHLCV data.
    Okresy (int): Number of periods to calculate the Grucha Index.
    MA_Okresy (int): Number of periods to calculate the moving average of the Grucha Index.

    Returns:
    pd.DataFrame: DataFrame of the Grucha Index and its moving average.
    """
    ExtMapBuffer1 = [0.0] * len(price_data)
    tab = [0.0] * len(price_data)
    srednia = [0.0] * len(price_data)

    for i in range(len(price_data)):
        close = price_data["Close"][i]
        open = price_data["Open"][i]

        dResult = open - close
        tab[i] = dResult

        if i >= period - 1:
            gora = 0
            dol = 0
            for j in range(i, i - period, -1):
                if tab[j] < 0:
                    gora += tab[j]
                elif tab[j] >= 0:
                    dol -= tab[j]

            if dol <= 0:
                dol = dol * (-1)
            if gora <= 0:
                gora = gora * (-1)
            suma = dol + gora

            if suma == 0:
                wynik = 0
            elif suma > 0:
                wynik = (gora / suma) * 100
            else:
                wynik = 0

            ExtMapBuffer1[i] = wynik

    for i in range(len(price_data) - ma_period + 1):
        srednia[i + ma_period - 1] = sum(ExtMapBuffer1[i : i + ma_period]) / ma_period

    return pd.DataFrame({"Grucha Index": ExtMapBuffer1, "MA of Grucha Index": srednia})


def HalfTrend(price_data: pd.DataFrame, amplitude: int = 2) -> pd.DataFrame:
    def single_atr(index: int) -> float:
        return (price_data["High"][index] - price_data["Low"][index]) / 2

    nexttrend = False
    maxlowprice = 0
    minhighprice = float("inf")
    up = [0] * len(price_data)
    down = [0] * len(price_data)
    atrlo = [0] * len(price_data)
    atrhi = [0] * len(price_data)
    trend = [0] * len(price_data)

    for i in range(len(price_data) - 1, -1, -1):
        lowprice_i = price_data["Low"][i - amplitude : i].min()
        highprice_i = price_data["High"][i - amplitude : i].max()
        lowma = price_data["Low"][i - amplitude : i].mean()
        highma = price_data["High"][i - amplitude : i].mean()
        trend[i] = trend[i + 1] if i + 1 < len(price_data) else trend[i]
        atr_val = single_atr(i)

        if i + 1 < len(price_data):
            if nexttrend:
                maxlowprice = max(lowprice_i, maxlowprice)

                if (
                    highma < maxlowprice
                    and price_data["Close"][i] < price_data["Low"][i + 1]
                ):
                    trend[i] = 1.0
                    nexttrend = False
                    minhighprice = highprice_i
            else:
                minhighprice = min(highprice_i, minhighprice)

                if (
                    lowma > minhighprice
                    and price_data["Close"][i] > price_data["High"][i + 1]
                ):
                    trend[i] = 0.0
                    nexttrend = True
                    maxlowprice = lowprice_i

        if trend[i] == 0.0:
            if i + 1 < len(price_data) and trend[i + 1] != 0.0:
                up[i] = down[i + 1]
                up[i + 1] = up[i]
            else:
                up[i] = (
                    max(maxlowprice, up[i + 1]) if i + 1 < len(price_data) else up[i]
                )

            atrhi[i] = up[i] - atr_val
            atrlo[i] = up[i]
            down[i] = 0.0
        else:
            if i + 1 < len(price_data) and trend[i + 1] != 1.0:
                down[i] = up[i + 1]
                down[i + 1] = down[i]
            else:
                down[i] = (
                    min(minhighprice, down[i + 1])
                    if i + 1 < len(price_data)
                    else down[i]
                )

            atrhi[i] = down[i] + atr_val
            atrlo[i] = down[i]
            up[i] = 0.0

    return pd.DataFrame(
        {"Up": up, "Down": down}
    )  # , 'AtrLo': atrlo, 'AtrHi': atrhi, 'Trend': trend})


def J_TPO(price_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the J_TPO_Velocity indicator.

    J_TPO is an oscillator between -1 and +1, a nonparametric statistic quantifying how well the prices are ordered
    in consecutive ups (+1) or downs (-1) or intermediate cases in between. J_TPO_Velocity takes that value and
    multiplies it by the range, highest high to lowest low in the period (in pips), divided by the period length.
    Therefore, J_TPO_Velocity is a rough estimate of "velocity" as in "pips per bar". Positive of course means going
    up and negative means going down. J_TPO_Velocity thus crosses zero at exactly the same time as J_TPO, but the
    absolute magnitude is different.

    Args:
    price_data: Pandas DataFrame of OHLCV data.
    period: Length of the indicator. Default is 14.

    Returns:
    Pandas Series of J_TPO_Velocity values.
    """
    close_prices = price_data["Close"]
    high_prices = price_data["High"]
    low_prices = price_data["Low"]
    # Default assumes 4-digit forex pairs; now, it's 5-digit for all non-JPY pairs
    # point = 0.0001 # Assuming a 4-digit forex pair
    # Count the number of digits after the decimal point
    point = 10 ** (
        -len(str(close_prices[0]).split(".")[1])
    )  # To normalize the indicator's value

    def j_tpo_range(high_prices, low_prices, period, shift):
        highest_high = high_prices[shift : shift + period].max()
        lowest_low = low_prices[shift : shift + period].min()

        return highest_high - lowest_low  # /point

    ext_map_buffer = [np.nan] * len(close_prices)

    if period < 3:
        # print("J_TPO_B: length must be at least 3")
        return ext_map_buffer

    for i in range(len(close_prices) - period):
        ext_map_buffer[i] = (
            j_tpo_value(close_prices, period, i)
            * j_tpo_range(high_prices, low_prices, period, i)
            / period
        )

    j_tpo = pd.Series(ext_map_buffer, index=close_prices.index)

    return j_tpo


def KVO(
    price_data: pd.DataFrame,
    fast_ema: int = 34,
    slow_ema: int = 55,
    signal_ema: int = 13,
) -> pd.DataFrame:
    """
    Calculates the Klinger Volume Oscillator (KVO) indicator.

    Parameters:
    price_data (pd.DataFrame): DataFrame of OHLCV data.
    FastEMA (int): The number of periods for the fast EMA. Default is 34.
    SlowEMA (int): The number of periods for the slow EMA. Default is 55.
    SignalEMA (int): The number of periods for the signal EMA. Default is 13.

    Returns:
    pd.DataFrame: A DataFrame of the KVO indicator values.
    """
    tpc = (price_data["High"] + price_data["Low"] + price_data["Close"]) / 3
    tpp = tpc.shift(-1)
    v = np.where(
        tpc > tpp, price_data["Volume"], np.where(tpc < tpp, -price_data["Volume"], 0)
    )

    MainBuffer = (
        pd.Series(v).ewm(span=fast_ema).mean() - pd.Series(v).ewm(span=slow_ema).mean()
    )
    SignalBuffer = MainBuffer.ewm(span=signal_ema).mean()

    kvo = pd.DataFrame({"KVO": MainBuffer, "KVO_Signal": SignalBuffer})

    return kvo


def LWPI(price_data: pd.DataFrame, period: int = 8) -> pd.DataFrame:
    """
    Larry Williams Proxy Index indicator

    Args:
    price_data: Pandas DataFrame of OHLCV data
    length: Length of the indicator period

    Returns:
    Pandas DataFrame of Larry Williams Proxy Index values
    """
    Raw = price_data["Open"] - price_data["Close"]
    ma = Raw.rolling(window=period).mean()
    atr = (price_data["High"] - price_data["Low"]).rolling(window=period).mean()
    lwpi = 50 * ma / atr + 50
    lwpi[atr == 0] = 0
    return pd.Series(lwpi)


def SuperTrend(
    price_data: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculate the Supertrend indicator from OHLCV price data.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    nbr_periods (int, optional): Number of periods for ATR calculation. Default is 10.
    multiplier (float, optional): Multiplier for ATR. Default is 3.0.

    Returns:
    pd.DataFrame: DataFrame containing the Supertrend indicator values.
    """
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]

    median_price = (high + low) / 2
    atr_values = ATR(price_data, period)
    up = median_price + multiplier * atr_values
    down = median_price - multiplier * atr_values
    trend = pd.Series(1, index=price_data.index)

    change_of_trend = 0
    for i in range(1, len(price_data)):
        if close[i] > up[i - 1]:
            trend[i] = 1
            if trend[i - 1] == -1:
                change_of_trend = 1
        elif close[i] < down[i - 1]:
            trend[i] = -1
            if trend[i - 1] == 1:
                change_of_trend = 1
        else:
            trend[i] = trend[i - 1]
            change_of_trend = 0

        flag = 1 if trend[i] < 0 and trend[i - 1] > 0 else 0
        flagh = 1 if trend[i] > 0 and trend[i - 1] < 0 else 0

        if trend[i] > 0 and down[i] < down[i - 1]:
            down[i] = down[i - 1]
        if trend[i] < 0 and up[i] > up[i - 1]:
            up[i] = up[i - 1]

        if flag == 1:
            up[i] = median_price[i] + multiplier * atr_values[i]
        if flagh == 1:
            down[i] = median_price[i] - multiplier * atr_values[i]

        if trend[i] == 1:
            current_supertrend = down[i]
        elif change_of_trend == 1:
            if i + 1 < len(price_data):
                current_supertrend = up[i + 1]
                change_of_trend = 0
            else:
                current_supertrend = up[i]  # fallback
                change_of_trend = 0
        elif trend[i] == -1:
            current_supertrend = up[i]
        elif change_of_trend == 1:
            current_supertrend = down[i + 1]
            change_of_trend = 0

    supertrend = pd.DataFrame({"Supertrend": current_supertrend, "Trend": trend})

    return supertrend


def TTF(
    price_data: pd.DataFrame,
    period: int = 8,
    top_line: int = 75,
    bottom_line: int = -75,
    t3_period: int = 3,
    b: float = 0.7,
) -> pd.DataFrame:
    """
    Trend Trigger Factor (TTF) indicator.

    Args:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period (int): Number of bars for computation.
    top_line (int): Top line value.
    bottom_lin (int): Bottom line value.
    t3_period (int): Period of T3.
    b (float): Value of b.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated indicator's values.
    """
    highest_high_recent = price_data["High"].rolling(window=period).max()
    highest_high_older = highest_high_recent.shift(period)
    lowest_low_recent = price_data["Low"].rolling(window=period).min()
    lowest_low_older = lowest_low_recent.shift(period)

    buy_power = highest_high_recent - lowest_low_older
    sell_power = highest_high_older - lowest_low_recent
    ttf = (buy_power - sell_power) / (0.5 * (buy_power + sell_power)) * 100

    c1 = -(b**3)
    c2 = 3 * b**2 + 3 * b**3
    c3 = -6 * b**2 - 3 * b - 3 * b**3
    c4 = 1 + 3 * b + b**2 + 3 * b**3
    t3 = (
        c1 * ttf
        + c2 * ttf.shift(t3_period)
        + c3 * ttf.shift(2 * t3_period)
        + c4 * ttf.shift(3 * t3_period)
    ).rename("T3")

    e1 = t3
    e2 = e1.ewm(span=2, adjust=False).mean()
    e3 = e2.ewm(span=2, adjust=False).mean()
    e4 = e3.ewm(span=2, adjust=False).mean()
    e5 = e4.ewm(span=2, adjust=False).mean()
    e6 = e5.ewm(span=2, adjust=False).mean()
    ttf = (c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3).rename("TTF")

    signal = np.where(ttf >= 0, top_line, bottom_line)

    return pd.DataFrame({"TTF": ttf, "Signal": signal}, index=price_data.index)


def Vortex(price_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the Vortex Indicator (VI) and Vortex Movement (VM) for a given length.

    Parameters:
    price_data (pd.DataFrame): OHLCV data
    VI_Length (int): length of VI calculation (default=14)

    Returns:
    pd.DataFrame: VI+ and VI- values
    """
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]
    tr = pd.Series(data=0.0, index=price_data.index)
    plus_vm = abs(high - low.shift())
    minus_vm = abs(low - high.shift())

    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    sum_plus_vm = plus_vm.rolling(window=period).sum()
    sum_minus_vm = minus_vm.rolling(window=period).sum()
    sum_tr = tr.rolling(window=period).sum()

    plus_vi = sum_plus_vm / sum_tr
    minus_vi = sum_minus_vm / sum_tr

    return pd.DataFrame({"PlusVI": plus_vi, "MinusVI": minus_vi})


def BraidFilterHist(
    price_data: pd.DataFrame,
    ma1_period: int = 3,
    ma2_period: int = 7,
    ma3_period: int = 14,
    atr_period: int = 14,
    pips_min_sep_percent: float = 40,
) -> pd.DataFrame:
    """
    Braid Filter indicator of Robert Hill stocks and commodities magazine 2006.

    Args:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    ma_period1 (int): Period of first moving average.
    ma_period2 (int): Period of second moving average.
    ma_period3 (int): Period of third moving average.
    atr_period (int): Period of ATR.
    pips_min_sep_percent (float): Minimum separation percent; separates MAs by minimum this % of ATR.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated indicator's values.
    """
    atr = ATR(price_data, atr_period)
    ma1 = price_data["Close"].rolling(window=ma1_period).mean()
    ma2 = price_data["Open"].rolling(window=ma2_period).mean()
    ma3 = price_data["Close"].rolling(window=ma3_period).mean()

    max_val = pd.concat([ma1, ma2, ma3], axis=1).max(axis=1)
    min_val = pd.concat([ma1, ma2, ma3], axis=1).min(axis=1)
    diff = max_val - min_val
    fil = atr * pips_min_sep_percent / 100

    ma1_gt_ma2 = ma1 > ma2
    ma2_gt_ma1 = ma2 > ma1
    diff_gt_fil = diff > fil

    trend = pd.Series(
        np.where(
            ma1_gt_ma2 & diff_gt_fil, 1, np.where(ma2_gt_ma1 & diff_gt_fil, -1, 0)
        ),
        index=price_data.index,
    )
    trend.fillna(method="ffill", inplace=True)

    UpH = pd.Series(np.where(trend == 1, diff, np.nan), index=price_data.index)
    DnH = pd.Series(np.where(trend == -1, diff, np.nan), index=price_data.index)

    return pd.DataFrame({"UpH": UpH, "DnH": DnH}, index=price_data.index)


def BraidFilter(
    price_data: pd.DataFrame,
    period1: int = 5,
    period2: int = 8,
    period3: int = 20,
    pips_min_sep_percent: float = 0.5,
) -> pd.DataFrame:
    """
    Braid Filter indicator of Robert Hill stocks and commodities magazine 2006.

    Args:
    price_data (pd.DataFrame): Pandas DataFrame of OHLCV price data.
    period1 (int): Period of first moving average.
    period2 (int): Period of second moving average.
    period3 (int): Period of third moving average.
    pips_min_sep_percent (int): Minimum separation percent; separates MAs by minimum this % of ATR.
        Specifies the minimum separation between the three moving averages as a fraction of the current average true range (ATR) value.
        The ATR is a measure of volatility that takes into account the range of price movement of an asset over a certain number of periods. By multiplying the ATR with pips_min_sep_percent/100, we get the minimum separation between the moving averages in pips.
        For example, if pips_min_sep_percent is set to 40, and the current ATR value is 100 pips, the minimum separation between the three moving averages would be 40% of 100 pips, or 40 pips.
        This parameter is used to filter out noise and prevent the three moving averages from getting too close to each other, which could result in false signals.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the calculated indicator's values.
    """
    ema1 = price_data["Close"].ewm(span=period1, adjust=False).mean()
    ema2 = price_data["Open"].ewm(span=period2, adjust=False).mean()
    ema3 = price_data["Close"].ewm(span=period3, adjust=False).mean()

    CrossUp, CrossDown = np.where((ema1 > ema2) & (ema2 > ema3), 1, 0), np.where(
        (ema1 < ema2) & (ema2 < ema3), -1, 0
    )
    ATR = price_data["High"] - price_data["Low"]
    Filter = ATR.rolling(window=14).mean().abs() * pips_min_sep_percent

    # Return a DataFrame
    return pd.DataFrame(
        {"CrossUp": CrossUp, "CrossDown": CrossDown, "Filter": Filter},
        index=price_data.index,
    )


def Laguerre(price_data: pd.DataFrame, gamma: float = 0.7) -> pd.Series:
    """Calculates the Laguerre indicator.

    Args:
        price_data (pd.DataFrame): A Pandas DataFrame of OHLCV price data.
        gamma (float, optional): The gamma. Defaults to 0.7.

    Returns:
        pd.Series: A Pandas Series containing the calculated indicator's values.
    """
    close = price_data["Close"].values

    laguerre = np.zeros(len(price_data))

    i = len(price_data) - 1
    l0, l1, l2, l3, lrsi = 0, 0, 0, 0, 0

    while i >= 0:
        l0a, l1a, l2a, l3a = l0, l1, l2, l3

        l0 = (1 - gamma) * close[i] + gamma * l0a
        l1 = -gamma * l0 + l0a + gamma * l1a
        l2 = -gamma * l1 + l1a + gamma * l2a
        l3 = -gamma * l2 + l2a + gamma * l3a

        cu, cd = 0, 0

        if l0 >= l1:
            cu = l0 - l1
        else:
            cd = l1 - l0

        if l1 >= l2:
            cu += l1 - l2
        else:
            cd += l2 - l1

        if l2 >= l3:
            cu += l2 - l3
        else:
            cd += l3 - l2

        if cu + cd != 0:
            lrsi = cu / (cu + cd)

        laguerre[i] = lrsi

        i -= 1

    df = pd.Series(laguerre, index=price_data.index)

    return df


def RecursiveMA(price_data: pd.DataFrame, period=2, recursions=20):
    data_length = len(price_data)
    open_prices = price_data["Open"].values

    xema_buffer = np.zeros(data_length)
    trigger_buffer = np.zeros(data_length)

    for i in range(data_length - 1, -1, -1):
        ema = np.full(recursions, open_prices[i])
        alpha = 2.0 / (period + 1.0)

        for j in range(recursions - 1):
            ema[1:] = alpha * ema[:-1] + (1 - alpha) * ema[1:]

        xema_buffer[i] = ema[-1]
        trigger_buffer[i] = (
            np.dot(ema, np.arange(recursions, 0, -1))
            * 2
            / (recursions * (recursions + 1))
        )

    recursive_ma = pd.DataFrame(
        {"Xema": xema_buffer, "Trigger": trigger_buffer}, index=price_data.index
    )
    return recursive_ma


def SchaffTrendCycle(
    price_data: pd.DataFrame,
    period: int = 10,
    fast_ma_period: int = 23,
    slow_ma_period: int = 50,
    signal_period: int = 3,
) -> pd.DataFrame:
    """
    Calculates the Schaff Trend Cycle (STC) indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    period (int, optional): Schaff period. Default is 10.
    fast_ma_period (int, optional): Fast MACD period. Default is 23.
    slow_ma_period (int, optional): Slow MACD period. Default is 50.
    signal_period (int, optional): Signal period. Default is 3.

    Returns:
    pd.DataFrame: Pandas DataFrame containing the STC values.
    """

    # Calculate MACD
    macd = EMA(price_data, fast_ma_period) - EMA(price_data, slow_ma_period)

    # Calculate fastK and fastD
    macd_low = macd.rolling(window=period).min()
    macd_high = macd.rolling(window=period).max()
    fast_k = 100 * (macd - macd_low) / (macd_high - macd_low)
    alpha = 2.0 / (1.0 + signal_period)
    fast_d = fast_k.ewm(alpha=alpha, adjust=False).mean()

    # Calculate STC values
    stoch_low = fast_d.rolling(window=period).min()
    stoch_high = fast_d.rolling(window=period).max()
    fast_kk = 100 * (fast_d - stoch_low) / (stoch_high - stoch_low)
    stc = fast_kk.ewm(alpha=alpha, adjust=False).mean()

    # Calculate STC difference between current and previous values
    diff = stc.diff().abs()

    # Create DataFrame with STC values
    stc_df = pd.DataFrame({"STC": stc, "Diff": diff}, index=price_data.index)

    return stc_df


def SmoothStep(
    price_data: pd.DataFrame,
    period: int = 32,
) -> pd.DataFrame:
    """
    Calculates the SmoothStep indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    period (int, optional): Period. Default is 32.

    Returns:
    pd.DataFrame: Pandas DataFrame containing the SmoothStep values.
    """
    price = price_data["Close"]

    # Calculate SmoothStep values
    min_price = price.rolling(window=period).min()
    max_price = price.rolling(window=period).max()
    raw_value = (price - min_price) / (max_price - min_price)
    smooth_step = raw_value**2 * (3 - 2 * raw_value)

    # Create DataFrame with SmoothStep values
    smooth_step_df = pd.DataFrame({"SmoothStep": smooth_step}, index=price_data.index)

    return smooth_step_df


def TopTrend(
    price_data: pd.DataFrame,
    period: int = 20,
    deviation: int = 2,
    money_risk: float = 1.00,
) -> pd.Series:
    """
    Calculates the TopTrend indicator.

    Parameters:
    price_data (pd.DataFrame): Pandas DataFrame containing OHLCV price data.
    period (int, optional): Bollinger Bands Period. Default is 20.
    deviation (int, optional): Deviation. Default is 2.
    money_risk (float, optional): Offset Factor. Default is 1.00.

    Returns:
    pd.Series: Pandas Series containing the TopTrend values.
    """
    close = price_data["Close"]
    data_length = len(close)

    # Calculate Bollinger Bands
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    smax = sma + deviation * std
    smin = sma - deviation * std

    # Initialize arrays
    trend = np.zeros(data_length)
    bsmax = np.zeros(data_length)
    bsmin = np.zeros(data_length)
    uptrend_buffer = np.full(data_length, np.nan)
    downtrend_buffer = np.full(data_length, np.nan)

    for shift in range(data_length - 1, -1, -1):
        if shift < len(close) - 1:
            if close.iloc[shift] > smax.iloc[shift + 1]:
                trend[shift] = 1
            if close.iloc[shift] < smin.iloc[shift + 1]:
                trend[shift] = -1

            if trend[shift] > 0 and smin.iloc[shift] < smin.iloc[shift + 1]:
                smin.iloc[shift] = smin.iloc[shift + 1]
            if trend[shift] < 0 and smax.iloc[shift] > smax.iloc[shift + 1]:
                smax.iloc[shift] = smax.iloc[shift + 1]

            bsmax[shift] = smax.iloc[shift] + 0.5 * (money_risk - 1) * (
                smax.iloc[shift] - smin.iloc[shift]
            )
            bsmin[shift] = smin.iloc[shift] - 0.5 * (money_risk - 1) * (
                smax.iloc[shift] - smin.iloc[shift]
            )

            if trend[shift] > 0 and bsmin[shift] < bsmin[shift + 1]:
                bsmin[shift] = bsmin[shift + 1]
            if trend[shift] < 0 and bsmax[shift] > bsmax[shift + 1]:
                bsmax[shift] = bsmax[shift + 1]

            if trend[shift] > 0:
                uptrend_buffer[shift] = bsmin[shift]
                downtrend_buffer[shift] = np.nan
            elif trend[shift] < 0:
                downtrend_buffer[shift] = bsmax[shift]
                uptrend_buffer[shift] = np.nan

    # For the final data point
    if close.iloc[-1] > smax.iloc[-1]:
        trend[-1] = 1
    elif close.iloc[-1] < smin.iloc[-1]:
        trend[-1] = -1
    else:
        trend[-1] = trend[-2]

    # Create the result Series
    result = pd.Series(trend, index=close.index)

    return result


def TrendLord(
    price_data: pd.DataFrame,
    period: int = 12,
    ma_method: str = "smma",
    applied_price: str = "close",
    show_high_low: bool = False,
    signal_bar: int = 1,
) -> pd.DataFrame:
    """
    Calculate the TrendLord indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        Pandas DataFrame containing OHLCV price data.
    period : int, default 12
        The number of periods for the moving average.
    ma_method : str, default 'smma'
        The moving average method: 'sma', 'ema', 'smma', or 'lwma'.
    applied_price : str, default 'close'
        The price to apply the moving average to: 'open', 'high', 'low', 'close', 'hl2', 'hlc3', or 'ohlc4'.
    show_high_low : bool, default False
        Whether to use the high/low values in the calculation.
    signal_bar : int, default 1
        The bar to signal the indicator value.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the calculated indicator values.
    """
    # Map the 'applied_price' string to the corresponding price values
    if applied_price == "open":
        price = price_data["Open"]
    elif applied_price == "high":
        price = price_data["High"]
    elif applied_price == "low":
        price = price_data["Low"]
    elif applied_price == "close":
        price = price_data["Close"]
    elif applied_price == "hl2":
        price = (price_data["High"] + price_data["Low"]) / 2
    elif applied_price == "hlc3":
        price = (price_data["High"] + price_data["Low"] + price_data["Close"]) / 3
    elif applied_price == "ohlc4":
        price = (
            price_data["Open"]
            + price_data["High"]
            + price_data["Low"]
            + price_data["Close"]
        ) / 4

    # Calculate the moving average based on the 'ma_method' parameter
    if ma_method == "sma":
        MA = price.rolling(window=period).mean()
        Array1 = MA.rolling(window=period).mean()
    elif ma_method == "ema":
        MA = price.ewm(span=period).mean()
        Array1 = MA.ewm(span=period).mean()
    elif ma_method == "smma":
        MA = price.ewm(alpha=1 / period).mean()
        Array1 = MA.ewm(alpha=1 / period).mean()
    elif ma_method == "lwma":
        weights = np.arange(1, period + 1)
        MA = price.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        Array1 = MA.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    # Initialize the MAIN and SIGNAL arrays (SELL and BUYY in og MQL4)
    MAIN = np.zeros_like(price)
    SIGNAL = np.zeros_like(price)

    # Iterate over the data and calculate the BUYY and SELL values
    for i in range(signal_bar, len(price)):
        slotLL = price_data["Low"][i] if show_high_low else MA[i]
        slotHH = price_data["High"][i] if show_high_low else MA[i]

        if Array1[i] > Array1[i - 1]:
            SIGNAL[i] = slotLL
            MAIN[i] = Array1[i]
        if Array1[i] < Array1[i - 1]:
            SIGNAL[i] = slotHH
            MAIN[i] = Array1[i]

    # Create a DataFrame to store the calculated indicator values
    indicator_values = pd.DataFrame({"Main": MAIN, "Signal": SIGNAL})

    return indicator_values


def TwiggsMF(price_data: pd.DataFrame, period: int = 21) -> pd.DataFrame:
    """
    Calculate the Twigg's Money Flow indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        Pandas DataFrame containing OHLCV price data.
    period : int, default 21
        The number of periods for the moving average.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the calculated indicator values.
    """
    # Check if there are at least 4 bars
    if len(price_data) < period:
        return pd.DataFrame()

    # Calculate TR, ADV, and Vol
    price_data["TRH"] = price_data[["High", "Close"]].max(axis=1)
    price_data["TRL"] = price_data[["Low", "Close"]].min(axis=1)
    price_data["TR"] = price_data["TRH"] - price_data["TRL"]

    price_data["ADV"] = (
        (2 * price_data["Close"] - price_data["TRL"] - price_data["TRH"])
        / price_data["TR"]
        * price_data["Volume"]
    )
    price_data["Vol"] = price_data["Volume"]

    # Handle any division by zero that may have occurred
    price_data["ADV"].replace([np.inf, -np.inf], 0, inplace=True)

    # Calculate k
    k = 2 / (period + 1)

    # Calculate WMA_ADV and WMA_V
    price_data["WMA_ADV"] = price_data["ADV"].ewm(span=period, adjust=False).mean()
    price_data["WMA_V"] = price_data["Vol"].ewm(span=period, adjust=False).mean()

    # Calculate TMF
    price_data["TMF"] = price_data["WMA_ADV"] / price_data["WMA_V"]

    # Create a DataFrame to store the calculated indicator values
    indicator_values = pd.DataFrame({"TMF": price_data["TMF"]})

    return indicator_values


def UF2018(price_data: pd.DataFrame, period: int = 54) -> pd.DataFrame:
    """
    Calculate the uf2018 indicator.

    Parameters
    ----------
    price_data : pd.DataFrame
        Pandas DataFrame containing OHLCV price data.
    period : int, default 54
        The period for the Zig Zag calculation.
    bar_n : int, default 1000
        The number of bars to calculate the indicator for.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the calculated indicator values.
    """
    SELL = pd.Series(index=price_data.index, dtype=float)
    BUY = pd.Series(index=price_data.index, dtype=float)

    li_20 = 0
    li_16 = 0
    index_24 = 0
    bar_n = len(price_data) - 1
    high_60 = price_data["High"][bar_n]
    low_68 = price_data["Low"][bar_n]
    down = False
    up = False

    for i in range(bar_n, -1, -1):
        low = 10000000
        high = -100000000

        for j in range(i + period, i, -1):
            if j > bar_n:
                continue
            if price_data["Low"][j] < low:
                low = price_data["Low"][j]
            if price_data["High"][j] > high:
                high = price_data["High"][j]

        if price_data["Low"][i] < low and price_data["High"][i] > high:
            li_16 = 2
        else:
            if price_data["Low"][i] < low:
                li_16 = -1
            if price_data["High"][i] > high:
                li_16 = 1

        if li_16 != li_20 and li_20 != 0:
            if li_16 == 2:
                li_16 = -li_20
                high_60 = price_data["High"][i]
                low_68 = price_data["Low"][i]
                down = False
                up = False

            index_24 += 1

            up = True if li_16 == 1 else False
            down = True if li_16 == -1 else False

            high_60 = price_data["High"][i]
            low_68 = price_data["Low"][i]

        if li_16 == 1 and price_data["High"][i] >= high_60:
            high_60 = price_data["High"][i]

        if li_16 == -1 and price_data["Low"][i] <= low_68:
            low_68 = price_data["Low"][i]

        li_20 = li_16

        BUY[i] = 1 if up else 0
        SELL[i] = 1 if down else 0

    # Return the results as a DataFrame
    return pd.DataFrame({"BUY": BUY, "SELL": SELL})


def LSMA(df: pd.DataFrame, period: int = 14, shift: int = 0) -> pd.Series:
    """
    Least Squares Moving Average (LSMA) calculation.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' column.
    period (int): Period for LSMA calculation.
    shift (int): Number of periods to shift the source data.

    Returns:
    pd.Series: Series of LSMA values.
    """
    lengthvar = (period + 1) / 3
    weights = np.array([(i - lengthvar) for i in range(period, 0, -1)])

    # Define a rolling apply function using dot product of weights
    def calc_lsma(window):
        return np.dot(weights, window) * 6 / (period * (period + 1))

    shifted_close = df["Close"].shift(shift)
    lsma_series = shifted_close.rolling(window=period).apply(calc_lsma, raw=True)

    return lsma_series


def AcceleratorLSMA(
    df: pd.DataFrame, long_period: int = 21, short_period: int = 9
) -> pd.DataFrame:
    """
    Calculate Accelerator/Decelerator Oscillator.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' column

    Returns:
    pd.DataFrame: DataFrame with calculated values
    """
    # Calculate LSMA for both short and long periods
    short_lsma = LSMA(df, short_period)
    long_lsma = LSMA(df, long_period)

    # Calculate ExtBuffer3 as the difference between short and long LSMA
    ExtBuffer3 = short_lsma - long_lsma

    # Handle any NaN values that may result from LSMA or the rolling operation
    ExtBuffer3 = ExtBuffer3.fillna(0)

    # Calculate rolling mean (ExtBuffer4) for ExtBuffer3 using short_period
    ExtBuffer4 = ExtBuffer3.rolling(window=short_period).mean()

    # Handle NaN values in ExtBuffer4 (if any)
    ExtBuffer4 = ExtBuffer4.fillna(0)

    # Current difference from the rolling mean
    current = ExtBuffer3 - ExtBuffer4

    # Shift current for comparison with previous
    prev = np.roll(current, 1)

    # Determine the direction (up or down)
    up = current > prev

    # Separate into positive (up) and negative (down) values
    ExtBuffer1 = np.where(up, current, 0)
    ExtBuffer2 = np.where(~up, current, 0)
    ExtBuffer0 = current  # Final oscillator values

    # Return results in a DataFrame
    result = pd.DataFrame(
        {
            "ExtBuffer0": ExtBuffer0,
            "ExtBuffer1": ExtBuffer1,
            "ExtBuffer2": ExtBuffer2,
            "ExtBuffer3": ExtBuffer3,
            "ExtBuffer4": ExtBuffer4,
        }
    )

    return result


def SSL(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    Calculate SSL channels.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close', 'High', and 'Low' columns
    lb (int): Lookback period

    Returns:
    pd.DataFrame: DataFrame with calculated values
    """
    # Calculate rolling high and low
    high_sma = df["High"].rolling(window=period).mean()
    low_sma = df["Low"].rolling(window=period).mean()

    # Initialize hlv as a DataFrame of zeros
    hlv = pd.Series(0, index=df.index)

    # Conditions for hlv
    hlv[df["Close"] > high_sma] = 1
    hlv[df["Close"] < low_sma] = -1
    hlv = hlv.ffill()

    # Initialize ssld and sslu
    ssld = pd.Series(index=df.index, dtype=float)
    sslu = pd.Series(index=df.index, dtype=float)

    # Conditions for ssld and sslu
    ssld[hlv == 1] = low_sma
    ssld[hlv == -1] = high_sma
    sslu[hlv == 1] = high_sma
    sslu[hlv == -1] = low_sma

    ssld = ssld.ffill()
    sslu = sslu.ffill()

    return pd.DataFrame({"SSL_Down": ssld, "SSL_Up": sslu})


# Volume Functions


def ADX(price_data: pd.DataFrame, period=14) -> pd.DataFrame:
    # Calculate the True Range (TR)
    tr = np.maximum(
        price_data["High"] - price_data["Low"],
        np.maximum(
            abs(price_data["High"] - price_data["Close"].shift()),
            abs(price_data["Low"] - price_data["Close"].shift()),
        ),
    )

    # Calculate the Directional Movement (DM) and True Directional Movement (TDM)
    dm_plus = np.where(
        price_data["High"] - price_data["High"].shift()
        > price_data["Low"].shift() - price_data["Low"],
        price_data["High"] - price_data["High"].shift(),
        0,
    )
    dm_minus = np.where(
        price_data["Low"].shift() - price_data["Low"]
        > price_data["High"] - price_data["High"].shift(),
        price_data["Low"].shift() - price_data["Low"],
        0,
    )
    tdm_plus = pd.Series(dm_plus).rolling(period).sum()
    tdm_minus = pd.Series(dm_minus).rolling(period).sum()

    # Calculate the Positive Directional Index (+DI) and Negative Directional Index (-DI)
    di_plus = 100 * tdm_plus / tr.rolling(period).sum()
    di_minus = 100 * tdm_minus / tr.rolling(period).sum()

    # Calculate the Directional Movement Index (DX)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

    # Calculate the ADX
    adx = dx.rolling(period).mean()

    result = pd.DataFrame(
        {
            "ADX": adx,
            "+DI": di_plus,
            "-DI": di_minus,
        }
    )

    # Return the dataframe with the ADX values
    return result


def TDFI(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """
    Calculate the Trend Direction Force Index (TDFI) using a Pandas DataFrame.

    :param df: Pandas DataFrame containing OHLC and Volume columns
    :param period: Period for the Exponential Moving Average (EMA) calculation, default is 13
    :return: Pandas Series containing the TDFI values
    """
    # Calculate the Force Index
    force_index = (df["Close"] - df["Close"].shift(1)) * df["Volume"]

    # Calculate the smoothing factor
    smoothing_factor = 2 / (period + 1)

    # Calculate the smoothed Force Index (TDFI) using EMA
    smoothed_force_index = force_index.ewm(alpha=smoothing_factor, adjust=False).mean()

    # Normalize the smoothed Force Index to range between -1 and 1
    tdfi = (
        smoothed_force_index / smoothed_force_index.abs().rolling(window=period).max()
    )

    # Convert to DataFrame
    tdfi = pd.Series(tdfi)

    return tdfi


def WAE(
    price_data: pd.DataFrame,
    minutes: int = 0,
    sensitivity: int = 150,
    dead_zone_pip: int = 15,
) -> pd.DataFrame:
    """
    Calculate the Waddah Attar Explosion indicator values and return them as a Pandas DataFrame.

    Parameters:
    price_data (pd.DataFrame): Input DataFrame containing OHLCV columns.
    minutes (int): Timeframe in minutes. Default is 0.
    sensitivity (int): Sensitivity parameter. Default is 150.
    dead_zone_pip (int): Dead zone pip parameter. Default is 15.

    Returns:
    pd.DataFrame: DataFrame of calculated indicator values.
    """
    # Resample price data to the desired timeframe
    if minutes > 0:
        price_data = price_data.resample(f"{minutes}T").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def macd(
        series: pd.Series, fast_period: int, slow_period: int, signal_period: int
    ) -> pd.Series:
        fast_ema = ema(series, fast_period)
        slow_ema = ema(series, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, signal_period)
        return macd_line, signal_line

    def bollinger_bands(series: pd.Series, period: int, std_dev: float) -> pd.DataFrame:
        middle = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = middle + (std_dev * std_dev)
        lower = middle - (std_dev * std_dev)
        return upper, middle, lower

    close = price_data["Close"]

    # Calculate MACD
    macd_line, signal_line = macd(
        close, fast_period=20, slow_period=40, signal_period=9
    )

    # Calculate Bollinger Bands
    upper, middle, lower = bollinger_bands(close, period=20, std_dev=2)

    # Calculate Trend and Explo
    trend = (macd_line - signal_line) * sensitivity
    explo = upper - lower

    # Calculate Dead zone
    dead_zone = price_data["Close"].apply(lambda x: x * dead_zone_pip)

    # Create output DataFrame
    output = pd.DataFrame(index=price_data.index)
    output["Trend"] = trend
    output["Explosion"] = explo
    output["Dead"] = dead_zone

    return output


def NormalizedVolume(price_data, period: int = 14) -> pd.DataFrame:
    """Calculates the Normalized Volume indicator values for a given OHLCV dataframe.

    Args:
        price_data (_type_): A Pandas DataFrame of OHLCV price data.
        period (int, optional): The period for the MA of the volume. Defaults to 14.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the normalized volume values.
    """
    volume = price_data["Volume"].values

    volume_buffer = volume[::-1].copy()

    # Calculate MA of volume buffer
    volume_ma = pd.Series(volume_buffer).rolling(period).mean().values

    vol = volume / volume_ma * 100
    up = np.where(vol > 100, vol, np.nan)
    dn = np.where(vol <= 100, vol, np.nan)

    return pd.DataFrame({"Vol": vol}, index=price_data.index)


def VolatilityRatio(
    price_data: pd.DataFrame, period: int = 25, inp_price: str = "Close"
) -> pd.DataFrame:
    """
    Calculate the Volatility Ratio.

    Parameters:
    price_data (pd.DataFrame): OHLCV price data
    period (int): Volatility period
    inp_price (str): Column name in the price_data DataFrame for the price (default 'Close')

    Returns:
    pd.DataFrame: DataFrame containing the Volatility Ratio values
    """
    rates_total = len(price_data)
    price = price_data[inp_price].values

    val = np.empty(rates_total, dtype=float)
    valda = np.empty(rates_total, dtype=float)
    valdb = np.empty(rates_total, dtype=float)
    valc = np.empty(rates_total, dtype=int)

    m_array = {
        "price": np.zeros(rates_total),
        "price2": np.zeros(rates_total),
        "sum": np.zeros(rates_total),
        "sum2": np.zeros(rates_total),
        "sumd": np.zeros(rates_total),
        "deviation": np.zeros(rates_total),
    }

    for i in range(rates_total - 1, -1, -1):
        m_array["price"][i] = price[i]
        m_array["price2"][i] = price[i] * price[i]

        if i > period:
            m_array["sum"][i] = (
                m_array["sum"][i - 1]
                + m_array["price"][i]
                - m_array["price"][i - period]
            )
            m_array["sum2"][i] = (
                m_array["sum2"][i - 1]
                + m_array["price2"][i]
                - m_array["price2"][i - period]
            )
        else:
            m_array["sum"][i] = m_array["price"][i]
            m_array["sum2"][i] = m_array["price2"][i]
            for k in range(1, period):
                if i >= k:
                    m_array["sum"][i] += m_array["price"][i - k]
                    m_array["sum2"][i] += m_array["price2"][i - k]

        m_array["deviation"][i] = np.sqrt(
            (m_array["sum2"][i] - m_array["sum"][i] * m_array["sum"][i] / period)
            / period
        )

        if i > period:
            m_array["sumd"][i] = (
                m_array["sumd"][i - 1]
                + m_array["deviation"][i]
                - m_array["deviation"][i - period]
            )
        else:
            m_array["sumd"][i] = m_array["deviation"][i]
            for k in range(1, period):
                if i >= k:
                    m_array["sumd"][i] += m_array["deviation"][i - k]

        deviation_average = m_array["sumd"][i] / period
        val[i] = (
            m_array["deviation"][i] / deviation_average if deviation_average != 0 else 1
        )
        valc[i] = 1 if val[i] > 1 else 2 if val[i] < 1 else 0

        if valc[i] == 2:
            valda[i] = valdb[i] = np.nan
        elif valc[i] == 1:
            valda[i] = val[i]
            valdb[i] = np.nan
        else:
            valda[i] = np.nan
            valdb[i] = val[i]

    vr = pd.DataFrame(
        {"VR": val, "VR Up": valda, "VR Down": valdb}, index=price_data.index
    )

    return vr


# TODO
def TCF(
    price_data: pd.DataFrame,
    period: int = 20,
    count_bars: int = 5000,
    t3_period: int = 5,
    b: float = 0.618,
) -> pd.DataFrame:

    close = price_data["Close"]
    bars = len(close)

    accounted_bars = max(0, bars - count_bars)

    t3_k_p = pd.Series(index=close.index, dtype=float)
    t3_k_n = pd.Series(index=close.index, dtype=float)

    for cnt in range(accounted_bars, bars):
        shift = bars - 1 - cnt

        change_p = close.shift(-1) - close
        change_n = close - close.shift(-1)

        cf_p = change_p.where(change_p > 0, 0).cumsum()
        cf_n = change_n.where(change_n > 0, 0).cumsum()

        ch_p = change_p[shift : shift + period].sum()
        ch_n = change_n[shift : shift + period].sum()
        cff_p = cf_p[shift : shift + period].sum()
        cff_n = cf_n[shift : shift + period].sum()

        k_p = ch_p - cff_n
        k_n = ch_n - cff_p

        a1 = k_p
        a2 = k_n

        e1 = e2 = e3 = e4 = e5 = e6 = 0
        e12 = e22 = e32 = e42 = e52 = e62 = 0

        b2 = b * b
        b3 = b2 * b
        c1 = -b3
        c2 = 3 * (b2 + b3)
        c3 = -3 * (2 * b2 + b + b3)
        c4 = 1 + 3 * b + b3 + 3 * b2

        n1 = t3_period
        if n1 < 1:
            n1 = 1
        n1 = 1 + 0.5 * (n1 - 1)
        w1 = 2 / (n1 + 1)
        w2 = 1 - w1

        for _ in range(t3_period):
            e1 = w1 * a1 + w2 * e1
            e2 = w1 * e1 + w2 * e2
            e3 = w1 * e2 + w2 * e3
            e4 = w1 * e3 + w2 * e4
            e5 = w1 * e4 + w2 * e5
            e6 = w1 * e5 + w2 * e6

            e12 = w1 * a2 + w2 * e12
            e22 = w1 * e12 + w2 * e22
            e32 = w1 * e22 + w2 * e32
            e42 = w1 * e32 + w2 * e42
            e52 = w1 * e42 + w2 * e52
            e62 = w1 * e52 + w2 * e62

        t3_k_p[shift] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        t3_k_n[shift] = c1 * e62 + c2 * e52 + c3 * e42 + c4 * e32

    tcf = pd.DataFrame({"T3KP": t3_k_p, "T3KN": t3_k_n})

    return tcf


# TODO
def DSP(data, signal_period=9, dsp_period=14):
    pass


# Auto-generated by merge_indicators.py
# Combined indicator calculation functions


# === BEGIN 2ndOrderGaussianHighPassFilterMtfZones_calc.py ===
def SecondOrderGaussianHighPassFilterMtfZones(
    df: pd.DataFrame,
    alpha: float = 0.14,
    timeframe: str | None = None,
    interpolate: bool = True,
    maxbars: int = 2000,
) -> pd.Series:
    """Return the primary indicator line aligned to df.index; vectorized where possible, handles NaNs, stable defaults.

    Parameters
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume'].
    - alpha: smoothing coefficient (MQL default 0.14).
    - timeframe: pandas offset alias (e.g., '15T') for multi-timeframe; None to use native timeframe.
    - interpolate: when timeframe is not None, linearly interpolate HTF values to LTF bars if True; else step (ffill).
    - maxbars: compute last N bars (like MQL maxbars); earlier values are NaN.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype=float, name="HPF")
    if "Timestamp" not in df.columns or "Close" not in df.columns:
        raise ValueError("df must contain 'Timestamp' and 'Close' columns")

    # Work on a time-ascending copy for consistent resampling/processing
    ts = pd.to_datetime(df["Timestamp"])
    asc_idx = np.argsort(ts.values.astype("datetime64[ns]"))
    df_asc = df.iloc[asc_idx].copy()
    df_asc["Timestamp"] = pd.to_datetime(df_asc["Timestamp"])
    dt_index = pd.DatetimeIndex(df_asc["Timestamp"])

    def _hpf_from_close(
        close_arr: np.ndarray, a: float, b: float, d: float
    ) -> np.ndarray:
        n = close_arr.shape[0]
        # Reverse so index 0 is the most recent (MQL-style), then compute backward recursion
        rev = close_arr[::-1].astype(float)
        # Fill NaNs to avoid propagating through recursion; then restore NaNs after computation
        rev_series = pd.Series(rev)
        rev_filled = rev_series.ffill().bfill().to_numpy()
        s = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            c0 = rev_filled[i]
            c1 = rev_filled[i + 1] if (i + 1) < n else 0.0
            c2 = rev_filled[i + 2] if (i + 2) < n else 0.0
            s1 = s[i + 1] if (i + 1) < n else 0.0
            s2 = s[i + 2] if (i + 2) < n else 0.0
            s[i] = a * (c0 - 2.0 * c1 + c2) + b * s1 - d * s2
        out = s[::-1]
        # Restore NaNs where original close was NaN
        nan_mask = ~np.isfinite(close_arr)
        if nan_mask.any():
            out[nan_mask] = np.nan
        return out

    a = (1.0 - alpha / 2.0) ** 2
    b = 2.0 * (1.0 - alpha)
    d = (1.0 - alpha) ** 2

    if timeframe is None:
        # Compute on native timeframe
        close_asc = df_asc["Close"].to_numpy(dtype=float)
        n_total = close_asc.shape[0]
        n_use = min(maxbars if maxbars is not None else n_total, n_total)
        hpf_full = np.full(n_total, np.nan, dtype=float)
        if n_use > 0:
            hpf_tail = _hpf_from_close(close_asc[-n_use:], a, b, d)
            hpf_full[-n_use:] = hpf_tail
        hpf_asc = pd.Series(hpf_full, index=df_asc.index, name="HPF")
    else:
        # Multi-timeframe: resample Close to HTF, compute HPF there, then align to LTF
        close_htf = pd.Series(df_asc["Close"].to_numpy(dtype=float), index=dt_index)
        close_htf = close_htf.resample(timeframe).last().dropna()
        if close_htf.empty:
            hpf_asc = pd.Series(np.nan, index=df_asc.index, name="HPF")
        else:
            hpf_vals_htf = _hpf_from_close(close_htf.to_numpy(), a, b, d)
            hpf_htf = pd.Series(hpf_vals_htf, index=close_htf.index, name="HPF")
            if interpolate:
                union_idx = hpf_htf.index.union(dt_index)
                aligned = (
                    hpf_htf.reindex(union_idx)
                    .sort_index()
                    .interpolate(method="time")
                    .reindex(dt_index)
                )
            else:
                aligned = hpf_htf.reindex(dt_index, method="ffill")
            # Apply maxbars on LTF alignment
            n_total = aligned.shape[0]
            n_use = min(maxbars if maxbars is not None else n_total, n_total)
            hpf_arr = aligned.to_numpy()
            if n_use < n_total:
                hpf_arr[: n_total - n_use] = np.nan
            hpf_asc = pd.Series(hpf_arr, index=df_asc.index, name="HPF")

    # Map back to original df.index order
    inv_order = np.empty_like(asc_idx)
    inv_order[asc_idx] = np.arange(len(asc_idx))
    hpf_out = hpf_asc.reindex(df_asc.index).to_numpy()
    hpf_out = hpf_out[inv_order]

    return pd.Series(hpf_out, index=df.index, name="HPF")


# === END 2ndOrderGaussianHighPassFilterMtfZones_calc.py ===


# === BEGIN 3rdgenma_calc.py ===
def ThirdGenMA(
    df: pd.DataFrame,
    ma_period: int = 220,
    sampling_period: int = 50,
    method: int = 1,
    applied_price: int = 5,
) -> pd.DataFrame:
    """
    3rd Generation Moving Average (Durschner).
    Returns:
      - MA3G: (alpha + 1) * MA1 - alpha * MA2
      - MA1:  first-pass MA of the applied price

    Params:
      - ma_period: main MA period (default 220)
      - sampling_period: sampling MA period for second pass (default 50)
      - method: 0=SMA, 1=EMA, 2=SMMA (Wilder/Smoothed), 3=LWMA (default 1)
      - applied_price: 0=Close,1=Open,2=High,3=Low,4=Median,5=Typical,6=Weighted (default 5)

    Output aligned to df.index and preserves length with NaNs for warmup.
    """
    if ma_period <= 0 or sampling_period <= 0:
        raise ValueError("ma_period and sampling_period must be positive integers.")
    if ma_period < 2 * sampling_period:
        raise ValueError("ma_period should be >= sampling_period * 2.")

    # Select applied price
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    if applied_price == 0:
        price = close
    elif applied_price == 1:
        price = open_
    elif applied_price == 2:
        price = high
    elif applied_price == 3:
        price = low
    elif applied_price == 4:
        price = (high + low) / 2.0
    elif applied_price == 5:
        price = (high + low + close) / 3.0
    elif applied_price == 6:
        price = (high + low + 2.0 * close) / 4.0
    else:
        price = close

    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).mean()

    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def _smma(s: pd.Series, n: int) -> pd.Series:
        # Wilder/Smoothed MA: seed with SMA(n), then recursive
        arr = s.to_numpy(dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        if len(arr) == 0:
            return pd.Series(out, index=s.index)

        seed = s.rolling(window=n, min_periods=n).mean().to_numpy()
        start = n - 1
        if start < len(arr):
            out[start] = seed[start]
            for i in range(start + 1, len(arr)):
                xi = arr[i]
                yi_1 = out[i - 1]
                if np.isnan(xi) or np.isnan(yi_1):
                    out[i] = np.nan
                else:
                    out[i] = (yi_1 * (n - 1) + xi) / n
        return pd.Series(out, index=s.index)

    def _lwma(s: pd.Series, n: int) -> pd.Series:
        weights = np.arange(1, n + 1, dtype=float)
        denom = weights.sum()
        # Using rolling.apply to handle NaNs within windows gracefully
        return s.rolling(window=n, min_periods=n).apply(
            lambda x: np.dot(x, weights) / denom, raw=True
        )

    def _ma(s: pd.Series, n: int, m: int) -> pd.Series:
        if m == 0:
            return _sma(s, n)
        elif m == 1:
            return _ema(s, n)
        elif m == 2:
            return _smma(s, n)
        elif m == 3:
            return _lwma(s, n)
        else:
            return _sma(s, n)

    # First pass MA
    ma1 = _ma(price, ma_period, method)

    # Second pass MA on the first MA (same method)
    ma2 = _ma(ma1, sampling_period, method)

    # Parameters
    lambda_ = float(ma_period) / float(sampling_period)
    denom = ma_period - lambda_
    # denom should be > 0 given earlier validation
    alpha = lambda_ * (ma_period - 1.0) / denom

    ma3g = (alpha + 1.0) * ma1 - alpha * ma2

    out = pd.DataFrame(
        {
            "MA3G": ma3g.to_numpy(dtype=float),
            "MA1": ma1.to_numpy(dtype=float),
        },
        index=df.index,
    )
    return out


# === END 3rdgenma_calc.py ===


# === BEGIN AdaptiveSmootherTriggerlinesMtfAlertsNmc_calc.py ===
def AdaptiveSmootherTriggerlinesMtfAlertsNmc(
    df: pd.DataFrame,
    LsmaPeriod: int = 50,
    LsmaPrice: int = 0,
    AdaptPeriod: int = 21,
    MultiColor: bool = True,
) -> pd.DataFrame:
    """
    Triggerlines (adaptive smoother) converted from MQL4.
    Returns all lines aligned to df.index:
      - lsma: adaptive smoother
      - lsmaUa, lsmaUb: segmented down-slope parts for multi-color plotting
      - lwma: previous-bar lsma
      - lwmaUa, lwmaUb: segmented down-slope parts of lwma
      - lstrend, lwtrend: +1/-1 trend of lsma/lwma based on slope sign (ties forward-filled)

    Parameters:
      - LsmaPeriod: base period for smoothing (default 50)
      - LsmaPrice: price type (0 Close,1 Open,2 High,3 Low,4 Median,5 Typical,6 Weighted)
      - AdaptPeriod: adaptation lookback for std-dev/avg of std-dev (default 21)
      - MultiColor: if True, compute Ua/Ub segmented series for down-trend
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "lsma",
                "lsmaUa",
                "lsmaUb",
                "lwma",
                "lwmaUa",
                "lwmaUb",
                "lstrend",
                "lwtrend",
            ],
            index=df.index,
        )

    # Map MQL4 price constants
    if LsmaPrice == 0:
        price = df["Close"].astype(float).to_numpy()
    elif LsmaPrice == 1:
        price = df["Open"].astype(float).to_numpy()
    elif LsmaPrice == 2:
        price = df["High"].astype(float).to_numpy()
    elif LsmaPrice == 3:
        price = df["Low"].astype(float).to_numpy()
    elif LsmaPrice == 4:
        price = ((df["High"] + df["Low"]) / 2.0).astype(float).to_numpy()
    elif LsmaPrice == 5:
        price = ((df["High"] + df["Low"] + df["Close"]) / 3.0).astype(float).to_numpy()
    elif LsmaPrice == 6:
        price = (
            ((df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0)
            .astype(float)
            .to_numpy()
        )
    else:
        price = df["Close"].astype(float).to_numpy()

    close = df["Close"].astype(float)

    # Adaptive period calculation
    dev = close.rolling(window=AdaptPeriod, min_periods=1).std(ddof=0)
    avg = dev.rolling(window=AdaptPeriod, min_periods=1).mean()
    dev_vals = dev.to_numpy()
    avg_vals = avg.to_numpy()

    period = np.empty(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dev_vals != 0.0, avg_vals / dev_vals, np.nan)
    # Where dev==0 or ratio nan, fallback to LsmaPeriod
    period[:] = LsmaPeriod
    mask_ratio = np.isfinite(ratio)
    period[mask_ratio & (dev_vals != 0.0)] = (
        LsmaPeriod * ratio[mask_ratio & (dev_vals != 0.0)]
    )
    period = np.maximum(period, 3.0)

    # iSmooth recursive filter (stateful, dynamic alpha per bar)
    alpha = 0.45 * (period - 1.0) / (0.45 * (period - 1.0) + 2.0)
    y = np.full(n, np.nan, dtype=float)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0

    for t in range(n):
        pr = price[t]
        if not np.isfinite(pr):
            # Do not update states on NaN input; output NaN
            y[t] = np.nan
            continue

        if t <= 2:
            s0 = pr
            s1 = 0.0
            s2 = pr
            s3 = 0.0
            s4 = pr
            y[t] = pr
            continue

        a = float(alpha[t]) if np.isfinite(alpha[t]) else 0.0
        prev_s4 = s4
        prev_s3 = s3
        prev_s1 = s1

        s0 = pr + a * (s0 - pr)
        s1 = (pr - s0) * (1.0 - a) + a * prev_s1
        s2 = s0 + s1
        s3 = (s2 - prev_s4) * (1.0 - a) * (1.0 - a) + (a * a) * prev_s3
        s4 = s3 + prev_s4
        y[t] = s4

    lsma = pd.Series(y, index=df.index, name="lsma")
    lwma = lsma.shift(1)
    lwma.name = "lwma"

    # Trends based on slope sign; ties keep previous value
    def slope_trend(s: pd.Series) -> pd.Series:
        d = s.diff()
        tr = pd.Series(
            np.where(d > 0, 1.0, np.where(d < 0, -1.0, np.nan)), index=s.index
        )
        return tr.ffill()

    lstrend = slope_trend(lsma).rename("lstrend")
    lwtrend = slope_trend(lwma).rename("lwtrend")

    # MultiColor segmented down-trend series
    def split_down_runs(base: pd.Series, trend: pd.Series):
        if not MultiColor:
            return base.where(
                pd.Series([False] * len(base), index=base.index)
            ), base.where(pd.Series([False] * len(base), index=base.index))
        m = trend == -1.0
        starts = m & (~m.shift(1, fill_value=False))
        run_id = starts.cumsum().where(m, 0).astype(int)
        parity = run_id % 2
        ua = base.where((run_id > 0) & (parity == 1))
        ub = base.where((run_id > 0) & (parity == 0))
        return ua, ub

    lsmaUa, lsmaUb = split_down_runs(lsma, lstrend)
    lwmaUa, lwmaUb = split_down_runs(lwma, lwtrend)

    out = pd.DataFrame(
        {
            "lsma": lsma,
            "lsmaUa": lsmaUa,
            "lsmaUb": lsmaUb,
            "lwma": lwma,
            "lwmaUa": lwmaUa,
            "lwmaUb": lwmaUb,
            "lstrend": lstrend,
            "lwtrend": lwtrend,
        },
        index=df.index,
    )
    return out


# === END AdaptiveSmootherTriggerlinesMtfAlertsNmc_calc.py ===


# === BEGIN ASO_calc.py ===
def ASO(
    df: pd.DataFrame,
    period: int = 10,
    mode: int = 0,
    bulls: bool = True,
    bears: bool = True,
) -> pd.DataFrame:
    """Average Sentiment Oscillator (ASO).
    Returns a DataFrame with columns ['ASO_Bulls','ASO_Bears'] aligned to df.index.
    Parameters:
      - period: lookback period for group metrics and SMA smoothing (default 10)
      - mode: 0 = average of intra-bar and group, 1 = intra-bar only, 2 = group only (default 0)
      - bulls/bears: enable computation for the respective line (default True)
    """
    p = max(int(period), 1)
    m = int(mode)
    if m not in (0, 1, 2):
        m = 0

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)

    intrarange = h - l
    intrarange_safe = intrarange.copy()
    intrarange_safe[intrarange_safe == 0] = 1.0

    intrabar_bulls = (((c - l) + (h - o)) / 2.0) * 100.0 / intrarange_safe
    intrabar_bears = (((h - c) + (o - l)) / 2.0) * 100.0 / intrarange_safe

    grouplow = l.rolling(window=p, min_periods=p).min()
    grouphigh = h.rolling(window=p, min_periods=p).max()
    groupopen = o.shift(p - 1)
    grouprange = grouphigh - grouplow
    grouprange_safe = grouprange.copy()
    grouprange_safe[grouprange_safe == 0] = 1.0

    group_bulls = (
        (((c - grouplow) + (grouphigh - groupopen)) / 2.0) * 100.0 / grouprange_safe
    )
    group_bears = (
        (((grouphigh - c) + (groupopen - grouplow)) / 2.0) * 100.0 / grouprange_safe
    )

    if m == 0:
        temp_bulls = (intrabar_bulls + group_bulls) / 2.0
        temp_bears = (intrabar_bears + group_bears) / 2.0
    elif m == 1:
        temp_bulls = intrabar_bulls
        temp_bears = intrabar_bears
    else:  # m == 2
        temp_bulls = group_bulls
        temp_bears = group_bears

    aso_bulls = temp_bulls.rolling(window=p, min_periods=p).mean()
    aso_bears = temp_bears.rolling(window=p, min_periods=p).mean()

    out = pd.DataFrame({"ASO_Bulls": aso_bulls, "ASO_Bears": aso_bears}, index=df.index)
    if not bulls:
        out["ASO_Bulls"] = np.nan
    if not bears:
        out["ASO_Bears"] = np.nan
    return out


# === END ASO_calc.py ===


# === BEGIN ATRBasedEMAVariant1_calc.py ===
def ATRBasedEMAVariant1(
    df: pd.DataFrame, ema_fastest: float = 14.0, multiplier: float = 300.0
) -> pd.DataFrame:
    """Return ALL lines as columns ['EMA_ATR_var1','EMA_Equivalent']; aligned to df.index.
    Based on ATR% (High/Low) smoothed by EMA(14), computes a dynamic EMA period and applies it to Close.
    Higher ATR -> slower EMA.
    """
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")

    # Avoid division by zero/negatives
    ratio = high / low.where(low > 0)

    # EMA of ratio with alpha=1/14 (equivalent to int_atr*13/14 + ratio/14)
    alpha_atr = 1.0 / 14.0
    int_atr = ratio.ewm(alpha=alpha_atr, adjust=False, min_periods=1).mean()

    ema_equiv = ((int_atr - 1.0) * multiplier + 1.0) * float(ema_fastest)

    # Variable alpha for the Close EMA
    alpha_dyn = 2.0 / (ema_equiv + 1.0)

    n = len(df)
    signal = np.full(n, np.nan, dtype=float)
    c_vals = close.to_numpy(dtype=float)
    a_vals = alpha_dyn.to_numpy(dtype=float)

    # Find first index where both alpha and close are valid
    valid_mask = np.isfinite(c_vals) & np.isfinite(a_vals)
    if valid_mask.any():
        start = int(np.argmax(valid_mask))  # first True index
        signal[start] = c_vals[start]
        for t in range(start + 1, n):
            c = c_vals[t]
            a = a_vals[t]
            if np.isfinite(c) and np.isfinite(a):
                signal[t] = signal[t - 1] * (1.0 - a) + c * a
            else:
                signal[t] = signal[t - 1]

    out = pd.DataFrame(
        {
            "EMA_ATR_var1": signal,
            "EMA_Equivalent": ema_equiv.astype(float).reindex(df.index),
        },
        index=df.index,
    )
    return out


# === END ATRBasedEMAVariant1_calc.py ===


# === BEGIN BamsBung3_calc.py ===
def BamsBung3(
    df: pd.DataFrame,
    length: int = 14,
    deviation: float = 2.0,
    money_risk: float = 0.02,
    signal_mode: int = 1,  # 1: signals & stops, 0: only stops, 2: only signals
    line_mode: int = 1,  # 1: show line, 0: hide line
) -> pd.DataFrame:
    """Return all Bams Bung 3 outputs as columns aligned to df.index.
    - Uses SMA-based Bollinger Bands on Close with ddof=0 for std.
    - Sequential logic replicates the original MQL4 algorithm.
    - Preserves length; NaNs at warmup (pre-length).
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    n = len(close)

    sma = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std(ddof=0)
    upper = sma + deviation * std
    lower = sma - deviation * std

    smax = upper.to_numpy(dtype=float)
    smin = lower.to_numpy(dtype=float)

    smax_adj = np.copy(smax)
    smin_adj = np.copy(smin)
    bsmax = np.full(n, np.nan, dtype=float)
    bsmin = np.full(n, np.nan, dtype=float)
    trend = np.zeros(n, dtype=int)

    up_stop = np.full(n, np.nan, dtype=float)
    down_stop = np.full(n, np.nan, dtype=float)
    up_signal = np.full(n, np.nan, dtype=float)
    down_signal = np.full(n, np.nan, dtype=float)
    up_line = np.full(n, np.nan, dtype=float)
    down_line = np.full(n, np.nan, dtype=float)

    c = close.to_numpy(dtype=float)

    for t in range(n):
        if (
            t == 0
            or np.isnan(smax[t])
            or np.isnan(smin[t])
            or np.isnan(c[t])
            or np.isnan(smax_adj[t - 1])
            or np.isnan(smin_adj[t - 1])
        ):
            # Warmup or insufficient history
            if t > 0:
                trend[t] = trend[t - 1]
            continue

        tr = trend[t - 1]

        if c[t] > smax_adj[t - 1]:
            tr = 1
        if c[t] < smin_adj[t - 1]:
            tr = -1

        # Adjust smin/smax based on trend persistence
        smin_t = smin[t]
        smax_t = smax[t]
        if tr > 0:
            smin_t = np.maximum(smin_t, smin_adj[t - 1])
        if tr < 0:
            smax_t = np.minimum(smax_t, smax_adj[t - 1])

        smin_adj[t] = smin_t
        smax_adj[t] = smax_t

        rng = smax_t - smin_t
        if np.isfinite(rng):
            bsmax_t = smax_t + 0.5 * (money_risk - 1.0) * rng
            bsmin_t = smin_t - 0.5 * (money_risk - 1.0) * rng
            if tr > 0 and np.isfinite(bsmin[t - 1]):
                bsmin_t = np.maximum(bsmin_t, bsmin[t - 1])
            if tr < 0 and np.isfinite(bsmax[t - 1]):
                bsmax_t = np.minimum(bsmax_t, bsmax[t - 1])
            bsmax[t] = bsmax_t
            bsmin[t] = bsmin_t

        trend[t] = tr

        if tr > 0 and np.isfinite(bsmin[t]):
            # Up trend
            prev_up_inactive = t > 0 and up_stop[t - 1] == -1.0
            if signal_mode > 0 and prev_up_inactive:
                up_signal[t] = bsmin[t]
                up_stop[t] = bsmin[t]
                if line_mode > 0:
                    up_line[t] = bsmin[t]
            else:
                up_stop[t] = bsmin[t]
                if line_mode > 0:
                    up_line[t] = bsmin[t]
                up_signal[t] = -1.0

            if signal_mode == 2:
                up_stop[t] = 0.0

            down_signal[t] = -1.0
            down_stop[t] = -1.0
            down_line[t] = np.nan

        elif tr < 0 and np.isfinite(bsmax[t]):
            # Down trend
            prev_down_inactive = t > 0 and down_stop[t - 1] == -1.0
            if signal_mode > 0 and prev_down_inactive:
                down_signal[t] = bsmax[t]
                down_stop[t] = bsmax[t]
                if line_mode > 0:
                    down_line[t] = bsmax[t]
            else:
                down_stop[t] = bsmax[t]
                if line_mode > 0:
                    down_line[t] = bsmax[t]
                down_signal[t] = -1.0

            if signal_mode == 2:
                down_stop[t] = 0.0

            up_signal[t] = -1.0
            up_stop[t] = -1.0
            up_line[t] = np.nan
        # else: trend == 0; leave NaNs (warmup or no established trend)

    out = pd.DataFrame(
        {
            "UpTrendStop": up_stop,
            "DownTrendStop": down_stop,
            "UpTrendSignal": up_signal,
            "DownTrendSignal": down_signal,
            "UpTrendLine": up_line,
            "DownTrendLine": down_line,
        },
        index=df.index,
    )

    return out


# === END BamsBung3_calc.py ===


# === BEGIN BandPassFilter_calc.py ===
def BandPassFilter(
    df: pd.DataFrame, period: int = 50, price: str = "median", delta: float = 0.1
) -> pd.DataFrame:
    """
    Ehlers Band-Pass Filter with slope/trend-segmented histogram outputs.

    Parameters
    - period: int, default 50
    - price: str, one of:
        'close','open','high','low','median','medianb','typical','weighted','average',
        'tbiased','tbiased2',
        'ha_close','ha_open','ha_high','ha_low','ha_median','ha_medianb','ha_typical','ha_weighted','ha_average','ha_tbiased','ha_tbiased2',
        'hab_close','hab_open','hab_high','hab_low','hab_median','hab_medianb','hab_typical','hab_weighted','hab_average','hab_tbiased','hab_tbiased2'
      default 'median'
    - delta: float, default 0.1

    Returns
    - DataFrame with columns:
        ['BP_StrongUp','BP_WeakUp','BP_StrongDown','BP_WeakDown','BP']
      aligned to df.index.
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(df)

    period = int(max(1, period))
    delta = float(delta)

    prices = _compute_price(o, h, l, c, price)

    # Constants
    beta = np.cos(2.0 * np.pi / period)
    cos_term = np.cos(4.0 * np.pi * delta / period)
    # Avoid division by zero and numerical issues
    cos_term = np.where(np.isclose(cos_term, 0.0), np.finfo(float).tiny, cos_term)
    gamma = 1.0 / cos_term
    under = np.maximum(gamma * gamma - 1.0, 0.0)
    alpha = gamma - np.sqrt(under)

    # MQL4 code operates on series arrays (index 0 is most recent).
    # Convert to series-orientation by reversing, compute, then reverse back.
    p_s = prices[::-1]

    bp_s = np.full(n, np.nan, dtype=float)
    slope_s = np.zeros(n, dtype=float)
    trend_s = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        if i >= n - 2:
            bp_s[i] = p_s[i]
        else:
            x0 = p_s[i]
            x2 = p_s[i + 2]
            b1 = bp_s[i + 1]
            b2 = bp_s[i + 2]
            if not (
                np.isnan(x0)
                or np.isnan(x2)
                or np.isnan(b1)
                or np.isnan(b2)
                or np.isnan(alpha)
                or np.isnan(beta)
            ):
                bp_s[i] = (
                    0.5 * (1.0 - alpha) * (x0 - x2)
                    + beta * (1.0 + alpha) * b1
                    - alpha * b2
                )
            else:
                bp_s[i] = np.nan

        if i == n - 1:
            slope_s[i] = 0.0
            trend_s[i] = 0.0
        else:
            # slope
            if not (np.isnan(bp_s[i]) or np.isnan(bp_s[i + 1])):
                if bp_s[i] > bp_s[i + 1]:
                    slope_s[i] = 1.0
                elif bp_s[i] < bp_s[i + 1]:
                    slope_s[i] = -1.0
                else:
                    slope_s[i] = slope_s[i + 1]
            else:
                slope_s[i] = slope_s[i + 1]
            # trend
            if not np.isnan(bp_s[i]):
                if bp_s[i] > 0.0:
                    trend_s[i] = 1.0
                elif bp_s[i] < 0.0:
                    trend_s[i] = -1.0
                else:
                    trend_s[i] = trend_s[i + 1]
            else:
                trend_s[i] = trend_s[i + 1]

    # Reverse back to chronological order
    bp = bp_s[::-1]
    slope = slope_s[::-1]
    trend = trend_s[::-1]

    # Histograms
    strong_up = np.where((trend == 1.0) & (slope == 1.0), bp, np.nan)
    weak_up = np.where((trend == 1.0) & (slope == -1.0), bp, np.nan)
    strong_down = np.where((trend == -1.0) & (slope == -1.0), bp, np.nan)
    weak_down = np.where((trend == -1.0) & (slope == 1.0), bp, np.nan)

    out = pd.DataFrame(
        {
            "BP_StrongUp": strong_up,
            "BP_WeakUp": weak_up,
            "BP_StrongDown": strong_down,
            "BP_WeakDown": weak_down,
            "BP": bp,
        },
        index=df.index,
    )

    return out


def _compute_price(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, price: str
) -> np.ndarray:
    p = (
        (price or "median")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )
    use_better = False
    is_ha = False

    base_map = {
        "close": c,
        "open": o,
        "high": h,
        "low": l,
        "median": (h + l) / 2.0,
        "medianb": (o + c) / 2.0,
        "typical": (h + l + c) / 3.0,
        "weighted": (h + l + 2.0 * c) / 4.0,
        "average": (h + l + c + o) / 4.0,
        "tbiased": np.where(c > o, (h + c) / 2.0, (l + c) / 2.0),
        "tbiased2": np.where(c > o, h, np.where(c < o, l, c)),
    }

    if p in base_map:
        return base_map[p].astype(float)

    if p.startswith("ha"):
        is_ha = True
        if p.startswith("hab"):
            use_better = True
            key = p[3:]  # after 'hab'
        else:
            key = p[2:]  # after 'ha'
        ha_o, ha_h, ha_l, ha_c = _heiken_ashi(o, h, l, c, use_better=use_better)
        if key == "close":
            return ha_c
        if key == "open":
            return ha_o
        if key == "high":
            return ha_h
        if key == "low":
            return ha_l
        if key == "median":
            return (ha_h + ha_l) / 2.0
        if key == "medianb":
            return (ha_o + ha_c) / 2.0
        if key == "typical":
            return (ha_h + ha_l + ha_c) / 3.0
        if key == "weighted":
            return (ha_h + ha_l + 2.0 * ha_c) / 4.0
        if key == "average":
            return (ha_h + ha_l + ha_c + ha_o) / 4.0
        if key == "tbiased":
            return np.where(ha_c > ha_o, (ha_h + ha_c) / 2.0, (ha_l + ha_c) / 2.0)
        if key == "tbiased2":
            return np.where(ha_c > ha_o, ha_h, np.where(ha_c < ha_o, ha_l, ha_c))

    # Fallback to median if unknown price type
    return ((h + l) / 2.0).astype(float)


def _heiken_ashi(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, use_better: bool = False
):
    n = len(o)
    ha_o = np.full(n, np.nan, dtype=float)
    ha_h = np.full(n, np.nan, dtype=float)
    ha_l = np.full(n, np.nan, dtype=float)
    ha_c = np.full(n, np.nan, dtype=float)

    for i in range(n):
        oi, hi, li, ci = o[i], h[i], l[i], c[i]
        if i == 0:
            prev_ha_o = (
                (oi + ci) / 2.0 if not (np.isnan(oi) or np.isnan(ci)) else np.nan
            )
        else:
            prev_ha_o = ha_o[i - 1]

        if (
            np.isnan(oi)
            or np.isnan(hi)
            or np.isnan(li)
            or np.isnan(ci)
            or np.isnan(prev_ha_o)
        ):
            ha_o[i] = np.nan
            ha_c[i] = np.nan
            ha_h[i] = np.nan
            ha_l[i] = np.nan
            continue

        if use_better:
            if hi != li:
                ha_c_i = (oi + ci) / 2.0 + ((ci - oi) / (hi - li)) * abs(
                    (ci - oi) / 2.0
                )
            else:
                ha_c_i = (oi + ci) / 2.0
        else:
            ha_c_i = (oi + hi + li + ci) / 4.0

        ha_o_i = (prev_ha_o + (ha_c[i - 1] if i > 0 else (oi + ci) / 2.0)) / 2.0
        # The MQL version uses (prev_ha_open + prev_ha_close) / 2.0; our first bar uses (o+c)/2 as prev close proxy
        if i > 0 and not np.isnan(ha_c[i - 1]):
            ha_o_i = (prev_ha_o + ha_c[i - 1]) / 2.0
        else:
            ha_o_i = (prev_ha_o + (oi + ci) / 2.0) / 2.0

        hi_i = max(hi, ha_o_i, ha_c_i)
        li_i = min(li, ha_o_i, ha_c_i)

        ha_o[i] = ha_o_i
        ha_c[i] = ha_c_i
        ha_h[i] = hi_i
        ha_l[i] = li_i

    return ha_o, ha_h, ha_l, ha_c


# === END BandPassFilter_calc.py ===


# === BEGIN Chandelierexit_calc.py ===
def Chandelierexit(
    df: pd.DataFrame,
    lookback: int = 7,
    atr_period: int = 9,
    atr_mult: float = 2.5,
    shift: int = 0,
) -> pd.DataFrame:
    """Return Chandelier Exit long/short lines as columns aligned to df.index.
    Parameters:
      - lookback: window for highest high / lowest low
      - atr_period: ATR period (Wilder)
      - atr_mult: ATR multiplier
      - shift: bars to offset the window/ATR into the past
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            {"Chandelier_Long": [], "Chandelier_Short": []}, index=df.index
        )

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # ATR (Wilder's smoothing via EWM alpha=1/period; NaN warmup handled by min_periods)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()

    # Shift inputs per MQL logic (use values from 'shift' bars ago)
    high_s = high.shift(shift)
    low_s = low.shift(shift)
    atr_s = atr.shift(shift)

    # Raw stop levels
    hh = high_s.rolling(window=lookback, min_periods=lookback).max()
    ll = low_s.rolling(window=lookback, min_periods=lookback).min()

    raw_up = hh - atr_s * atr_mult
    raw_dn = ll + atr_s * atr_mult

    # Allocate outputs
    long_line = np.full(n, np.nan, dtype=float)
    short_line = np.full(n, np.nan, dtype=float)
    direction = np.zeros(n, dtype=int)

    ru = raw_up.to_numpy()
    rd = raw_dn.to_numpy()
    c = close.to_numpy()

    for j in range(n):
        prev_dir = direction[j - 1] if j > 0 else 0
        prev_ru = ru[j - 1] if j > 0 else np.nan
        prev_rd = rd[j - 1] if j > 0 else np.nan

        d = prev_dir
        if not np.isnan(c[j]):
            if not np.isnan(prev_rd) and c[j] > prev_rd:
                d = 1
            if not np.isnan(prev_ru) and c[j] < prev_ru:
                d = -1
        direction[j] = d

        up = ru[j]
        dn = rd[j]

        if d > 0:
            if j > 0 and not np.isnan(prev_ru):
                if np.isnan(up) or up < prev_ru:
                    up = prev_ru
            long_line[j] = up
            short_line[j] = np.nan
        elif d < 0:
            if j > 0 and not np.isnan(prev_rd):
                if np.isnan(dn) or dn > prev_rd:
                    dn = prev_rd
            short_line[j] = dn
            long_line[j] = np.nan
        else:
            long_line[j] = np.nan
            short_line[j] = np.nan

    return pd.DataFrame(
        {"Chandelier_Long": long_line, "Chandelier_Short": short_line}, index=df.index
    )


# === END Chandelierexit_calc.py ===


# === BEGIN CMO_calc.py ===
def CMO(df: pd.DataFrame, length: int = 9, price: str = "Close") -> pd.Series:
    """
    Chande Momentum Oscillator (CMO)
    Returns a pandas Series aligned to df.index.
    Parameters:
    - length: lookback period (default 9)
    - price: applied price, one of ['Close','Open','High','Low','Median','Typical','Weighted'] (default 'Close')
    """
    if length is None or int(length) < 1:
        return pd.Series(np.nan, index=df.index, name="CMO")
    length = int(length)

    p = price.lower()
    if p == "close":
        ap = df["Close"]
    elif p == "open":
        ap = df["Open"]
    elif p == "high":
        ap = df["High"]
    elif p == "low":
        ap = df["Low"]
    elif p == "median":
        ap = (df["High"] + df["Low"]) / 2.0
    elif p == "typical":
        ap = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif p == "weighted":
        ap = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        ap = df["Close"]

    diff = ap - ap.shift(1)
    gains = diff.clip(lower=0)
    losses = (-diff).clip(lower=0)

    s1 = gains.rolling(window=length, min_periods=length).mean()
    s2 = losses.rolling(window=length, min_periods=length).mean()
    denom = s1 + s2

    cmo = np.where(denom != 0, (s1 - s2) / denom * 100.0, np.nan)
    out = pd.Series(cmo, index=df.index, name="CMO")
    return out


# === END CMO_calc.py ===


# === BEGIN Coral_calc.py ===
def Coral(df: pd.DataFrame, length: int = 34, coef: float = 0.4) -> pd.DataFrame:
    """THV Coral indicator (MT4) using pandas/numpy only.

    Returns a DataFrame with:
    - 'Coral': primary coral line
    - 'Coral_Yellow', 'Coral_RoyalBlue', 'Coral_Red': segmented lines for coloring logic (NaN where not active)

    Parameters:
    - length: base length (default 34)
    - coef: filter coefficient (default 0.4)
    """
    idx = df.index
    close = df["Close"].astype(float).to_numpy()

    # Ensure valid length
    n = max(int(length), 1)
    # MT4 code: g = (n - 1)/2 + 1; alpha = 2/(g + 1) => alpha = 4/(n + 3)
    alpha = 4.0 / (n + 3.0)

    c = float(coef)
    c2 = c * c
    c3 = c2 * c
    A0 = -c3
    A1 = 3.0 * (c2 + c3)
    A2 = -3.0 * (2.0 * c2 + c + c3)
    A3 = 3.0 * c + 1.0 + c3 + 3.0 * c2

    def _ema_zero_init(x: np.ndarray, a: float) -> np.ndarray:
        # Prepend a zero so the first output equals a*x0 (MT4 arrays start from zero-initialized state)
        z = np.empty(len(x) + 1, dtype=float)
        z[0] = 0.0
        z[1:] = x
        y = pd.Series(z).ewm(alpha=a, adjust=False).mean().to_numpy()
        return y[1:]

    # 6 cascaded EMAs with same alpha, each zero-initialized
    e1 = _ema_zero_init(close, alpha)
    e2 = _ema_zero_init(e1, alpha)
    e3 = _ema_zero_init(e2, alpha)
    e4 = _ema_zero_init(e3, alpha)
    e5 = _ema_zero_init(e4, alpha)
    e6 = _ema_zero_init(e5, alpha)

    coral = A0 * e6 + A1 * e5 + A2 * e4 + A3 * e3
    coral_s = pd.Series(coral, index=idx, name="Coral")

    # Segmented color buffers (following MT4 logic)
    yellow = coral_s.copy()
    blue = coral_s.copy()
    red = coral_s.copy()

    prev = coral_s.shift(1)
    down_mask = prev > coral_s
    up_mask = prev < coral_s
    flat_mask = ~(down_mask | up_mask)

    # MT4 logic:
    # if prev > curr: empty blue
    # else if prev < curr: empty red
    # else: empty yellow
    blue = blue.mask(down_mask, np.nan)
    red = red.mask(up_mask, np.nan)
    yellow = yellow.mask(flat_mask, np.nan)

    out = pd.DataFrame(
        {
            "Coral": coral_s,
            "Coral_Yellow": yellow,
            "Coral_RoyalBlue": blue,
            "Coral_Red": red,
        },
        index=idx,
    )

    return out


# === END Coral_calc.py ===


# === BEGIN CVIMulti_calc.py ===
def CVIMulti(
    df: pd.DataFrame, length: int = 14, method: int = 0, use_modified: bool = False
) -> pd.Series:
    """Return the Chartmill Value Indicator (CVI) as a Series aligned to df.index.

    Params:
    - length: int = 14 (lookback period)
    - method: int = 0 (0=SMA, 1=EMA, 2=SMMA/RMA, 3=LWMA)
    - use_modified: bool = False (if True, divide by ATR * sqrt(length), else by ATR)
    """
    if length <= 0:
        return pd.Series(np.nan, index=df.index, name="CVI")
    if method not in (0, 1, 2, 3):
        raise ValueError("method must be one of {0: SMA, 1: EMA, 2: SMMA, 3: LWMA}")

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    median_price = (high + low) / 2.0

    def sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=n).mean()

    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def smma(s: pd.Series, n: int) -> pd.Series:
        # Wilder's smoothing (RMA)
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    def lwma(s: pd.Series, n: int) -> pd.Series:
        # Linear Weighted MA with weights increasing to the most recent
        arr = s.to_numpy(dtype=float)
        w = np.arange(1, n + 1, dtype=float)
        m = (~np.isnan(arr)).astype(float)
        x = np.where(np.isnan(arr), 0.0, arr)

        num = np.convolve(x, w, mode="full")[n - 1 : n - 1 + arr.size]
        den = np.convolve(m, w, mode="full")[n - 1 : n - 1 + arr.size]

        full_weight = w.sum()
        out = np.where(den == full_weight, num / den, np.nan)
        return pd.Series(out, index=s.index)

    if method == 0:
        vc = sma(median_price, length)
    elif method == 1:
        vc = ema(median_price, length)
    elif method == 2:
        vc = smma(median_price, length)
    else:
        vc = lwma(median_price, length)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    denom = atr * (np.sqrt(length) if use_modified else 1.0)
    denom = denom.where(denom != 0.0, np.nan)

    cvi = (close - vc) / denom
    cvi.name = "CVI"
    return cvi


# === END CVIMulti_calc.py ===


# === BEGIN CyberCycle_calc.py ===
def CyberCycle(
    df: pd.DataFrame, alpha: float = 0.07, price: str | int = "hl2"
) -> pd.DataFrame:
    """
    Cyber Cycle (Ehlers) with Trigger.
    Returns a DataFrame with ['Cycle','Trigger'], aligned to df.index, preserving length with NaNs for warmup.

    Parameters:
    - alpha: float (default 0.07)
    - price: one of {'close','open','high','low','hl2','median','typical','hlc3','ohlc4','wclose','hlcc4'} or MT4 codes {0..6}
             0: Close, 1: Open, 2: High, 3: Low, 4: (H+L)/2, 5: (H+L+C)/3, 6: (H+L+2C)/4
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    # Map price selection
    if isinstance(price, int):
        code = price
        if code == 0:
            s = c
        elif code == 1:
            s = o
        elif code == 2:
            s = h
        elif code == 3:
            s = l
        elif code == 4:
            s = (h + l) / 2.0
        elif code == 5:
            s = (h + l + c) / 3.0
        elif code == 6:
            s = (h + l + 2.0 * c) / 4.0
        else:
            raise ValueError("Unsupported MT4 price code. Use 0..6.")
    else:
        key = str(price).lower()
        if key in ("close", "c"):
            s = c
        elif key in ("open", "o"):
            s = o
        elif key in ("high", "h"):
            s = h
        elif key in ("low", "l"):
            s = l
        elif key in ("hl2", "median"):
            s = (h + l) / 2.0
        elif key in ("typical", "hlc3"):
            s = (h + l + c) / 3.0
        elif key in ("ohlc4",):
            s = (o + h + l + c) / 4.0
        elif key in ("wclose", "hlcc4", "weighted"):
            s = (h + l + 2.0 * c) / 4.0
        else:
            raise ValueError("Unsupported price string.")

    # Smooth = (Price + 2*Price[1] + 2*Price[2] + Price[3]) / 6
    smooth = (s + 2.0 * s.shift(1) + 2.0 * s.shift(2) + s.shift(3)) / 6.0

    # Precompute second difference of smoothed price
    d_smooth = smooth - 2.0 * smooth.shift(1) + smooth.shift(2)

    n = len(df)
    cycle = np.full(n, np.nan, dtype=float)

    s_vals = s.to_numpy(dtype=float)
    d_vals = d_smooth.to_numpy(dtype=float)

    c0 = (1.0 - 0.5 * alpha) ** 2
    b1 = 2.0 * (1.0 - alpha)
    b2 = -((1.0 - alpha) ** 2)

    # Seed first bars as per Ehlers: for currentbar < 7 use raw price second difference / 4
    # Cycle_t = (Price_t - 2*Price_{t-1} + Price_{t-2}) / 4
    for t in range(n):
        if t < 7:
            if (
                t >= 2
                and np.isfinite(s_vals[t])
                and np.isfinite(s_vals[t - 1])
                and np.isfinite(s_vals[t - 2])
            ):
                cycle[t] = (s_vals[t] - 2.0 * s_vals[t - 1] + s_vals[t - 2]) / 4.0
            else:
                cycle[t] = np.nan
        else:
            # Full recursive filter
            if (
                t >= 2
                and np.isfinite(d_vals[t])
                and np.isfinite(cycle[t - 1])
                and np.isfinite(cycle[t - 2])
            ):
                cycle[t] = c0 * d_vals[t] + b1 * cycle[t - 1] + b2 * cycle[t - 2]
            else:
                cycle[t] = np.nan

    cycle_s = pd.Series(cycle, index=df.index, name="Cycle")
    trigger_s = cycle_s.shift(1).rename("Trigger")

    return pd.DataFrame({"Cycle": cycle_s, "Trigger": trigger_s})


# === END CyberCycle_calc.py ===


# === BEGIN DecyclerOscillator_calc.py ===
def DecyclerOscillator(
    df: pd.DataFrame,
    hp_period: int = 125,
    k: float = 1.0,
    hp_period2: int = 100,
    k2: float = 1.2,
    price: str = "close",
) -> pd.DataFrame:
    """
    Simple Decycler Oscillator x 2.
    Returns a DataFrame with columns:
      - DEO: 100*k*HP( price - HP(price, hp_period), hp_period) / price
      - DEO2: 100*k2*HP( price - HP(price, hp_period2), hp_period2) / price
      - DEO2da, DEO2db: segmented DEO2 values for down-trend visualization (alternating segments)
    All outputs aligned to df.index; NaNs for warmup/undefined.

    Parameters:
      hp_period: int >= 2 (slow high-pass period)
      k: multiplier for DEO
      hp_period2: int >= 2 (fast high-pass period)
      k2: multiplier for DEO2
      price: one of
        close, open, high, low, median, typical, weighted, average, medianb, tbiased,
        haclose, haopen, hahigh, halow, hamedian, hatypical, haweighted, haaverage, hamedianb, hatbiased
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(df)

    def _heiken_ashi(o, h, l, c):
        ha_close = (o + h + l + c) / 4.0
        ha_open = np.empty_like(ha_close)
        ha_open[:] = np.nan
        # Init
        ha_open[0] = (o[0] + c[0]) / 2.0
        # Recursive
        for i in range(1, len(o)):
            if np.isnan(ha_close[i - 1]) or np.isnan(ha_open[i - 1]):
                ha_open[i] = np.nan
            else:
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high = np.nanmax(np.vstack([h, ha_open, ha_close]), axis=0)
        ha_low = np.nanmin(np.vstack([l, ha_open, ha_close]), axis=0)
        return ha_open, ha_high, ha_low, ha_close

    def _select_price(kind: str) -> np.ndarray:
        k = kind.lower()
        if k == "close":
            return c
        if k == "open":
            return o
        if k == "high":
            return h
        if k == "low":
            return l
        if k == "median":
            return (h + l) / 2.0
        if k == "medianb":
            return (o + c) / 2.0
        if k == "typical":
            return (h + l + c) / 3.0
        if k == "weighted":
            return (h + l + 2.0 * c) / 4.0
        if k == "average":
            return (h + l + c + o) / 4.0
        if k == "tbiased":
            out = np.where(c > o, (h + c) / 2.0, (l + c) / 2.0)
            out[np.isnan(c) | np.isnan(o) | np.isnan(h) | np.isnan(l)] = np.nan
            return out
        # Heiken Ashi derived prices
        if k.startswith("ha"):
            ha_open, ha_high, ha_low, ha_close = _heiken_ashi(o, h, l, c)
            if k == "haclose":
                return ha_close
            if k == "haopen":
                return ha_open
            if k == "hahigh":
                return ha_high
            if k == "halow":
                return ha_low
            if k == "hamedian":
                return (ha_high + ha_low) / 2.0
            if k == "hamedianb":
                return (ha_open + ha_close) / 2.0
            if k == "hatypical":
                return (ha_high + ha_low + ha_close) / 3.0
            if k == "haweighted":
                return (ha_high + ha_low + 2.0 * ha_close) / 4.0
            if k == "haaverage":
                return (ha_high + ha_low + ha_close + ha_open) / 4.0
            if k == "hatbiased":
                out = np.where(
                    ha_close > ha_open,
                    (ha_high + ha_close) / 2.0,
                    (ha_low + ha_close) / 2.0,
                )
                out[
                    np.isnan(ha_close)
                    | np.isnan(ha_open)
                    | np.isnan(ha_high)
                    | np.isnan(ha_low)
                ] = np.nan
                return out
        raise ValueError(f"Unsupported price type: {kind}")

    def _hp_2pole(x: np.ndarray, period: float) -> np.ndarray:
        # MQL iHp implementation
        out = np.empty_like(x)
        out[:] = np.nan
        if period is None or period <= 1 or len(x) == 0:
            # MQL returns 0 for all i when period<=1
            out = np.zeros_like(x, dtype=float)
            return out
        angle = 0.707 * 2.0 * np.pi / float(period)
        ca = np.cos(angle)
        sa = np.sin(angle)
        alpha = (ca + sa - 1.0) / ca
        a = (1.0 - alpha / 2.0) ** 2
        b = 2.0 * (1.0 - alpha)
        c2 = (1.0 - alpha) ** 2
        # Initialize first three values to 0 as per MQL (i<=2 -> 0)
        for i in range(len(x)):
            if i <= 2:
                out[i] = 0.0 if not np.isnan(x[i]) else np.nan
            else:
                p0, p1, p2 = x[i], x[i - 1], x[i - 2]
                y1, y2 = out[i - 1], out[i - 2]
                if (
                    np.isnan(p0)
                    or np.isnan(p1)
                    or np.isnan(p2)
                    or np.isnan(y1)
                    or np.isnan(y2)
                ):
                    out[i] = np.nan
                else:
                    out[i] = a * (p0 - 2.0 * p1 + p2) + b * y1 - c2 * y2
        return out

    px = _select_price(price)

    # DEO
    hp1_inner = _hp_2pole(px, hp_period)
    deo_num = _hp_2pole(px - hp1_inner, hp_period)
    deo = np.full(n, np.nan, dtype=float)
    np.divide(100.0 * k * deo_num, px, out=deo, where=(px != 0.0))

    # DEO2
    hp2_inner = _hp_2pole(px, hp_period2)
    deo2_num = _hp_2pole(px - hp2_inner, hp_period2)
    deo2 = np.full(n, np.nan, dtype=float)
    np.divide(100.0 * k2 * deo2_num, px, out=deo2, where=(px != 0.0))

    # Trend: sign(deo2 - deo) with tie -> carry forward previous
    diff = deo2 - deo
    trend = np.empty(n, dtype=float)
    trend[:] = np.nan
    for i in range(n):
        d = diff[i]
        if np.isnan(d):
            trend[i] = trend[i - 1] if i > 0 else np.nan
        else:
            if d > 0:
                trend[i] = 1.0
            elif d < 0:
                trend[i] = -1.0
            else:
                trend[i] = trend[i - 1] if i > 0 else np.nan

    # Segment down-trend (-1) into alternating groups -> two buffers
    idx = df.index
    trend_s = pd.Series(trend, index=idx)
    down_mask = trend_s.eq(-1.0)

    starts = down_mask & ~down_mask.shift(1, fill_value=False)
    group = starts.cumsum().where(down_mask, 0)  # 0 outside down segments
    # Alternate buffers by group parity
    grp_vals = group.to_numpy()
    deo2da = np.where((grp_vals > 0) & ((grp_vals % 2) == 1), deo2, np.nan)
    deo2db = np.where((grp_vals > 0) & ((grp_vals % 2) == 0), deo2, np.nan)

    out = pd.DataFrame(
        {
            "DEO": deo,
            "DEO2": deo2,
            "DEO2da": deo2da,
            "DEO2db": deo2db,
        },
        index=df.index,
    )
    return out


# === END DecyclerOscillator_calc.py ===


# === BEGIN DetrendedSyntheticPriceGoscillators_calc.py ===
def DetrendedSyntheticPriceGoscillators(
    df: pd.DataFrame,
    dsp_period: int = 14,
    price_mode: str = "median",
    signal_period: int = 9,
    color_on: str = "outer",
) -> pd.DataFrame:
    """
    DSP oscillator (Mladen): returns all buffers:
      - DSP_LevelUp, DSP_LevelDown (outer EMAs gated by sign)
      - DSP (main line = EMA(alpha_m) - EMA(alpha_m/2))
      - DSP_Up_A, DSP_Up_B (alternating up-color segments)
      - DSP_Down_A, DSP_Down_B (alternating down-color segments)
      - DSP_State (int: -1,0,1)

    Parameters:
      - dsp_period: int, default 14
      - price_mode: one of
          close, open, high, low, median, typical, weighted, average, medianb,
          tbiased, tbiased2,
          haclose, haopen, hahigh, halow, hamedian, hatypical, haweighted,
          haaverage, hamedianb, hatbiased, hatbiased2
        (case-insensitive; 'pr_' prefixes accepted)
      - signal_period: int, default 9
      - color_on: one of outer, outer2, zero, slope (case-insensitive; 'chg_on' prefixes accepted)
    """
    idx = df.index
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "DSP_LevelUp",
                "DSP_LevelDown",
                "DSP",
                "DSP_Up_A",
                "DSP_Up_B",
                "DSP_Down_A",
                "DSP_Down_B",
                "DSP_State",
            ],
            index=idx,
        )

    O = df["Open"].astype(float).to_numpy(copy=False)
    H = df["High"].astype(float).to_numpy(copy=False)
    L = df["Low"].astype(float).to_numpy(copy=False)
    C = df["Close"].astype(float).to_numpy(copy=False)

    # Normalize params
    pm = price_mode.strip().lower()
    if pm.startswith("pr_"):
        pm = pm[3:]
    co = color_on.strip().lower()
    if co.startswith("chg_on"):
        co = co[6:] if color_on.lower().startswith("chg_on") else co

    # Heiken Ashi series (needs recursion for ha_open)
    ha_close = (O + H + L + C) / 4.0
    ha_open = np.full(n, np.nan)
    # initial ha_open as (open+close)/2 if available
    if n > 0:
        if np.isfinite(O[0]) and np.isfinite(C[0]):
            ha_open[0] = (O[0] + C[0]) / 2.0
    for t in range(1, n):
        if np.isfinite(ha_open[t - 1]) and np.isfinite(ha_close[t - 1]):
            ha_open[t] = 0.5 * (ha_open[t - 1] + ha_close[t - 1])
        else:
            # restart if direct values available, else remain NaN
            if np.isfinite(O[t]) and np.isfinite(C[t]):
                ha_open[t] = (O[t] + C[t]) / 2.0
    ha_high = np.maximum.reduce([H, ha_open, ha_close])
    ha_low = np.minimum.reduce([L, ha_open, ha_close])

    # Price selection
    def select_price(mode: str) -> np.ndarray:
        if mode == "close":
            return C
        if mode == "open":
            return O
        if mode == "high":
            return H
        if mode == "low":
            return L
        if mode == "median":
            return (H + L) / 2.0
        if mode == "medianb":
            return (O + C) / 2.0
        if mode == "typical":
            return (H + L + C) / 3.0
        if mode == "weighted":
            return (H + L + 2.0 * C) / 4.0
        if mode == "average":
            return (H + L + C + O) / 4.0
        if mode == "tbiased":
            return np.where(C > O, (H + C) / 2.0, (L + C) / 2.0)
        if mode == "tbiased2":
            return np.where(C > O, H, np.where(C < O, L, C))
        # Heiken Ashi derived
        if mode == "haclose":
            return ha_close
        if mode == "haopen":
            return ha_open
        if mode == "hahigh":
            return ha_high
        if mode == "halow":
            return ha_low
        if mode == "hamedian":
            return (ha_high + ha_low) / 2.0
        if mode == "hamedianb":
            return (ha_open + ha_close) / 2.0
        if mode == "hatypical":
            return (ha_high + ha_low + ha_close) / 3.0
        if mode == "haweighted":
            return (ha_high + ha_low + 2.0 * ha_close) / 4.0
        if mode == "haaverage":
            return (ha_high + ha_low + ha_close + ha_open) / 4.0
        if mode == "hatbiased":
            return np.where(
                ha_close > ha_open,
                (ha_high + ha_close) / 2.0,
                (ha_low + ha_close) / 2.0,
            )
        if mode == "hatbiased2":
            return np.where(
                ha_close > ha_open,
                ha_high,
                np.where(ha_close < ha_open, ha_low, ha_close),
            )
        raise ValueError(f"Unsupported price_mode: {price_mode}")

    px = select_price(pm)
    px_s = pd.Series(px, index=idx)

    # EMAs with specified alphas (ignore NaNs)
    alpha_m = 2.0 / (1.0 + float(dsp_period))
    ema_fast = px_s.ewm(alpha=alpha_m, adjust=False, ignore_na=True).mean()
    ema_slow = px_s.ewm(alpha=alpha_m / 2.0, adjust=False, ignore_na=True).mean()

    dsp = ema_fast - ema_slow  # main line
    v = dsp.to_numpy(copy=False)

    # Gated EMA levels
    alpha_s = 2.0 / (1.0 + float(signal_period))
    levelu = np.full(n, np.nan)
    leveld = np.full(n, np.nan)

    pu = 0.0
    pdn = 0.0
    for t in range(n):
        vt = v[t]
        if np.isfinite(vt):
            if vt > 0.0:
                pu = pu + alpha_s * (vt - pu)
            # else keep pu
            if vt < 0.0:
                pdn = pdn + alpha_s * (vt - pdn)
            # else keep pdn
        # if NaN, keep previous values
        levelu[t] = pu
        leveld[t] = pdn

    # State calculation
    state = np.zeros(n, dtype=int)
    if co == "outer":
        for t in range(n):
            vt = v[t]
            if np.isfinite(vt):
                if vt > levelu[t]:
                    state[t] = 1
                elif vt < leveld[t]:
                    state[t] = -1
                else:
                    state[t] = 0
            else:
                state[t] = 0
    elif co == "outer2":
        prev = 0
        for t in range(n):
            vt = v[t]
            cur = prev
            if np.isfinite(vt):
                if vt > levelu[t]:
                    cur = 1
                elif vt < leveld[t]:
                    cur = -1
            state[t] = cur
            prev = cur
    elif co == "zero":
        for t in range(n):
            vt = v[t]
            if np.isfinite(vt):
                state[t] = 1 if vt > 0 else (-1 if vt < 0 else 0)
            else:
                state[t] = 0
    elif co == "slope":
        prev = 0
        prev_v = np.nan
        for t in range(n):
            vt = v[t]
            cur = prev
            if np.isfinite(vt) and np.isfinite(prev_v):
                if vt > prev_v:
                    cur = 1
                elif vt < prev_v:
                    cur = -1
            state[t] = cur
            prev = cur
            prev_v = vt if np.isfinite(vt) else prev_v
    else:
        raise ValueError(f"Unsupported color_on: {color_on}")

    # Build alternating segment buffers for up and down states
    def alternating_segments(values: np.ndarray, mask: np.ndarray):
        # Identify starts of True runs
        prev = np.concatenate(([False], mask[:-1]))
        starts = mask & (~prev)
        seg_id = np.cumsum(starts.astype(np.int64))
        parity = seg_id % 2  # 1,0,1,0,...
        a_mask = mask & (parity == 0)
        b_mask = mask & (parity == 1)
        a = np.where(a_mask, values, np.nan)
        b = np.where(b_mask, values, np.nan)
        return a, b

    mask_up = state == 1
    mask_dn = state == -1

    up_a, up_b = alternating_segments(v, mask_up)
    dn_a, dn_b = alternating_segments(v, mask_dn)

    out = pd.DataFrame(
        {
            "DSP_LevelUp": pd.Series(levelu, index=idx, dtype=float),
            "DSP_LevelDown": pd.Series(leveld, index=idx, dtype=float),
            "DSP": dsp,
            "DSP_Up_A": pd.Series(up_a, index=idx, dtype=float),
            "DSP_Up_B": pd.Series(up_b, index=idx, dtype=float),
            "DSP_Down_A": pd.Series(dn_a, index=idx, dtype=float),
            "DSP_Down_B": pd.Series(dn_b, index=idx, dtype=float),
            "DSP_State": pd.Series(state, index=idx, dtype=int),
        },
        index=idx,
    )
    return out


# === END DetrendedSyntheticPriceGoscillators_calc.py ===


# === BEGIN DodaStochasticModified_calc.py ===
def DodaStochasticModified(
    df: pd.DataFrame, Slw: int = 8, Pds: int = 13, Slwsignal: int = 9
) -> pd.DataFrame:
    """
    Doda-Stochastic (modified) based on MT4 source:
    - EmaClose: EMA(Close, Slw)
    - StochOfEma: 0-100 stochastic of EmaClose over Pds window
    - DodaStoch: EMA(StochOfEma, Slw)
    - DodaSignal: EMA(DodaStoch, Slwsignal)

    Returns a DataFrame with all lines aligned to df.index and NaNs during warmup.
    """
    Slw = max(int(Slw), 1)
    Pds = max(int(Pds), 1)
    Slwsignal = max(int(Slwsignal), 1)

    close = df["Close"].astype(float)

    ema_close = close.ewm(span=Slw, adjust=False, min_periods=Slw).mean()

    roll_min = ema_close.rolling(window=Pds, min_periods=Pds).min()
    roll_max = ema_close.rolling(window=Pds, min_periods=Pds).max()
    denom = roll_max - roll_min

    mask = denom != 0
    stoch_of_ema = pd.Series(np.nan, index=df.index, dtype=float)
    stoch_of_ema.loc[mask] = (
        100.0 * (ema_close.loc[mask] - roll_min.loc[mask]) / denom.loc[mask]
    )

    doda_stoch = stoch_of_ema.ewm(span=Slw, adjust=False, min_periods=Slw).mean()
    doda_signal = doda_stoch.ewm(
        span=Slwsignal, adjust=False, min_periods=Slwsignal
    ).mean()

    out = pd.DataFrame(
        {
            "DodaStoch": doda_stoch,
            "DodaSignal": doda_signal,
            "EmaClose": ema_close,
            "StochOfEma": stoch_of_ema,
        },
        index=df.index,
    )
    return out


# === END DodaStochasticModified_calc.py ===


# === BEGIN DorseyInertia_calc.py ===
def DorseyInertia(
    df: pd.DataFrame,
    rvi_period: int = 10,
    avg_period: int = 14,
    smoothing_period: int = 20,
) -> pd.Series:
    """Dorsey Inertia indicator (Mladen implementation).

    Parameters
    - rvi_period: int = 10, period for rolling std of High/Low (population std, SMA-based)
    - avg_period: int = 14, EMA period (alpha=1/avg_period) for averaging up/down std components
    - smoothing_period: int = 20, final SMA period of the combined RVI

    Returns
    - pd.Series named 'Inertia', aligned to df.index with NaNs for warmup
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Rolling standard deviations (population, SMA-based)
    std_high = high.rolling(window=int(rvi_period), min_periods=int(rvi_period)).std(
        ddof=0
    )
    std_low = low.rolling(window=int(rvi_period), min_periods=int(rvi_period)).std(
        ddof=0
    )

    # Up/Down moves
    dh = high.diff()
    dl = low.diff()
    up_h = dh > 0
    dn_h = dh < 0
    up_l = dl > 0
    dn_l = dl < 0

    # Build u/d components with NaNs prior to sufficient std history
    stdh_vals = std_high.values
    stdl_vals = std_low.values
    valid_h = ~np.isnan(stdh_vals)
    valid_l = ~np.isnan(stdl_vals)

    u_high = np.where(
        valid_h, np.where(up_h.fillna(False).values, stdh_vals, 0.0), np.nan
    )
    d_high = np.where(
        valid_h, np.where(dn_h.fillna(False).values, stdh_vals, 0.0), np.nan
    )
    u_low = np.where(
        valid_l, np.where(up_l.fillna(False).values, stdl_vals, 0.0), np.nan
    )
    d_low = np.where(
        valid_l, np.where(dn_l.fillna(False).values, stdl_vals, 0.0), np.nan
    )

    u_high = pd.Series(u_high, index=df.index)
    d_high = pd.Series(d_high, index=df.index)
    u_low = pd.Series(u_low, index=df.index)
    d_low = pd.Series(d_low, index=df.index)

    # EMA smoothing with alpha=1/avg_period (matches ((p-1)*prev + x)/p)
    alpha = 1.0 / float(avg_period)
    HUp = u_high.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    HDo = d_high.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    LUp = u_low.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    LDo = d_low.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

    # RVI components (guard division by zero; preserve NaNs for warmup)
    denom_h = HUp + HDo
    denom_l = LUp + LDo
    rvih = 100.0 * HUp / denom_h
    rvil = 100.0 * LUp / denom_l
    rvih = rvih.mask(denom_h == 0.0, 0.0)
    rvil = rvil.mask(denom_l == 0.0, 0.0)

    rvi = (rvih + rvil) / 2.0

    # Final SMA smoothing
    inertia = rvi.rolling(
        window=int(smoothing_period), min_periods=int(smoothing_period)
    ).mean()
    return inertia.rename("Inertia")


# === END DorseyInertia_calc.py ===


# === BEGIN DpoHistogramIndicator_calc.py ===
def DpoHistogramIndicator(
    df: pd.DataFrame, period: int = 14, ma: str = "sma"
) -> pd.DataFrame:
    """Detrended Price Oscillator with up/down histograms.
    Returns DataFrame with columns ['DPO_Up','DPO_Dn','DPO'] aligned to df.index.

    Parameters:
    - period: lookback length for the moving average (default 14).
    - ma: moving average type: 'sma' (default), 'ema', 'smma' (Wilder/RMA), or 'wma' (LWMA).
    """
    if period is None or period < 1:
        raise ValueError("period must be a positive integer")

    close = pd.to_numeric(df["Close"], errors="coerce")

    ma_lower = str(ma).lower()
    if ma_lower == "sma":
        ma_series = close.rolling(window=period, min_periods=period).mean()
    elif ma_lower == "ema":
        ma_series = close.ewm(span=period, adjust=False, min_periods=period).mean()
    elif ma_lower in ("smma", "rma", "wilder"):
        # Wilder's smoothing (SMMA/RMA) is EMA with alpha = 1/period
        ma_series = close.ewm(
            alpha=1.0 / period, adjust=False, min_periods=period
        ).mean()
    elif ma_lower in ("wma", "lwma"):
        weights = np.arange(1, period + 1, dtype=float)
        w_sum = weights.sum()
        ma_series = close.rolling(window=period, min_periods=period).apply(
            lambda x: np.dot(x, weights) / w_sum, raw=True
        )
    else:
        raise ValueError(
            "ma must be one of: 'sma', 'ema', 'smma'/'rma'/'wilder', 'wma'/'lwma'"
        )

    # DPO uses MA shifted forward by t_prd bars (iMA ma_shift = period//2 + 1)
    t_prd = period // 2 + 1
    shifted_ma = ma_series.shift(-t_prd)
    dpo = close - shifted_ma

    # State: 1 if dpo>0, -1 if dpo<0, zeros inherit previous state; NaNs remain NaN then filled with 0
    valc = pd.Series(np.sign(dpo.values), index=df.index)
    zero_mask = valc.eq(0)
    valc = valc.mask(zero_mask, np.nan).ffill().fillna(0.0)

    dpo_up = dpo.where(valc.eq(1.0))
    dpo_dn = dpo.where(valc.eq(-1.0))

    out = pd.DataFrame(
        {
            "DPO_Up": dpo_up.astype(float),
            "DPO_Dn": dpo_dn.astype(float),
            "DPO": dpo.astype(float),
        },
        index=df.index,
    )
    return out


# === END DpoHistogramIndicator_calc.py ===


# === BEGIN EhlersDELIDetrendedLeadingIndicator_calc.py ===
def EhlersDELIDetrendedLeadingIndicator(
    df: pd.DataFrame, period: int = 14
) -> pd.Series:
    """Return the DELI primary indicator line aligned to df.index; vectorized, handles NaNs."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Stepwise extremes per original logic
    cond_high = high > high.shift(1)
    prevhigh = pd.Series(np.where(cond_high, high, np.nan), index=df.index)
    prevhigh.iloc[0] = high.iloc[0] if high.notna().iloc[0] else np.nan
    prevhigh = prevhigh.ffill()

    cond_low = low < low.shift(1)
    prevlow = pd.Series(np.where(cond_low, low, np.nan), index=df.index)
    prevlow.iloc[0] = low.iloc[0] if low.notna().iloc[0] else np.nan
    prevlow = prevlow.ffill()

    price = (prevhigh + prevlow) / 2.0

    p = max(int(period), 1)
    alpha = 2.0 / (p + 1.0)
    alpha2 = alpha / 2.0

    ema1 = price.ewm(alpha=alpha, adjust=False).mean()
    ema2 = price.ewm(alpha=alpha2, adjust=False).mean()
    dsp = ema1 - ema2
    temp = dsp.ewm(alpha=alpha, adjust=False).mean()
    deli = dsp - temp
    deli.name = "DELI"

    return deli


# === END EhlersDELIDetrendedLeadingIndicator_calc.py ===


# === BEGIN EhlersEarlyOnsetTrend_calc.py ===
def EhlersEarlyOnsetTrend(
    df: pd.DataFrame, period: int = 20, q1: float = 0.8, q2: float = 0.4
) -> pd.DataFrame:
    """Return both Ehlers Early Onset Trend lines (for q1 and q2) as columns aligned to df.index."""
    close = df["Close"].to_numpy(dtype=float)
    n = close.shape[0]

    PI = np.pi
    angle = 0.707 * 2.0 * PI / 100.0
    alpha1 = (np.cos(angle) + np.sin(angle) - 1.0) / np.cos(angle)

    def _quotient(
        close_arr: np.ndarray, lp_period: int, k: float, alpha: float
    ) -> np.ndarray:
        a1 = np.exp(-1.414 * PI / lp_period)
        b1 = 2.0 * a1 * np.cos(1.414 * PI / lp_period)
        c3 = -a1 * a1
        c1 = 1.0 - b1 - c3

        hp = np.zeros(n, dtype=float)
        filt = np.zeros(n, dtype=float)
        pk = np.zeros(n, dtype=float)
        out = np.full(n, np.nan, dtype=float)

        decay = 0.991
        one_minus_a = 1.0 - alpha
        coef_hp0 = (1.0 - alpha / 2.0) ** 2
        coef_hp1 = 2.0 * one_minus_a
        coef_hp2 = one_minus_a**2

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

            hp[t] = (
                coef_hp0 * (c0 - 2.0 * c1c + c2)
                + coef_hp1 * hp[t - 1]
                - coef_hp2 * hp[t - 2]
            )
            filt[t] = (
                c1 * (hp[t] + hp[t - 1]) / 2.0 + b1 * filt[t - 1] + c3 * filt[t - 2]
            )
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


# === END EhlersEarlyOnsetTrend_calc.py ===


# === BEGIN EhlersReverseEMA_calc.py ===
def EhlersReverseEMA(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Parameters:
      - alpha: smoothing factor in (0,1], default 0.1.
    """
    close = df["Close"].astype(float)
    alpha = float(alpha)
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    delta = 1.0 - alpha

    # Reverse-time EMA (anti-causal): reverse, ewm, reverse back
    ema = close.iloc[::-1].ewm(alpha=alpha, adjust=False).mean().iloc[::-1]

    # Cascaded reverse components
    re = delta * ema + ema.shift(-1)  # re1
    exp = 2
    for _ in range(7):  # re2 .. re8 with exponents 2,4,8,16,32,64,128
        re = (delta**exp) * re + re.shift(-1)
        exp *= 2

    main = ema - alpha * re

    # Warmup NaNs (75 bars as per original implementation note)
    warmup = min(75, len(main))
    if warmup > 0:
        main.iloc[:warmup] = np.nan
        ema.iloc[:warmup] = np.nan

    out = pd.DataFrame({"Main": main, "EMA": ema}, index=df.index)
    return out


# === END EhlersReverseEMA_calc.py ===


# === BEGIN Ehlersroofingfiltera_calc.py ===
def EhlersRoofingFilterA(
    df: pd.DataFrame,
    hp_length: int = 80,
    lp_length: int = 40,
    arrow_distance: float = 100.0,
    point: float = 1.0,
) -> pd.DataFrame:
    """Return all lines of the Ehlers Roofing Filter A as columns aligned to df.index.
    Columns: ['rfilt','trigger','hp','up','down'].
    Vectorized where possible; recursion computed with a tight NumPy loop. NaNs at warmup.
    """
    if hp_length <= 0 or lp_length <= 0:
        raise ValueError("hp_length and lp_length must be positive integers.")

    close = df["Close"].astype(float).to_numpy()
    n = close.size

    # Constants per Ehlers formulation
    twoPiPrd = np.sqrt(0.5) * 2.0 * np.pi / float(hp_length)
    a1 = (np.cos(twoPiPrd) + np.sin(twoPiPrd) - 1.0) / np.cos(twoPiPrd)

    a2 = np.exp(-np.sqrt(2.0) * np.pi / float(lp_length))
    beta = 2.0 * a2 * np.cos(np.sqrt(2.0) * np.pi / float(lp_length))
    c2 = beta
    c3 = -a2 * a2
    c1 = 1.0 - c2 - c3

    t1 = (1.0 - (a1 / 2.0)) ** 2
    t2 = 2.0 * (1.0 - a1)
    t3 = (1.0 - a1) ** 2

    # Allocate output arrays
    hp_arr = np.full(n, np.nan, dtype=float)
    rb_arr = np.full(n, np.nan, dtype=float)  # roofing filter (rfilt)
    tb_arr = np.full(n, np.nan, dtype=float)  # trigger

    # Internal state (zero-initialized as in original)
    h1 = 0.0  # hp[i-1]
    h2 = 0.0  # hp[i-2]
    r1 = 0.0  # rb[i-1]
    r2 = 0.0  # rb[i-2]

    for i in range(n):
        if i < 2:
            # Not enough history to compute second-order recursions; keep NaNs
            continue

        # High-pass filter (second-order recursive)
        h = t1 * (close[i] - 2.0 * close[i - 1] + close[i - 2]) + t2 * h1 - t3 * h2
        hp_arr[i] = h

        # Roofing filter (low-pass on HP) and trigger
        r = c1 * ((h + h1) / 2.0) + c2 * r1 + c3 * r2
        rb_arr[i] = r
        tb_arr[i] = r2  # trigger is rb lagged by 2

        # Update states
        h2, h1 = h1, h
        r2, r1 = r1, r

    # Build Series
    idx = df.index
    s_hp = pd.Series(hp_arr, index=idx, name="hp")
    s_rb = pd.Series(rb_arr, index=idx, name="rfilt")
    s_tb = pd.Series(tb_arr, index=idx, name="trigger")

    # Arrows (signals placed one bar after the crossover, per original)
    dst = float(arrow_distance) * float(point)
    cross_up = (s_rb.shift(1) > s_tb.shift(1)) & (s_rb.shift(2) < s_tb.shift(2))
    cross_dn = (s_rb.shift(1) < s_tb.shift(1)) & (s_rb.shift(2) > s_tb.shift(2))

    up_vals = np.where(cross_up.to_numpy(), s_rb.to_numpy() - dst, np.nan)
    dn_vals = np.where(cross_dn.to_numpy(), s_rb.to_numpy() + dst, np.nan)

    s_up = pd.Series(up_vals, index=idx, name="up")
    s_dn = pd.Series(dn_vals, index=idx, name="down")

    return pd.DataFrame(
        {
            "rfilt": s_rb,
            "trigger": s_tb,
            "hp": s_hp,
            "up": s_up,
            "down": s_dn,
        },
        index=idx,
    )


# === END Ehlersroofingfiltera_calc.py ===


# === BEGIN EhlersTwoPoleSuperSmootherFilter_calc.py ===
def EhlersTwoPoleSuperSmootherFilter(
    df: pd.DataFrame, cutoff_period: int = 15
) -> pd.Series:
    """Return the Two-Pole Super Smoother Filter (Ehlers) as a Series aligned to df.index.
    - Uses Open price (as in the original MQL4 code).
    - Handles NaNs by restarting the recursion on each contiguous valid segment.
    - Preserves length with NaNs where inputs are NaN.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(
            out,
            index=df.index,
            name=f"EhlersTwoPoleSuperSmootherFilter_{cutoff_period}",
        )

    # Validate/prepare parameters
    p = float(cutoff_period)
    if not np.isfinite(p) or p <= 0:
        return pd.Series(
            out,
            index=df.index,
            name=f"EhlersTwoPoleSuperSmootherFilter_{cutoff_period}",
        )

    # Coefficients per Ehlers' formulation (using sqrt(2) instead of 1.414)
    a1 = np.exp(-np.sqrt(2.0) * np.pi / p)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / p)
    coef3 = -(a1 * a1)
    coef2 = b1
    coef1 = 1.0 - coef2 - coef3

    x = df["Open"].to_numpy(dtype=float, copy=False)
    valid = np.isfinite(x)

    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        # Find end of this contiguous valid segment
        j = i
        while j < n and valid[j]:
            j += 1
        seg_len = j - i

        # Warmup: first three values of the segment equal to price (as in original MQL)
        if seg_len >= 1:
            out[i] = x[i]
        if seg_len >= 2:
            out[i + 1] = x[i + 1]
        if seg_len >= 3:
            out[i + 2] = x[i + 2]

        # Recurrence from the 4th element of the segment onward
        for k in range(i + 3, j):
            out[k] = coef1 * x[k] + coef2 * out[k - 1] + coef3 * out[k - 2]

        i = j

    return pd.Series(
        out, index=df.index, name=f"EhlersTwoPoleSuperSmootherFilter_{cutoff_period}"
    )


# === END EhlersTwoPoleSuperSmootherFilter_calc.py ===


# === BEGIN ErgodicTVI_calc.py ===
def ErgodicTVI(
    df: pd.DataFrame,
    Period1: int = 12,
    Period2: int = 12,
    Period3: int = 1,
    EPeriod1: int = 5,
    EPeriod2: int = 5,
    EPeriod3: int = 5,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Return ETVI and Signal lines as columns aligned to df.index.
    Uses MQL4-style EMA (seeded with SMA of first 'period' values). Vectorized; preserves NaNs for warmup.
    """

    def _ema_mql(s: pd.Series, period: int) -> pd.Series:
        s = s.astype(float)
        if period is None or period <= 1:
            return s.copy()
        sma = s.rolling(window=period, min_periods=period).mean()
        arr = s.to_numpy(dtype=float)
        sma_arr = sma.to_numpy(dtype=float)
        idxs = np.flatnonzero(~np.isnan(sma_arr))
        if idxs.size == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        first = idxs[0]
        arr2 = arr.copy()
        arr2[:first] = np.nan
        arr2[first] = sma_arr[first]
        ema = (
            pd.Series(arr2, index=s.index)
            .ewm(span=period, adjust=False, min_periods=1)
            .mean()
        )
        return ema

    if df.empty:
        return pd.DataFrame(
            {"ETVI": pd.Series(dtype=float), "Signal": pd.Series(dtype=float)},
            index=df.index,
        )

    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    upticks = (v + (c - o) / float(pip_size)) / 2.0
    dnticks = v - upticks

    ema_up1 = _ema_mql(upticks, Period1)
    ema_dn1 = _ema_mql(dnticks, Period1)
    ema_up2 = _ema_mql(ema_up1, Period2)
    ema_dn2 = _ema_mql(ema_dn1, Period2)

    denom = ema_up2 + ema_dn2
    tv = 100.0 * (ema_up2 - ema_dn2) / denom
    tv = tv.mask(denom == 0.0, 0.0)

    tvi = _ema_mql(tv, Period3)
    ema_tvi1 = _ema_mql(tvi, EPeriod1)
    etvi = _ema_mql(ema_tvi1, EPeriod2)
    signal = _ema_mql(etvi, EPeriod3)

    return pd.DataFrame({"ETVI": etvi, "Signal": signal}, index=df.index)


# === END ErgodicTVI_calc.py ===


# === BEGIN Forecast_calc.py ===
def Forecast(df: pd.DataFrame, length: int = 20, price: int = 0) -> pd.Series:
    """Return the Forecast Oscillator (FO) as a percent: 100*(Price - TSF)/Price.
    TSF is computed from a rolling linear regression on the previous `length` bars (excluding current),
    following the original MQL4 implementation (tsf = a + b)."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if price not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError("price must be in {0,1,2,3,4,5,6}")

    # Applied price mapping (MQL4 PRICE_*)
    if price == 0:
        s = df["Close"].astype(float)
    elif price == 1:
        s = df["Open"].astype(float)
    elif price == 2:
        s = df["High"].astype(float)
    elif price == 3:
        s = df["Low"].astype(float)
    elif price == 4:  # Median
        s = ((df["High"] + df["Low"]) / 2.0).astype(float)
    elif price == 5:  # Typical
        s = ((df["High"] + df["Low"] + df["Close"]) / 3.0).astype(float)
    else:  # price == 6, Weighted (HLCC/4)
        s = ((df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0).astype(float)

    N = len(s)
    L = int(length)

    # Previous L values (exclude current bar)
    y = s.shift(1).to_numpy()

    # Prepare cumulative sums for efficient rolling weighted sums
    idx = np.arange(1, N + 1, dtype=float)  # 1-based positional index
    y0 = np.where(np.isnan(y), 0.0, y)

    cs_y = np.cumsum(y0)
    cs_i_y = np.cumsum(y0 * idx)
    cs_cnt = np.cumsum((~np.isnan(y)).astype(int))

    # Pad with leading zero for windowed differences
    cs_y = np.concatenate(([0.0], cs_y))
    cs_i_y = np.concatenate(([0.0], cs_i_y))
    cs_cnt = np.concatenate(([0], cs_cnt))

    sum_y = np.full(N, np.nan)
    sum_i_y = np.full(N, np.nan)
    cnt = np.zeros(N, dtype=float)

    valid_idx = np.arange(N) >= (L - 1)
    if valid_idx.any():
        t1 = np.where(valid_idx)[0] + 1  # 1-based end index of window
        t0 = t1 - L

        sum_y[valid_idx] = cs_y[t1] - cs_y[t0]
        sum_i_y[valid_idx] = cs_i_y[t1] - cs_i_y[t0]
        cnt[valid_idx] = cs_cnt[t1] - cs_cnt[t0]

    # Compute sxy with weights 1..L for the last L values (most recent has weight 1)
    sxy = np.full(N, np.nan)
    if valid_idx.any():
        sxy[valid_idx] = t1.astype(float) * sum_y[valid_idx] - sum_i_y[valid_idx]

    # Constants for x = 1..L
    sx = L * (L + 1) / 2.0
    sx2 = L * (L + 1) * (2 * L + 1) / 6.0
    den = L * sx2 - sx * sx

    # Linear regression parameters and TSF=a+b (per original MQL4 code)
    if L == 1 or den == 0.0:
        # With a single point, use last value as forecast
        tsf = sum_y  # equals the last y in the window
    else:
        b = (L * sxy - sx * sum_y) / den
        a = (sum_y - b * sx) / L
        tsf = a + b

    pr = s.to_numpy()
    fo = np.full(N, np.nan)
    mask_full = (cnt == L) & np.isfinite(tsf) & np.isfinite(pr) & (pr != 0.0)
    fo[mask_full] = 100.0 * (pr[mask_full] - tsf[mask_full]) / pr[mask_full]

    return pd.Series(fo, index=df.index, name="FO")


# === END Forecast_calc.py ===


# === BEGIN FramaIndicator_calc.py ===
def FramaIndicator(
    df: pd.DataFrame, period: int = 10, price_type: int = 0
) -> pd.Series:
    """Return FRAMA (Fractal Adaptive Moving Average) as a Series aligned to df.index.
    - period: window length (MQL4 PeriodFRAMA), default 10
    - price_type: 0 Close, 1 Open, 2 High, 3 Low, 4 Median (H+L)/2, 5 Typical (H+L+C)/3, 6 Weighted (H+L+2C)/4
    Vectorized rolling calculations; recursive smoothing requires a minimal loop. NaNs preserved for warmup.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer")

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)

    if price_type == 1:
        price = open_.copy()
    elif price_type == 2:
        price = high.copy()
    elif price_type == 3:
        price = low.copy()
    elif price_type == 4:
        price = (high + low) / 2.0
    elif price_type == 5:
        price = (high + low + close) / 3.0
    elif price_type == 6:
        price = (high + low + 2.0 * close) / 4.0
    else:
        price = close.copy()

    p = int(period)
    two_p = 2 * p

    # Rolling highs/lows for the three segments
    hi1 = high.rolling(window=p, min_periods=p).max()
    lo1 = low.rolling(window=p, min_periods=p).min()

    # Previous block (shift by period to end at t - p)
    hi2 = hi1.shift(p)
    lo2 = lo1.shift(p)

    # Two-period block
    hi3 = high.rolling(window=two_p, min_periods=two_p).max()
    lo3 = low.rolling(window=two_p, min_periods=two_p).min()

    n1 = (hi1 - lo1) / float(p)
    n2 = (hi2 - lo2) / float(p)
    n3 = (hi3 - lo3) / float(two_p)

    # Compute fractal dimension D and adaptive alpha
    eps = np.finfo(float).eps
    log2 = np.log(2.0)

    n1n2 = (n1 + n2).to_numpy(dtype=float)
    denom = n3.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        D = (np.log(np.maximum(n1n2, eps)) - np.log(np.maximum(denom, eps))) / log2
        alpha = np.exp(-4.6 * (D - 1.0))

    pr = price.to_numpy(dtype=float)
    n = pr.shape[0]
    out = np.full(n, np.nan, dtype=float)

    # Find first valid index where alpha and price are finite
    valid_mask = np.isfinite(alpha) & np.isfinite(pr)
    if not valid_mask.any():
        return pd.Series(out, index=df.index, name="FRAMA")

    start_idx = int(np.argmax(valid_mask))  # first True

    # Seed with price at first valid point
    out[start_idx] = pr[start_idx]

    # Recursive smoothing with time-varying alpha
    for t in range(start_idx + 1, n):
        if np.isfinite(alpha[t]) and np.isfinite(pr[t]) and np.isfinite(out[t - 1]):
            a = alpha[t]
            out[t] = a * pr[t] + (1.0 - a) * out[t - 1]
        else:
            # If current inputs invalid, carry forward previous value
            out[t] = out[t - 1]

    return pd.Series(out, index=df.index, name="FRAMA")


# === END FramaIndicator_calc.py ===


# === BEGIN Gchannel_calc.py ===
def Gchannel(df: pd.DataFrame, length: int = 100, price: str = "close") -> pd.DataFrame:
    """Return all G-Channel lines as columns ['Upper','Middle','Lower'], aligned to df.index.

    Params:
    - length: int >= 1 (default 100)
    - price: one of {'close','open','high','low','median','typical','weighted'} (default 'close')

    Notes:
    - Recursive by definition; runs in O(n) with a single forward pass.
    - NaNs in source are handled by carrying forward the last computed values; leading NaNs remain NaN.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            index=df.index, columns=["Upper", "Middle", "Lower"], dtype=float
        )

    length = int(max(1, length))
    p = str(price).lower()

    c = df["Close"].to_numpy(dtype=float, copy=False)
    h = df["High"].to_numpy(dtype=float, copy=False)
    l = df["Low"].to_numpy(dtype=float, copy=False)
    o = df["Open"].to_numpy(dtype=float, copy=False)

    if p == "close":
        src = c.copy()
    elif p == "open":
        src = o.copy()
    elif p == "high":
        src = h.copy()
    elif p == "low":
        src = l.copy()
    elif p == "median":
        src = (h + l) / 2.0
    elif p == "typical":
        src = (h + l + c) / 3.0
    elif p == "weighted":
        src = (h + l + 2.0 * c) / 4.0
    else:
        raise ValueError(
            "price must be one of {'close','open','high','low','median','typical','weighted'}"
        )

    a = np.full(n, np.nan, dtype=float)
    b = np.full(n, np.nan, dtype=float)

    prev_a = 0.0
    prev_b = 0.0
    have_prev = False  # track if we've computed at least one valid step

    for i in range(n):
        x = src[i]
        if not np.isfinite(x):
            if i > 0:
                # carry forward last computed values if available
                a[i] = a[i - 1]
                b[i] = b[i - 1]
            # do not update prev_a/prev_b on NaN
            continue

        ap = prev_a if have_prev else 0.0
        bp = prev_b if have_prev else 0.0

        ai = max(x, ap) - (ap - bp) / length
        bi = min(x, bp) + (ap - bp) / length

        a[i] = ai
        b[i] = bi
        prev_a = ai
        prev_b = bi
        have_prev = True

    mid = (a + b) / 2.0

    out = pd.DataFrame(
        {
            "Upper": a,
            "Middle": mid,
            "Lower": b,
        },
        index=df.index,
    )
    return out


# === END Gchannel_calc.py ===


# === BEGIN Gd_calc.py ===
def Gd(
    df: pd.DataFrame, length: int = 20, vf: float = 0.7, price: int = 0
) -> pd.DataFrame:
    """Return ALL lines as columns ['GD','EMA'] aligned to df.index. Vectorized, handles NaNs, and preserves warmup NaNs.

    Params:
    - length: EMA period (int >= 1)
    - vf: generalization factor (float)
    - price: applied price code:
        0=Close, 1=Open, 2=High, 3=Low, 4=Median(HL2), 5=Typical(HLC3), 6=Weighted(HLCC4)
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    # Applied price selection
    if price == 0:
        src = df["Close"]
    elif price == 1:
        src = df["Open"]
    elif price == 2:
        src = df["High"]
    elif price == 3:
        src = df["Low"]
    elif price == 4:
        src = (df["High"] + df["Low"]) / 2.0
    elif price == 5:
        src = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif price == 6:
        src = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        src = df["Close"]

    alpha = 2.0 / (length + 1.0)

    ema = src.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    ema_of_ema = ema.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    gd = (1.0 + vf) * ema - vf * ema_of_ema

    out = pd.DataFrame({"GD": gd, "EMA": ema}, index=df.index)
    return out


# === END Gd_calc.py ===


# === BEGIN GeominMA_calc.py ===
def GeominMA(df: pd.DataFrame, length: int = 10, price: int = 0) -> pd.Series:
    """Geometric Mean Moving Average of the applied price.

    Params:
    - length: window size (>=1), default 10
    - price: applied price enum (as in MQL4):
        0 Close, 1 Open, 2 High, 3 Low, 4 Median(HL2), 5 Typical(HLC3), 6 Weighted(WC)

    Returns:
    - pd.Series aligned to df.index with NaNs for warmup/invalid windows.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if price not in {0, 1, 2, 3, 4, 5, 6}:
        raise ValueError("price must be one of {0,1,2,3,4,5,6}")

    # Select applied price
    if price == 0:
        ap = df["Close"].astype(float)
        price_name = "Close"
    elif price == 1:
        ap = df["Open"].astype(float)
        price_name = "Open"
    elif price == 2:
        ap = df["High"].astype(float)
        price_name = "High"
    elif price == 3:
        ap = df["Low"].astype(float)
        price_name = "Low"
    elif price == 4:
        ap = ((df["High"] + df["Low"]) / 2.0).astype(float)
        price_name = "Median"
    elif price == 5:
        ap = ((df["High"] + df["Low"] + df["Close"]) / 3.0).astype(float)
        price_name = "Typical"
    else:  # price == 6
        ap = ((df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0).astype(float)
        price_name = "Weighted"

    # Geometric mean via log to avoid overflow; require strictly positive prices in window
    ap_pos = ap.where(ap > 0.0)  # non-positive -> NaN so window becomes invalid
    log_ap = np.log(ap_pos)
    mean_log = log_ap.rolling(window=length, min_periods=length).mean()
    geom = np.exp(mean_log)
    geom.name = f"GeominMA_{length}_{price_name}"
    return geom


# === END GeominMA_calc.py ===


# === BEGIN GlitchIndexFixed_calc.py ===
def GlitchIndexFixed(
    df: pd.DataFrame,
    MaPeriod: int = 30,
    MaMethod: str = "sma",
    Price: str = "median",
    level1: float = 1.0,
    level2: float = 1.0,
) -> pd.DataFrame:
    """
    Glitch Index Fixed (vectorized)
    Returns DataFrame with columns:
      ['gliUa','gliUb','gliDb','gliDa','gliNe','gli','state']
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(df)

    # Helpers
    def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
        s = pd.Series(arr)
        return s.ewm(alpha=alpha, adjust=False).mean().to_numpy()

    def _smma(arr: np.ndarray, period: int) -> np.ndarray:
        # Wilder smoothing: y[t] = y[t-1] + (x[t] - y[t-1]) / period, seed with first non-nan
        x = arr.astype(float).copy()
        out = np.full_like(x, np.nan)
        if period <= 1:
            return x
        # find first finite
        idx = np.where(np.isfinite(x))[0]
        if idx.size == 0:
            return out
        start = idx[0]
        out[start] = x[start]
        for t in range(start + 1, len(x)):
            if np.isfinite(x[t]):
                prev = out[t - 1]
                if np.isfinite(prev):
                    out[t] = prev + (x[t] - prev) / period
                else:
                    out[t] = x[t]
            else:
                out[t] = out[t - 1]
        return out

    def _weighted_ma(
        arr: np.ndarray, weights: np.ndarray, use_mask: bool = True
    ) -> np.ndarray:
        x = arr.astype(float)
        w = np.asarray(weights, dtype=float)
        x_filled = np.where(np.isfinite(x), x, 0.0)
        mask = (
            np.isfinite(x).astype(float) if use_mask else np.ones_like(x, dtype=float)
        )
        num = np.convolve(x_filled, w, mode="full")[: len(x)]
        den = np.convolve(mask, w, mode="full")[: len(x)]
        out = np.divide(num, den, out=np.full_like(x, np.nan), where=den != 0)
        return out

    def _lwma(arr: np.ndarray, period: int) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        w = np.arange(period, 0, -1, dtype=float)  # current gets highest weight
        return _weighted_ma(arr, w, use_mask=True)

    def _slwma(arr: np.ndarray, period: int) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        sqrt_p = int(np.floor(np.sqrt(period)))
        sqrt_p = max(sqrt_p, 1)
        first = _lwma(arr, period)
        second = _lwma(first, sqrt_p)
        return second

    def _sma(arr: np.ndarray, period: int) -> np.ndarray:
        s = pd.Series(arr)
        return s.rolling(window=int(max(1, period)), min_periods=1).mean().to_numpy()

    def _tema(arr: np.ndarray, period: int) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        alpha = 2.0 / (1.0 + period)
        e1 = _ema(arr, alpha)
        e2 = _ema(e1, alpha)
        e3 = _ema(e2, alpha)
        return e3 + 3.0 * (e1 - e2)

    def _dsema(arr: np.ndarray, period: float) -> np.ndarray:
        if period <= 1:
            return arr.astype(float)
        alpha = 2.0 / (1.0 + np.sqrt(period))
        e1 = _ema(arr, alpha)
        e2 = _ema(e1, alpha)
        return e2

    def _lsma(arr: np.ndarray, period: int) -> np.ndarray:
        period = max(int(period), 1)
        lwma_avg = _lwma(arr, period)
        sma_avg = _sma(arr, period)
        out = 3.0 * lwma_avg - 2.0 * sma_avg
        # match MQL behavior: for r<period return price
        if period > 1:
            out[:period] = arr[:period]
        return out

    def _nlma(arr: np.ndarray, length: float) -> np.ndarray:
        x = arr.astype(float)
        if (length is None) or (length < 5) or (len(x) < 4):
            return x.copy()
        Cycle = 4.0
        Coeff = 3.0 * np.pi
        Phase = int(length - 1)
        L = int(length * 4) + Phase
        t = np.empty(L, dtype=float)
        for k in range(L):
            if k <= Phase - 1 and Phase > 1:
                t[k] = 1.0 * k / (Phase - 1)
            else:
                t[k] = 1.0 + (k - Phase + 1) * (2.0 * Cycle - 1.0) / (
                    Cycle * length - 1.0
                )
        beta = np.cos(np.pi * t)
        g = np.where(t <= 0.5, 1.0, 1.0 / (Coeff * t + 1.0))
        w = g * beta
        w_sum = np.sum(w)
        # Convolution; divide by effective available weight (cap at w_sum for early edges)
        x_filled = np.where(np.isfinite(x), x, 0.0)
        mask = np.isfinite(x).astype(float)
        num = np.convolve(x_filled, w, mode="full")[: len(x)]
        eff = np.convolve(mask, w, mode="full")[: len(x)]
        # cap denominator at total theoretical weight (MQL divides by total for full windows; for partial/NaNs use available weight)
        den = np.minimum(eff, w_sum)
        out = np.divide(num, den, out=np.full_like(x, np.nan), where=den != 0)
        # for very early few bars, return price (MQL returns raw for r<3)
        out[:3] = x[:3]
        return out

    def _atr_wilder(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 50
    ) -> np.ndarray:
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        tr = np.maximum.reduce(
            [
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ]
        )
        atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
        return atr

    # Price selection
    def _heiken_ashi(better: bool = False):
        ha_open = np.full(n, np.nan, dtype=float)
        ha_close = np.full(n, np.nan, dtype=float)
        ha_high = np.full(n, np.nan, dtype=float)
        ha_low = np.full(n, np.nan, dtype=float)

        for i in range(n):
            if better:
                if np.isfinite(h[i]) and np.isfinite(l[i]) and h[i] != l[i]:
                    hc = (o[i] + c[i]) / 2.0 + ((c[i] - o[i]) / (h[i] - l[i])) * np.abs(
                        (c[i] - o[i]) / 2.0
                    )
                else:
                    hc = (o[i] + c[i]) / 2.0
            else:
                hc = (o[i] + h[i] + l[i] + c[i]) / 4.0
            if i == 0:
                ho = (o[i] + c[i]) / 2.0
            else:
                ho = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
            hh = np.nanmax([h[i], ho, hc])
            hl = np.nanmin([l[i], ho, hc])
            ha_open[i] = ho
            ha_close[i] = hc
            ha_high[i] = hh
            ha_low[i] = hl
        return ha_open, ha_high, ha_low, ha_close

    def _select_price(mode: str) -> np.ndarray:
        m = str(mode).strip().lower()
        if m in ("close", "pr_close"):
            return c.copy()
        if m in ("open", "pr_open"):
            return o.copy()
        if m in ("high", "pr_high"):
            return h.copy()
        if m in ("low", "pr_low"):
            return l.copy()
        if m in ("median", "pr_median"):
            return (h + l) / 2.0
        if m in ("medianb", "pr_medianb"):
            return (o + c) / 2.0
        if m in ("typical", "pr_typical"):
            return (h + l + c) / 3.0
        if m in ("weighted", "pr_weighted"):
            return (h + l + 2.0 * c) / 4.0
        if m in ("average", "pr_average"):
            return (h + l + c + o) / 4.0
        if m in ("tbiased", "pr_tbiased"):
            return np.where(c > o, (h + c) / 2.0, (l + c) / 2.0)
        if m in ("tbiased2", "pr_tbiased2"):
            out = c.copy()
            out[c > o] = h[c > o]
            out[c < o] = l[c < o]
            return out
        # Heiken Ashi
        if m.startswith("ha"):
            better = m.startswith("hab")
            ho, hh, hl, hc = _heiken_ashi(better=better)
            if m in ("haclose", "pr_haclose", "habclose", "pr_habclose"):
                return hc
            if m in ("haopen", "pr_haopen", "habopen", "pr_habopen"):
                return ho
            if m in ("hahigh", "pr_hahigh", "habhigh", "pr_habhigh"):
                return hh
            if m in ("halow", "pr_halow", "hablow", "pr_hablow"):
                return hl
            if m in ("hamedian", "pr_hamedian", "habmedian", "pr_habmedian"):
                return (hh + hl) / 2.0
            if m in ("hamedianb", "pr_hamedianb", "habmedianb", "pr_habmedianb"):
                return (ho + hc) / 2.0
            if m in ("hatypical", "pr_hatypical", "habtypical", "pr_habtypical"):
                return (hh + hl + hc) / 3.0
            if m in ("haweighted", "pr_haweighted", "habweighted", "pr_habweighted"):
                return (hh + hl + 2.0 * hc) / 4.0
            if m in ("haaverage", "pr_haaverage", "habaverage", "pr_habaverage"):
                return (hh + hl + hc + ho) / 4.0
            if m in ("hatbiased", "pr_hatbiased", "habtbiased", "pr_habtbiased"):
                return np.where(hc > ho, (hh + hc) / 2.0, (hl + hc) / 2.0)
            if m in ("hatbiased2", "pr_hatbiased2", "habtbiased2", "pr_habtbiased2"):
                out = hc.copy()
                out[hc > ho] = hh[hc > ho]
                out[hc < ho] = hl[hc < ho]
                return out
        # default
        return (h + l) / 2.0

    # Moving average selection
    def _ma(arr: np.ndarray, method: str, period: int) -> np.ndarray:
        m = str(method).strip().lower()
        if m in ("sma", "ma_sma"):
            return _sma(arr, int(np.ceil(period)))
        if m in ("ema", "ma_ema"):
            alpha = 2.0 / (1.0 + period)
            return _ema(arr, alpha)
        if m in ("smma", "ma_smma"):
            return _smma(arr, int(np.ceil(period)))
        if m in ("lwma", "ma_lwma"):
            return _lwma(arr, int(np.ceil(period)))
        if m in ("slwma", "ma_slwma"):
            return _slwma(arr, int(np.ceil(period)))
        if m in ("dsema", "ma_dsema"):
            return _dsema(arr, float(period))
        if m in ("tema", "ma_tema"):
            return _tema(arr, int(np.ceil(period)))
        if m in ("lsma", "ma_lsma"):
            return _lsma(arr, int(np.ceil(period)))
        if m in ("nlma", "ma_nlma"):
            return _nlma(arr, float(period))
        # default: return input
        return arr.astype(float)

    price_series = _select_price(Price)
    ma_series = _ma(price_series, MaMethod, int(MaPeriod))

    # ATR(50)
    atr = _atr_wilder(h, l, c, period=50)

    # Glitch Index
    gli = (c - ma_series) / atr
    gli = np.where(np.isfinite(gli), gli, np.nan)
    # Emulate MT4 warmup: compute only from index >= 49 (ensure ATR warmup)
    warm = 49
    valid_mask = np.arange(n) >= warm
    gli = np.where(valid_mask, gli, np.nan)

    # Buckets and state
    gliUa = np.where(gli > level2, gli, np.nan)
    gliUb = np.where((gli > level1) & (gli <= level2), gli, np.nan)
    gliDa = np.where(gli < -level2, gli, np.nan)
    gliDb = np.where((gli < -level1) & (gli >= -level2), gli, np.nan)
    gliNe = np.where((gli >= -level1) & (gli <= level1), gli, np.nan)

    state = np.full(n, np.nan)
    state = np.where(gli > level2, 2, state)
    state = np.where((gli > level1) & (gli <= level2), 1, state)
    state = np.where(gli < -level2, -2, state)
    state = np.where((gli < -level1) & (gli >= -level2), -1, state)
    state = np.where((gli >= -level1) & (gli <= level1), 0, state)

    out = pd.DataFrame(
        {
            "gliUa": gliUa,
            "gliUb": gliUb,
            "gliDb": gliDb,
            "gliDa": gliDa,
            "gliNe": gliNe,
            "gli": gli,
            "state": state,
        },
        index=df.index,
    )
    return out


# === END GlitchIndexFixed_calc.py ===


# === BEGIN Hacolt202Lines_calc.py ===
def Hacolt202Lines(
    df: pd.DataFrame, Length: int = 55, CandleSize: float = 1.1, LtLength: int = 60
) -> pd.DataFrame:
    """Long-term Heikin-Ashi Candlestick Oscillator (HACO lt) by Sylvain Vervoort.
    Returns a DataFrame with columns ['HACOLT_Up','HACOLT_Dn','HACOLT'] aligned to df.index.
    Vectorized core with necessary stateful loops; handles NaNs and preserves warmup with NaNs.
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    # Helpers
    def _tema(x: pd.Series, period: int) -> pd.Series:
        ema1 = x.ewm(span=period, adjust=False, min_periods=1).mean()
        ema2 = ema1.ewm(span=period, adjust=False, min_periods=1).mean()
        ema3 = ema2.ewm(span=period, adjust=False, min_periods=1).mean()
        return 3.0 * (ema1 - ema2) + ema3

    def _zero_lag_tema(x: pd.Series, period: int) -> pd.Series:
        t1 = _tema(x, period)
        t2 = _tema(t1, period)
        return 2.0 * t1 - t2

    n = len(df)
    if n == 0:
        return pd.DataFrame(
            index=df.index, columns=["HACOLT_Up", "HACOLT_Dn", "HACOLT"], dtype=float
        )

    # Heikin-Ashi components needed by the algo
    median = (h + l) / 2.0
    ha_close_prev = ((o + h + l + c) / 4.0).shift(1)
    # haOpen is an EMA of ha_close_prev with alpha=0.5: y_t = 0.5*x_t + 0.5*y_{t-1}
    ha_open = ha_close_prev.ewm(alpha=0.5, adjust=False, min_periods=1).mean()
    ha_c = (
        median
        + ha_open
        + pd.Series(np.maximum(ha_open.values, h.values), index=df.index)
        + pd.Series(np.minimum(ha_open.values, l.values), index=df.index)
    ) / 4.0

    # Zero-lag TEMA on HA close and on median price
    temaa = _zero_lag_tema(ha_c, Length)
    temab = _zero_lag_tema(median, Length)
    delta = temab - temaa

    # Long-term EMA for ltSell condition
    ema_close_lt = c.ewm(span=LtLength, adjust=False, min_periods=1).mean()
    lt_sell = (c < ema_close_lt).values

    # Prepare numpy arrays for loop
    ha_c_np = ha_c.values
    ha_open_np = ha_open.values
    ha_c_prev = np.roll(ha_c_np, 1)
    ha_c_prev[0] = np.nan
    ha_open_prev = np.roll(ha_open_np, 1)
    ha_open_prev[0] = np.nan
    h_prev = np.roll(h.values, 1)
    h_prev[0] = np.nan
    l_prev = np.roll(l.values, 1)
    l_prev[0] = np.nan
    c_prev = np.roll(c.values, 1)
    c_prev[0] = np.nan

    o_np = o.values
    h_np = h.values
    l_np = l.values
    c_np = c.values
    median_np = median.values
    delta_np = delta.values

    # State arrays
    keeping1 = np.zeros(n, dtype=bool)
    keepall1 = np.zeros(n, dtype=bool)
    utr = np.zeros(n, dtype=bool)

    keeping2 = np.zeros(n, dtype=bool)
    keepall2 = np.zeros(n, dtype=bool)
    dtr = np.zeros(n, dtype=bool)

    upw = np.zeros(n, dtype=bool)
    dnw = np.zeros(n, dtype=bool)
    ltResult = np.zeros(n, dtype=bool)

    # Track last wave event (index and type)
    last_event_idx = -1
    last_event_is_up = False

    # Output values
    hacolt_val = np.full(n, np.nan, dtype=float)

    for t in range(n):
        # Up side conditions
        keep1_u = (
            (
                (ha_c_np[t] > ha_open_np[t])
                if np.all(np.isfinite([ha_c_np[t], ha_open_np[t]]))
                else False
            )
            or (
                (ha_c_prev[t] >= ha_open_prev[t])
                if np.all(np.isfinite([ha_c_prev[t], ha_open_prev[t]]))
                else False
            )
            or (
                (c_np[t] >= ha_c_np[t])
                if np.all(np.isfinite([c_np[t], ha_c_np[t]]))
                else False
            )
            or (
                (h_np[t] > h_prev[t])
                if np.all(np.isfinite([h_np[t], h_prev[t]]))
                else False
            )
            or (
                (l_np[t] > l_prev[t])
                if np.all(np.isfinite([l_np[t], l_prev[t]]))
                else False
            )
        )
        keep2_u = (delta_np[t] >= 0) if np.isfinite(delta_np[t]) else False
        keep3_u = (
            (abs(c_np[t] - o_np[t]) < (h_np[t] - l_np[t]) * CandleSize)
            if np.all(np.isfinite([c_np[t], o_np[t], h_np[t], l_np[t]]))
            else False
        ) and (
            (h_np[t] >= l_prev[t])
            if np.all(np.isfinite([h_np[t], l_prev[t]]))
            else False
        )

        keeping1[t] = keep1_u or keep2_u
        prev_keep1 = keeping1[t - 1] if t > 0 else False
        keepall1[t] = keeping1[t] or (
            prev_keep1
            and (
                (c_np[t] >= o_np[t])
                or ((c_np[t] >= c_prev[t]) if np.isfinite(c_prev[t]) else False)
            )
        )
        prev_keepall1 = keepall1[t - 1] if t > 0 else False
        utr[t] = keepall1[t] or (prev_keepall1 and keep3_u)

        # Down side conditions
        keep1_d = (
            (ha_c_np[t] < ha_open_np[t])
            if np.all(np.isfinite([ha_c_np[t], ha_open_np[t]]))
            else False
        ) or (
            (ha_c_prev[t] < ha_open_prev[t])
            if np.all(np.isfinite([ha_c_prev[t], ha_open_prev[t]]))
            else False
        )
        keep2_d = (delta_np[t] < 0) if np.isfinite(delta_np[t]) else False
        keep3_d = (
            (abs(c_np[t] - o_np[t]) < (h_np[t] - l_np[t]) * CandleSize)
            if np.all(np.isfinite([c_np[t], o_np[t], h_np[t], l_np[t]]))
            else False
        ) and (
            (l_np[t] <= h_prev[t])
            if np.all(np.isfinite([l_np[t], h_prev[t]]))
            else False
        )

        keeping2[t] = keep1_d or keep2_d
        prev_keep2 = keeping2[t - 1] if t > 0 else False
        keepall2[t] = keeping2[t] or (
            prev_keep2
            and (
                (c_np[t] < o_np[t])
                or ((c_np[t] < c_prev[t]) if np.isfinite(c_prev[t]) else False)
            )
        )
        prev_keepall2 = keepall2[t - 1] if t > 0 else False
        dtr[t] = keepall2[t] or (prev_keepall2 and keep3_d)

        # Wave transitions
        prev_dtr = dtr[t - 1] if t > 0 else False
        prev_utr = utr[t - 1] if t > 0 else False
        upw[t] = (not dtr[t]) and prev_dtr and utr[t]
        dnw[t] = (not utr[t]) and prev_utr and dtr[t]

        # Update last event
        if upw[t]:
            last_event_idx = t
            last_event_is_up = True
        elif dnw[t]:
            last_event_idx = t
            last_event_is_up = False

        # Result logic
        result = False
        if upw[t]:
            result = True
        elif dnw[t]:
            result = False
        elif (last_event_idx >= 0) and ((t - last_event_idx) < 50) and last_event_is_up:
            result = True

        # Long-term result persistence
        ltResult[t] = ltResult[t - 1] if t > 0 else False
        if result:
            ltResult[t] = True
        elif (not result) and lt_sell[t]:
            ltResult[t] = False

        # Final HACOLT value: 1, 0, -1
        if result:
            hacolt_val[t] = 1.0
        elif ltResult[t]:
            hacolt_val[t] = 0.0
        else:
            hacolt_val[t] = -1.0

    # Warmup NaNs (triple EMA + LT EMA warmup)
    warmup = int(max(Length * 3, LtLength))
    if warmup > 0 and n > 0:
        warm_mask = np.zeros(n, dtype=bool)
        warm_mask[: min(warmup, n)] = True
    else:
        warm_mask = np.zeros(n, dtype=bool)

    hacolt = pd.Series(hacolt_val, index=df.index)
    hacolt[warm_mask] = np.nan
    hacolt_up = hacolt.where(hacolt > 0)
    hacolt_dn = hacolt.where(hacolt < 0)

    return pd.DataFrame(
        {"HACOLT_Up": hacolt_up, "HACOLT_Dn": hacolt_dn, "HACOLT": hacolt},
        index=df.index,
    )


# === END Hacolt202Lines_calc.py ===


# === BEGIN Hlctrend_calc.py ===
def Hlctrend(
    df: pd.DataFrame, close_period: int = 5, low_period: int = 13, high_period: int = 34
) -> pd.DataFrame:
    """Return ALL lines as columns ['first','second']; aligned to df.index.
    first = EMA(Close, close_period) - EMA(High, high_period)
    second = EMA(Low, low_period) - EMA(Close, close_period)
    Uses EMA with smoothing 2/(n+1); warmup NaNs via min_periods per leg.
    """
    close = df["Close"].astype(float)
    low = df["Low"].astype(float)
    high = df["High"].astype(float)

    emac = close.ewm(span=close_period, adjust=False, min_periods=close_period).mean()
    emal = low.ewm(span=low_period, adjust=False, min_periods=low_period).mean()
    emah = high.ewm(span=high_period, adjust=False, min_periods=high_period).mean()

    first = emac - emah
    second = emal - emac

    return pd.DataFrame({"first": first, "second": second}, index=df.index)


# === END Hlctrend_calc.py ===


# === BEGIN ISCalculation_calc.py ===
def ISCalculation(
    df: pd.DataFrame, period: int = 10, nbchandelier: int = 10, lag: int = 0
) -> pd.DataFrame:
    """Return ADJASUROPPO ('Pente') and its EMA trigger as columns aligned to df.index.
    Parameters:
      - period: EMA period for base EMA and trigger base
      - nbchandelier: lookback distance for slope (difference divided by nbchandelier)
      - lag: added to period for the trigger EMA period
    """
    close = df["Close"].astype(float)

    p = max(int(period), 1)
    n = max(int(nbchandelier), 1)
    p_trig = max(p + int(lag), 1)

    ema = close.ewm(span=p, adjust=False, min_periods=p).mean()
    pente = (ema - ema.shift(n)) / n
    trigger = pente.ewm(span=p_trig, adjust=False, min_periods=p_trig).mean()

    out = pd.DataFrame({"Pente": pente, "Trigger": trigger}, index=df.index)

    return out


# === END ISCalculation_calc.py ===


# === BEGIN Jposcillator_calc.py ===
def JpOscillator(
    df: pd.DataFrame, period: int = 5, mode: int = 0, smoothing: bool = True
) -> pd.DataFrame:
    """Return JpOscillator lines as columns ['Jp','JpUp','JpDown'], aligned to df.index.

    Params:
    - period: int, moving average length for smoothing (Period1 in MQL4). Default 5.
    - mode: int, MA type over buffer (Mode1 in MQL4): 0=SMA, 1=EMA, 2=SMMA (Wilder/Smoothed), 3=LWMA. Default 0.
    - smoothing: bool, if True apply MA on buffer; if False use raw buffer.
    """
    close = df["Close"].astype(float)

    # Buffer calculation (uses forward-looking shifts like MQL4 indexing)
    c0 = close
    c1 = close.shift(-1)
    c2 = close.shift(-2)
    c4 = close.shift(-4)
    buff = 2.0 * c0 - 0.5 * c1 - 0.5 * c2 - c4

    # Helper MA functions (iMAOnArray equivalents)
    def _sma(s: pd.Series, n: int) -> pd.Series:
        if n <= 1:
            return s.copy()
        return s.rolling(n, min_periods=n).mean()

    def _ema(s: pd.Series, n: int) -> pd.Series:
        if n <= 1:
            return s.copy()
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def _smma(s: pd.Series, n: int) -> pd.Series:
        # MT4 Smoothed MA (Wilder). Seeded with SMA(n), then recursive.
        if n <= 1:
            return s.copy()
        arr = s.to_numpy(dtype=float)
        out = np.full(arr.shape[0], np.nan, dtype=float)
        sma = pd.Series(arr).rolling(n, min_periods=n).mean().to_numpy()
        idxs = np.where(np.isfinite(sma))[0]
        if idxs.size == 0:
            return pd.Series(out, index=s.index)
        start = idxs[0]
        out[start] = sma[start]
        for i in range(start + 1, arr.shape[0]):
            if np.isfinite(arr[i]) and np.isfinite(out[i - 1]):
                out[i] = (out[i - 1] * (n - 1) + arr[i]) / n
            else:
                out[i] = np.nan
        return pd.Series(out, index=s.index)

    def _lwma(s: pd.Series, n: int) -> pd.Series:
        if n <= 1:
            return s.copy()
        w = np.arange(1, n + 1, dtype=float)
        denom = w.sum()
        return s.rolling(n, min_periods=n).apply(
            lambda x: np.dot(x, w) / denom if np.isfinite(x).all() else np.nan, raw=True
        )

    def _ma_on_array(s: pd.Series, n: int, m: int) -> pd.Series:
        if m == 0:
            return _sma(s, n)
        elif m == 1:
            return _ema(s, n)
        elif m == 2:
            return _smma(s, n)
        elif m == 3:
            return _lwma(s, n)
        else:
            return _sma(s, n)

    if smoothing:
        ma = _ma_on_array(buff, int(max(1, period)), int(mode))
    else:
        ma = buff.copy()

    cond_up = ma > ma.shift(1)
    up = ma.where(cond_up, np.nan)
    down = ma.where(~cond_up, np.nan)

    out = pd.DataFrame(
        {
            "Jp": ma.astype(float),
            "JpUp": up.astype(float),
            "JpDown": down.astype(float),
        },
        index=df.index,
    )
    return out


# === END Jposcillator_calc.py ===


# === BEGIN McginleyDynamic23_calc.py ===
def McginleyDynamic23(
    df: pd.DataFrame,
    period: int = 12,
    price: str = "close",
    constant: float = 5.0,
    method: str = "ema",
) -> pd.DataFrame:
    """
    McGinley Dynamic average (mladen version) with optional MA basis.
    Returns all plotted buffers as columns: mcg (main), mcg_down_a, mcg_down_b.

    Parameters:
    - period: int, default 12
    - price: {'close','open','high','low','median','typical','weighted'}, default 'close'
    - constant: float, default 5.0
    - method: {'sma','ema','smma','lwma','gen'}, default 'ema'
      'gen' is the average of SMA, EMA, SMMA, and LWMA (as in source).
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    price_key = str(price).lower()
    if price_key == "close":
        p = df["Close"].astype(float)
    elif price_key == "open":
        p = df["Open"].astype(float)
    elif price_key == "high":
        p = df["High"].astype(float)
    elif price_key == "low":
        p = df["Low"].astype(float)
    elif price_key == "median":
        p = (df["High"] + df["Low"]) / 2.0
    elif price_key == "typical":
        p = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif price_key == "weighted":
        p = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        raise ValueError("Unsupported price type")

    method_key = str(method).lower()

    # Helper MAs
    def ma_sma(x: pd.Series, n: int) -> pd.Series:
        return x.rolling(n, min_periods=n).mean()

    def ma_ema(x: pd.Series, n: int) -> pd.Series:
        return x.ewm(span=n, adjust=False, min_periods=1).mean()

    def ma_smma(x: pd.Series, n: int) -> pd.Series:
        # Wilder's smoothing approximated via EMA with alpha=1/n
        return x.ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean()

    def ma_lwma(x: pd.Series, n: int) -> pd.Series:
        if n == 1:
            return x.astype(float)
        w = np.arange(1, n + 1, dtype=float)
        w_sum = w.sum()
        return x.rolling(n, min_periods=n).apply(
            lambda a: np.dot(a, w) / w_sum, raw=True
        )

    if method_key == "sma":
        ma = ma_sma(p, period)
    elif method_key == "ema":
        ma = ma_ema(p, period)
    elif method_key == "smma":
        ma = ma_smma(p, period)
    elif method_key == "lwma":
        ma = ma_lwma(p, period)
    elif method_key == "gen":
        sma = ma_sma(p, period)
        ema = ma_ema(p, period)
        smma = ma_smma(p, period)
        lwma = ma_lwma(p, period)
        ma = (sma + ema + smma + lwma) / 4.0
    else:
        raise ValueError("Unsupported method")

    ma_shift = ma.shift(1)

    # McGinley Dynamic calculation
    # mcg[t] = ma[t-1] + (p[t] - ma[t-1]) / (constant * period * (p[t]/ma[t-1])^4)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p / ma_shift
        denom = constant * period * np.power(ratio, 4)
        mcg = ma_shift + (p - ma_shift) / denom

    # Mask invalid where MA unavailable or denom invalid
    invalid = (~np.isfinite(ma_shift)) | (~np.isfinite(denom)) | (ma_shift == 0.0)
    mcg = mcg.where(~invalid)

    # Down-slope segmentation into two alternating buffers
    neg = mcg < mcg.shift(1)
    starts = neg & ~neg.shift(1, fill_value=False)
    run_id = starts.cumsum()
    run_id = run_id.where(neg, 0)
    odd = (run_id % 2 == 1) & neg

    mcg_down_a = mcg.where(odd)
    mcg_down_b = mcg.where(neg & ~odd)

    out = pd.DataFrame(
        {
            "mcg": mcg.astype(float),
            "mcg_down_a": mcg_down_a.astype(float),
            "mcg_down_b": mcg_down_b.astype(float),
        },
        index=df.index,
    )
    return out


# === END McginleyDynamic23_calc.py ===


# === BEGIN MetroAdvanced_calc.py ===
def MetroAdvanced(
    df: pd.DataFrame,
    period_rsi: int = 14,
    rsi_type: str = "rsi",  # {"rsi","wilder","rsx","cutler"}
    price: str = "close",  # {"close","open","high","low","median","typical","weighted","median_body","average","trend_biased","volume"}
    step_size_fast: float = 5.0,
    step_size_slow: float = 15.0,
    over_sold: float = 10.0,
    over_bought: float = 90.0,
    minmax_period: int = 49,
) -> pd.DataFrame:
    """
    Return ALL lines as columns aligned to df.index.
    Columns:
      RSI, StepRSI_fast, StepRSI_slow, Level_Up, Level_Mid, Level_Dn,
      TrendFast, TrendSlow, MinFast, MinSlow, MaxFast, MaxSlow, Trend
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            index=df.index,
            columns=[
                "RSI",
                "StepRSI_fast",
                "StepRSI_slow",
                "Level_Up",
                "Level_Mid",
                "Level_Dn",
                "TrendFast",
                "TrendSlow",
                "MinFast",
                "MinSlow",
                "MaxFast",
                "MaxSlow",
                "Trend",
            ],
            dtype=float,
        )

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)

    price_key = str(price).lower()
    if price_key == "close":
        p = c
    elif price_key == "open":
        p = o
    elif price_key == "high":
        p = h
    elif price_key == "low":
        p = l
    elif price_key == "median":
        p = (h + l) / 2.0
    elif price_key == "typical":
        p = (h + l + c) / 3.0
    elif price_key == "weighted":
        p = (h + l + 2.0 * c) / 4.0
    elif price_key == "median_body":
        p = (o + c) / 2.0
    elif price_key == "average":
        p = (h + l + c + o) / 4.0
    elif price_key == "trend_biased":
        p = np.where(o > c, (h + c) / 2.0, (l + c) / 2.0)
    elif price_key == "volume":
        p = v
    else:
        p = c

    p = p.astype(float)
    # RSI computation (matching MQL4 logic)
    rsi_type_key = str(rsi_type).lower()
    rsi = np.full(n, np.nan, dtype=float)

    if rsi_type_key in ("rsi", "regular", "regular_rsi"):
        # Custom "regular" per source: EMA(alpha=1/period) of change and abs(change) with special warmup
        alpha = 1.0 / max(float(period_rsi), 1.0)
        change_ema = 0.0
        abs_change_ema = 0.0
        for t in range(n):
            if t == 0 or not np.isfinite(p[t]) or not np.isfinite(p[t - 1]):
                delta = 0.0 if t == 0 else (p[t] - p[t - 1])
            else:
                delta = p[t] - p[t - 1]

            if t < period_rsi:
                # warmup as in MQL: average change from first and average abs change over available data
                kmax = min(period_rsi, t + 1)  # number of points including current
                k = max(kmax - 1, 0)
                if k == 0:
                    avg_change = 0.0
                    avg_abs = 0.0
                else:
                    # sum |p[t-j] - p[t-j-1]| for j=0..k-1
                    diffs = p[max(0, t - k + 1) : t + 1] - p[max(0, t - k) : t]
                    diffs = diffs[np.isfinite(diffs)]
                    if diffs.size == 0:
                        avg_abs = 0.0
                        avg_change = 0.0
                    else:
                        avg_abs = np.abs(diffs).sum() / k
                        # (p[t] - p[0]) / k
                        if np.isfinite(p[t]) and np.isfinite(p[0]):
                            avg_change = (p[t] - p[0]) / k
                        else:
                            avg_change = 0.0
                change_ema = avg_change
                abs_change_ema = avg_abs
            else:
                change_ema = change_ema + alpha * (delta - change_ema)
                abs_change_ema = abs_change_ema + alpha * (abs(delta) - abs_change_ema)

            if abs_change_ema != 0.0:
                rsi[t] = 50.0 * (change_ema / abs_change_ema + 1.0)
            else:
                rsi[t] = 50.0

    elif rsi_type_key in ("wilder", "wilders", "wil", "rsi_wil"):
        # Wilder's RSI variant per source using SMMA with period = 0.5*(period-1)
        delta = np.diff(p, prepend=p[0])
        pos_in = np.maximum(delta, 0.0) * 0.5 + 0.5 * np.abs(delta)
        neg_in = np.maximum(-delta, 0.0) * 0.5 + 0.5 * np.abs(delta)
        # Note: in source pos = 0.5*(|d|+d) -> max(d,0); neg = 0.5*(|d|-d) -> max(-d,0)
        pos_in = np.maximum(delta, 0.0)
        neg_in = np.maximum(-delta, 0.0)
        smma_period = 0.5 * (float(period_rsi) - 1.0)
        smma_period = max(smma_period, 1e-12)  # avoid division by zero

        pos_smma = np.zeros(n, dtype=float)
        neg_smma = np.zeros(n, dtype=float)
        for t in range(n):
            if t < smma_period:
                pos_smma[t] = pos_in[t] if np.isfinite(pos_in[t]) else 0.0
                neg_smma[t] = neg_in[t] if np.isfinite(neg_in[t]) else 0.0
            else:
                pos_prev = pos_smma[t - 1]
                neg_prev = neg_smma[t - 1]
                x_pos = pos_in[t] if np.isfinite(pos_in[t]) else 0.0
                x_neg = neg_in[t] if np.isfinite(neg_in[t]) else 0.0
                pos_smma[t] = pos_prev + (x_pos - pos_prev) / smma_period
                neg_smma[t] = neg_prev + (x_neg - neg_prev) / smma_period

            denom = pos_smma[t] + neg_smma[t]
            if denom != 0.0:
                rsi[t] = 100.0 * pos_smma[t] / denom
            else:
                rsi[t] = 50.0

    elif rsi_type_key in ("rsx", "rsx_rsi", "rsi_rsx"):
        # RSX per source (Ehlers-like IIR)
        Kg = 3.0 / (2.0 + float(period_rsi))
        Hg = 1.0 - Kg
        # states for mom (m1..m6) and |mom| (a1..a6)
        m1 = m2 = m3 = m4 = m5 = m6 = 0.0
        a1 = a2 = a3 = a4 = a5 = a6 = 0.0
        for t in range(n):
            if t == 0 or not np.isfinite(p[t]) or not np.isfinite(p[t - 1]):
                mom = 0.0
            else:
                mom = p[t] - p[t - 1]
            moa = abs(mom)

            if t < period_rsi:
                # reset states as in MQL for r<period
                m1 = m2 = m3 = m4 = m5 = m6 = 0.0
                a1 = a2 = a3 = a4 = a5 = a6 = 0.0
                rsi[t] = 50.0
                continue

            # cascade 1
            m1 = Kg * mom + Hg * m1
            m2 = Kg * m1 + Hg * m2
            mom_eff = 1.5 * m1 - 0.5 * m2
            a1 = Kg * moa + Hg * a1
            a2 = Kg * a1 + Hg * a2
            moa_eff = 1.5 * a1 - 0.5 * a2
            # cascade 2
            m3 = Kg * mom_eff + Hg * m3
            m4 = Kg * m3 + Hg * m4
            mom_eff = 1.5 * m3 - 0.5 * m4
            a3 = Kg * moa_eff + Hg * a3
            a4 = Kg * a3 + Hg * a4
            moa_eff = 1.5 * a3 - 0.5 * a4
            # cascade 3
            m5 = Kg * mom_eff + Hg * m5
            m6 = Kg * m5 + Hg * m6
            mom_eff = 1.5 * m5 - 0.5 * m6
            a5 = Kg * moa_eff + Hg * a5
            a6 = Kg * a5 + Hg * a6
            moa_eff = 1.5 * a5 - 0.5 * a6

            if moa_eff != 0.0:
                rsi[t] = np.clip((mom_eff / moa_eff + 1.0) * 50.0, 0.0, 100.0)
            else:
                rsi[t] = 50.0

    elif rsi_type_key in ("cutler", "cutlers", "cut", "rsi_cut"):
        delta = np.diff(p, prepend=p[0])
        pos = np.where(delta > 0.0, delta, 0.0)
        neg = np.where(delta < 0.0, -delta, 0.0)
        pos_roll = (
            pd.Series(pos).rolling(period_rsi, min_periods=period_rsi).sum().to_numpy()
        )
        neg_roll = (
            pd.Series(neg).rolling(period_rsi, min_periods=period_rsi).sum().to_numpy()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(neg_roll > 0.0, pos_roll / neg_roll, np.nan)
        rsi = np.where(np.isfinite(rs), 100.0 - 100.0 / (1.0 + rs), np.nan)
        # per MQL return 50 when denominator is 0 or not enough data
        rsi = np.where(np.isfinite(rsi), rsi, 50.0)
    else:
        # default to custom "rsi" mode
        alpha = 1.0 / max(float(period_rsi), 1.0)
        change_ema = 0.0
        abs_change_ema = 0.0
        for t in range(n):
            if t == 0 or not np.isfinite(p[t]) or not np.isfinite(p[t - 1]):
                delta = 0.0 if t == 0 else (p[t] - p[t - 1])
            else:
                delta = p[t] - p[t - 1]

            if t < period_rsi:
                kmax = min(period_rsi, t + 1)
                k = max(kmax - 1, 0)
                if k == 0:
                    avg_change = 0.0
                    avg_abs = 0.0
                else:
                    diffs = p[max(0, t - k + 1) : t + 1] - p[max(0, t - k) : t]
                    diffs = diffs[np.isfinite(diffs)]
                    if diffs.size == 0:
                        avg_abs = 0.0
                        avg_change = 0.0
                    else:
                        avg_abs = np.abs(diffs).sum() / k
                        if np.isfinite(p[t]) and np.isfinite(p[0]):
                            avg_change = (p[t] - p[0]) / k
                        else:
                            avg_change = 0.0
                change_ema = avg_change
                abs_change_ema = avg_abs
            else:
                change_ema = change_ema + alpha * (delta - change_ema)
                abs_change_ema = abs_change_ema + alpha * (abs(delta) - abs_change_ema)

            if abs_change_ema != 0.0:
                rsi[t] = 50.0 * (change_ema / abs_change_ema + 1.0)
            else:
                rsi[t] = 50.0

    # Step logic (fast/slow)
    maxf = rsi + 2.0 * step_size_fast
    minf = rsi - 2.0 * step_size_fast
    maxs = rsi + 2.0 * step_size_slow
    mins = rsi - 2.0 * step_size_slow

    trendf = np.zeros(n, dtype=float)
    trends = np.zeros(n, dtype=float)

    # We need recursive adjustments based on prior bar
    for t in range(n):
        if t == 0 or not np.isfinite(rsi[t]):
            # keep initial values as computed; trend states stay 0
            continue

        # Fast
        tf = trendf[t - 1]
        maxf_prev = maxf[t - 1]
        minf_prev = minf[t - 1]
        if rsi[t] > maxf_prev:
            tf = 1.0
        if rsi[t] < minf_prev:
            tf = -1.0
        # adjust channel
        mf = minf[t]
        Mf = maxf[t]
        if tf > 0 and mf < minf_prev:
            mf = minf_prev
        if tf < 0 and Mf > maxf_prev:
            Mf = maxf_prev
        trendf[t] = tf
        minf[t] = mf
        maxf[t] = Mf

        # Slow
        ts = trends[t - 1]
        maxs_prev = maxs[t - 1]
        mins_prev = mins[t - 1]
        if rsi[t] > maxs_prev:
            ts = 1.0
        if rsi[t] < mins_prev:
            ts = -1.0
        ms = mins[t]
        Ms = maxs[t]
        if ts > 0 and ms < mins_prev:
            ms = mins_prev
        if ts < 0 and Ms > maxs_prev:
            Ms = maxs_prev
        trends[t] = ts
        mins[t] = ms
        maxs[t] = Ms

    # Step lines
    step_fast_line = np.full(n, np.nan, dtype=float)
    step_slow_line = np.full(n, np.nan, dtype=float)
    pos_fast = trendf > 0
    neg_fast = trendf < 0
    step_fast_line[pos_fast] = minf[pos_fast] + step_size_fast
    step_fast_line[neg_fast] = maxf[neg_fast] - step_size_fast

    pos_slow = trends > 0
    neg_slow = trends < 0
    step_slow_line[pos_slow] = mins[pos_slow] + step_size_slow
    step_slow_line[neg_slow] = maxs[neg_slow] - step_size_slow

    # Levels based on rolling hi/lo of RSI over MinMaxPeriod
    rsi_s = pd.Series(rsi)
    hi = rsi_s.rolling(minmax_period, min_periods=1).max().to_numpy()
    lo = rsi_s.rolling(minmax_period, min_periods=1).min().to_numpy()
    rn = hi - lo
    level_up = lo + rn * (over_bought / 100.0)
    level_dn = lo + rn * (over_sold / 100.0)
    level_mid = 0.5 * (level_up + level_dn)

    # Final trend based on step lines
    trend = np.zeros(n, dtype=float)
    for t in range(n):
        s2 = step_fast_line[t]
        s3 = step_slow_line[t]
        if np.isfinite(s2) and np.isfinite(s3):
            if s2 > s3:
                trend[t] = 1.0
            elif s2 < s3:
                trend[t] = -1.0
            else:
                trend[t] = trend[t - 1] if t > 0 else 0.0
        else:
            trend[t] = trend[t - 1] if t > 0 else 0.0

    out = pd.DataFrame(
        {
            "RSI": rsi,
            "StepRSI_fast": step_fast_line,
            "StepRSI_slow": step_slow_line,
            "Level_Up": level_up,
            "Level_Mid": level_mid,
            "Level_Dn": level_dn,
            "TrendFast": trendf,
            "TrendSlow": trends,
            "MinFast": minf,
            "MinSlow": mins,
            "MaxFast": maxf,
            "MaxSlow": maxs,
            "Trend": trend,
        },
        index=df.index,
    )
    return out


# === END MetroAdvanced_calc.py ===


# === BEGIN METROFixed_calc.py ===
def METROFixed(
    df: pd.DataFrame,
    period_rsi: int = 14,
    step_size_fast: float = 5.0,
    step_size_slow: float = 15.0,
) -> pd.DataFrame:
    """Return RSI and StepRSI (fast/slow) aligned to df.index.
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume']
    - period_rsi: RSI period (Wilder's smoothing)
    - step_size_fast: step size for fast StepRSI (in RSI points)
    - step_size_slow: step size for slow StepRSI (in RSI points)
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            {"RSI": [], "StepRSI_fast": [], "StepRSI_slow": []}, index=df.index
        )

    close = df["Close"].astype(float)

    # Wilder's RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / period_rsi, adjust=False, min_periods=period_rsi
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / period_rsi, adjust=False, min_periods=period_rsi
    ).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.clip(lower=0.0, upper=100.0)

    rsi_vals = rsi.to_numpy(dtype=float)

    step_fast = np.full(n, np.nan, dtype=float)
    step_slow = np.full(n, np.nan, dtype=float)

    # Initialize with last available RSI if finite
    if np.isfinite(rsi_vals[-1]):
        step_fast[-1] = rsi_vals[-1]
        step_slow[-1] = rsi_vals[-1]

        # Backward recursion (from end to start), clamped by step sizes
        for i in range(n - 2, -1, -1):
            r = rsi_vals[i]
            nf = step_fast[i + 1]
            ns = step_slow[i + 1]

            # Fast
            lo_f = r - step_size_fast
            hi_f = r + step_size_fast
            step_fast[i] = np.minimum(np.maximum(nf, lo_f), hi_f)

            # Slow
            lo_s = r - step_size_slow
            hi_s = r + step_size_slow
            step_slow[i] = np.minimum(np.maximum(ns, lo_s), hi_s)

    out = pd.DataFrame(
        {
            "RSI": rsi,
            "StepRSI_fast": step_fast,
            "StepRSI_slow": step_slow,
        },
        index=df.index,
    )
    return out


# === END METROFixed_calc.py ===


# === BEGIN MomentumCandlesModifiedWAtr_calc.py ===
def MomentumCandlesModifiedWAtr(
    df: pd.DataFrame, atr_period: int = 50, atr_multiplier: float = 2.5
) -> pd.DataFrame:
    """Return Value and threshold lines as columns aligned to df.index; vectorized, handle NaNs.

    Parameters:
    - atr_period: ATR lookback period (default 50 as in the MQL4 code)
    - atr_multiplier: threshold multiplier (default 2.5 as in the MQL4 code)
    """
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / float(atr_period), adjust=False, min_periods=atr_period
    ).mean()

    value = (close - open_) / atr.replace(0.0, np.nan)

    thr_pos_val = 1.0 / float(atr_multiplier)
    thr_neg_val = -thr_pos_val

    valid_mask = atr.notna()
    threshold_pos = pd.Series(
        np.where(valid_mask, thr_pos_val, np.nan), index=df.index, dtype=float
    )
    threshold_neg = pd.Series(
        np.where(valid_mask, thr_neg_val, np.nan), index=df.index, dtype=float
    )

    out = pd.DataFrame(
        {
            "Value": value.astype(float),
            "Threshold_Pos": threshold_pos,
            "Threshold_Neg": threshold_neg,
        },
        index=df.index,
    )

    return out


# === END MomentumCandlesModifiedWAtr_calc.py ===


# === BEGIN MomentumCandlesWATR_calc.py ===
def MomentumCandlesWATR(
    df: pd.DataFrame, atr_period: int = 50, atr_multiplier: float = 2.5
) -> pd.DataFrame:
    """Return Bull/Bear Open/Close buffers for Momentum Candles with ATR filter.
    Signals:
      - Bullish if Close > Open and ATR(atr_period)/abs(Close-Open) < atr_multiplier
      - Bearish if Close < Open and ATR(atr_period)/abs(Close-Open) < atr_multiplier
    Outputs are aligned to df.index with NaNs where no signal.
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    # True Range
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

    # Wilder's ATR via EWM(alpha=1/period, adjust=False)
    atr = tr.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()

    body = (c - o).abs()
    ratio = atr / body.replace(0, np.nan)

    up_mask = (c > o) & (ratio < atr_multiplier)
    dn_mask = (c < o) & (ratio < atr_multiplier)

    bull_open = np.where(up_mask, o, np.nan)
    bull_close = np.where(up_mask, c, np.nan)
    bear_open = np.where(dn_mask, o, np.nan)
    bear_close = np.where(dn_mask, c, np.nan)

    out = pd.DataFrame(
        {
            "BullOpen": bull_open,
            "BullClose": bull_close,
            "BearOpen": bear_open,
            "BearClose": bear_close,
        },
        index=df.index,
    )

    return out


# === END MomentumCandlesWATR_calc.py ===


# === BEGIN PrecisionTrendHistogram_calc.py ===
def PrecisionTrendHistogram(
    df: pd.DataFrame, avg_period: int = 30, sensitivity: float = 3.0
) -> pd.DataFrame:
    """Precision Trend Histogram (Up/Down histograms and internal Trend state).
    Returns a DataFrame with columns ['Up','Down','Trend'] aligned to df.index.
    - avg_period: rolling average period for range (High-Low)
    - sensitivity: multiplier applied to the rolling average
    Vectorized where possible; sequential state update as per original MQL4 logic."""
    if df.empty:
        return pd.DataFrame(
            index=df.index, columns=["Up", "Down", "Trend"], dtype=float
        )

    high = df["High"].to_numpy(dtype=float, copy=False)
    low = df["Low"].to_numpy(dtype=float, copy=False)
    close = df["Close"].to_numpy(dtype=float, copy=False)

    period = max(int(avg_period), 1)

    rng = high - low
    avg = pd.Series(rng, index=df.index).rolling(
        window=period, min_periods=1
    ).mean().to_numpy() * float(sensitivity)

    n = len(df)
    trend = np.full(n, np.nan, dtype=float)
    avgd = np.full(n, np.nan, dtype=float)
    avgu = np.full(n, np.nan, dtype=float)
    minc = np.full(n, np.nan, dtype=float)
    maxc = np.full(n, np.nan, dtype=float)

    valid = np.isfinite(close) & np.isfinite(avg)
    if not np.any(valid):
        return pd.DataFrame(
            {"Up": np.full(n, np.nan), "Down": np.full(n, np.nan), "Trend": trend},
            index=df.index,
        )

    # Initialize at first valid index
    k = int(np.argmax(valid))
    if valid[k]:
        trend[k] = 0.0
        avgd[k] = close[k] - avg[k]
        avgu[k] = close[k] + avg[k]
        minc[k] = close[k]
        maxc[k] = close[k]

    last = k
    # Sequential state updates
    for t in range(k + 1, n):
        if not valid[t]:
            continue

        # Carry forward previous state
        trend[t] = trend[last]
        avgd[t] = avgd[last]
        avgu[t] = avgu[last]
        minc[t] = minc[last]
        maxc[t] = maxc[last]

        tp = trend[last]

        if tp == 0:
            if close[t] > avgu[last]:
                minc[t] = close[t]
                avgd[t] = close[t] - avg[t]
                trend[t] = 1.0
            if close[t] < avgd[last]:
                maxc[t] = close[t]
                avgu[t] = close[t] + avg[t]
                trend[t] = -1.0

        elif tp == 1:
            avgd[t] = minc[last] - avg[t]
            if close[t] > minc[last]:
                minc[t] = close[t]
            if close[t] < avgd[last]:
                maxc[t] = close[t]
                avgu[t] = close[t] + avg[t]
                trend[t] = -1.0

        elif tp == -1:
            avgu[t] = maxc[last] + avg[t]
            if close[t] < maxc[last]:
                maxc[t] = close[t]
            if close[t] > avgu[last]:
                minc[t] = close[t]
                avgd[t] = close[t] - avg[t]
                trend[t] = 1.0

        last = t

    up = np.where(trend == 1.0, 1.0, np.nan)
    down = np.where(trend == -1.0, 1.0, np.nan)

    return pd.DataFrame({"Up": up, "Down": down, "Trend": trend}, index=df.index)


# === END PrecisionTrendHistogram_calc.py ===


# === BEGIN PriceMomentumOscillator_calc.py ===
def PriceMomentumOscillator(
    df: pd.DataFrame, one: int = 35, two: int = 20, period: int = 10
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Two-stage custom EMA smoothing (alpha=2/one, 2/two) of Close percentage change,
    then EMA signal with period='period' on PMO."""
    if one <= 0 or two <= 0 or period <= 0:
        raise ValueError(
            "Parameters 'one', 'two', and 'period' must be positive integers."
        )

    close = df["Close"].astype(float)

    raw_one = (close / close.shift(1) * 100.0) - 100.0

    alpha1 = 2.0 / float(one)
    alpha2 = 2.0 / float(two)

    c1 = raw_one.ewm(alpha=alpha1, adjust=False, min_periods=1).mean()
    raw_two = 10.0 * c1
    pmo = raw_two.ewm(alpha=alpha2, adjust=False, min_periods=1).mean()

    signal = pmo.ewm(span=period, adjust=False, min_periods=1).mean()

    out = pd.DataFrame({"PMO": pmo, "Signal": signal}, index=df.index)

    return out


# === END PriceMomentumOscillator_calc.py ===


# === BEGIN QQEWithAlerts_calc.py ===
def QQEWithAlerts(df: pd.DataFrame, rsi_period: int = 14, sf: int = 5) -> pd.DataFrame:
    """Return QQE lines as columns aligned to df.index:
    - 'QQE_RSI_MA': EMA of RSI
    - 'QQE_TrendLevel': trailing level line
    Vectorized where possible; preserves NaNs for warmup."""
    close = pd.to_numeric(df["Close"], errors="coerce")

    # Wilder's RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (alpha = 1/period)
    avg_gain = gain.ewm(
        alpha=1 / rsi_period, adjust=False, min_periods=rsi_period
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / rsi_period, adjust=False, min_periods=rsi_period
    ).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # EMA of RSI with period sf
    rsi_ma = rsi.ewm(span=sf, adjust=False, min_periods=sf).mean()

    # ATR of RSI MA (absolute step)
    atr_rsi = (rsi_ma - rsi_ma.shift(1)).abs()

    # Wilders_Period used as EMA length to approximate Wilder's smoothing
    wilders_period = rsi_period * 2 - 1

    # Double-smoothed ATR of RSI
    ma_atr_rsi_1 = atr_rsi.ewm(
        span=wilders_period, adjust=False, min_periods=wilders_period
    ).mean()
    ma_atr_rsi_2 = ma_atr_rsi_1.ewm(
        span=wilders_period, adjust=False, min_periods=wilders_period
    ).mean()

    dar = ma_atr_rsi_2 * 4.236

    # Sequential computation of trailing level (TrLevelSlow)
    n = len(close)
    tr_level = np.full(n, np.nan, dtype=float)

    rsi_ma_np = rsi_ma.to_numpy(dtype=float)
    dar_np = dar.to_numpy(dtype=float)

    # Find the first index where we have valid values and a previous value
    valid = ~np.isnan(rsi_ma_np) & ~np.isnan(dar_np)
    if n >= 2:
        # Need both t and t-1 valid
        start_idx_candidates = np.where(valid & np.roll(valid, 1))[0]
        start_idx_candidates = start_idx_candidates[start_idx_candidates >= 1]
        if start_idx_candidates.size > 0:
            t0 = int(start_idx_candidates[0])
            tr = rsi_ma_np[t0 - 1]
            rsi1 = rsi_ma_np[t0 - 1]
            for t in range(t0, n):
                rsi0 = rsi_ma_np[t]
                d = dar_np[t]
                if np.isnan(rsi0) or np.isnan(d):
                    # keep NaN; carry forward rsi1 if possible
                    if not np.isnan(rsi0):
                        rsi1 = rsi0
                    continue
                dv = tr
                if rsi0 < tr:
                    tr = rsi0 + d
                    if rsi1 < dv and tr > dv:
                        tr = dv
                elif rsi0 > tr:
                    tr = rsi0 - d
                    if rsi1 > dv and tr < dv:
                        tr = dv
                # else: tr remains unchanged
                tr_level[t] = tr
                rsi1 = rsi0

    out = pd.DataFrame(
        {"QQE_RSI_MA": rsi_ma, "QQE_TrendLevel": pd.Series(tr_level, index=df.index)},
        index=df.index,
    )

    return out


# === END QQEWithAlerts_calc.py ===


# === BEGIN RangeFilterModified_calc.py ===
def RangeFilterModified(
    df: pd.DataFrame, atr_period: int = 14, multiplier: float = 3.0
) -> pd.DataFrame:
    """Return ALL lines as columns ['LineCenter','LineUp','LineDn'] aligned to df.index.
    - Uses Wilder ATR(atr_period) * multiplier as offset.
    - Fully vectorized where possible; O(n) recursion for the center line by necessity.
    - Preserves length with NaNs during warmup.
    """
    n = len(df)
    idx = df.index

    if n == 0 or atr_period <= 0:
        return pd.DataFrame(
            {
                "LineCenter": pd.Series(index=idx, dtype=float),
                "LineUp": pd.Series(index=idx, dtype=float),
                "LineDn": pd.Series(index=idx, dtype=float),
            }
        )

    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    # True Range
    prev_c = np.empty_like(c)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)

    # Wilder's RMA for ATR
    def _rma(x: np.ndarray, period: int) -> np.ndarray:
        y = np.full_like(x, np.nan, dtype=float)
        if x.size == 0 or period <= 0:
            return y
        # determine first seed index as SMA over first 'period' values (or first window with full data)
        if np.all(np.isfinite(x[:period])):
            seed_idx = period - 1
            y[seed_idx] = x[:period].mean()
        else:
            roll_mean = pd.Series(x).rolling(period, min_periods=period).mean().values
            finite_idx = np.flatnonzero(np.isfinite(roll_mean))
            if finite_idx.size == 0:
                return y
            seed_idx = finite_idx[0]
            y[seed_idx] = roll_mean[seed_idx]
        alpha = 1.0 / period
        for i in range(seed_idx + 1, x.size):
            xi = x[i]
            if np.isfinite(xi):
                yi_1 = y[i - 1]
                if np.isfinite(yi_1):
                    y[i] = yi_1 + (xi - yi_1) * alpha
                else:
                    y[i] = xi
            else:
                y[i] = y[i - 1]
        return y

    atr = _rma(tr, atr_period)
    smooth = atr * float(multiplier)

    lc = np.full(n, np.nan, dtype=float)
    up = np.full(n, np.nan, dtype=float)
    dn = np.full(n, np.nan, dtype=float)

    # find first index where we can seed the center line
    valid = np.isfinite(smooth) & np.isfinite(c)
    if not np.any(valid):
        return pd.DataFrame(
            {
                "LineCenter": pd.Series(lc, index=idx),
                "LineUp": pd.Series(up, index=idx),
                "LineDn": pd.Series(dn, index=idx),
            }
        )

    i0 = int(np.flatnonzero(valid)[0])
    lc[i0] = c[
        i0
    ]  # seed with close; Up/Down remain NaN at seed to respect dependency on prior center

    for i in range(i0 + 1, n):
        prev = lc[i - 1]
        si = smooth[i]
        ci = c[i]

        if not np.isfinite(prev) or not np.isfinite(si) or not np.isfinite(ci):
            lc[i] = prev
            continue

        up[i] = prev + si
        dn[i] = prev - si

        lci = prev
        if ci > up[i]:
            lci = ci - si
        elif ci < dn[i]:
            lci = ci + si
        lc[i] = lci

    return pd.DataFrame(
        {
            "LineCenter": pd.Series(lc, index=idx),
            "LineUp": pd.Series(up, index=idx),
            "LineDn": pd.Series(dn, index=idx),
        }
    )


# === END RangeFilterModified_calc.py ===


# === BEGIN RWIBTF_calc.py ===
def RWIBTF(df: pd.DataFrame, length: int = 2, tf: Optional[str] = None) -> pd.DataFrame:
    """Random Walk Index (BTF-capable).

    Parameters:
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume']
    - length: maximum lookback i for RWI (i from 1..length); default 2
    - tf: optional pandas offset alias (e.g., '15T','1H','1D') to compute on higher timeframe
          and broadcast back to original index. If None, uses current dataframe timeframe.

    Returns:
    - DataFrame with columns ['RWIH','RWIL','TR'], aligned to df.index, NaNs for warmup.
    """
    length = int(max(1, length))

    def _compute_rwi(ohlc: pd.DataFrame, length: int) -> pd.DataFrame:
        high = pd.to_numeric(ohlc["High"], errors="coerce")
        low = pd.to_numeric(ohlc["Low"], errors="coerce")
        close = pd.to_numeric(ohlc["Close"], errors="coerce")

        prev_close = close.shift(1)
        tr = np.maximum(
            high - low, np.maximum((high - prev_close).abs(), (prev_close - low).abs())
        )

        h_cols = []
        l_cols = []
        for i in range(1, length + 1):
            atr_i = tr.rolling(window=i, min_periods=i).mean() / np.sqrt(i + 1)
            atr_i = atr_i.replace(0.0, np.nan)

            num_h = high - low.shift(i)
            num_l = high.shift(i) - low

            ratio_h = num_h / atr_i
            ratio_l = num_l / atr_i

            h_cols.append(ratio_h)
            l_cols.append(ratio_l)

        rwi_h = pd.concat(h_cols, axis=1).max(axis=1, skipna=True).clip(lower=0)
        rwi_l = pd.concat(l_cols, axis=1).max(axis=1, skipna=True).clip(lower=0)

        out = pd.DataFrame({"RWIH": rwi_h, "RWIL": rwi_l, "TR": tr}, index=ohlc.index)
        return out

    if tf is None:
        return _compute_rwi(df[["High", "Low", "Close"]], length).reindex(df.index)

    # Multi-timeframe path
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    # Resample to higher timeframe using OHLCV convention
    ohlc_htf = (
        df.set_index(ts)
        .resample(tf, label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    )

    rwi_htf = _compute_rwi(ohlc_htf[["High", "Low", "Close"]], length)

    # Broadcast back to original index by mapping each timestamp to its period start
    period_keys = ts.dt.floor(tf)
    mapped = rwi_htf.reindex(period_keys.values)
    mapped.index = df.index
    return mapped[["RWIH", "RWIL", "TR"]]


# === END RWIBTF_calc.py ===


# === BEGIN SherifHilo_calc.py ===
def SherifHilo(
    df: pd.DataFrame, period_high: int = 100, period_lows: int = 100
) -> pd.DataFrame:
    """Sherif HiLo indicator.
    Returns a DataFrame with columns: ['LineUp','LineDown','Data','Max','Min'] aligned to df.index.
    - period_high: rolling lookback for HHV of High
    - period_lows: rolling lookback for LLV of Low
    """
    if df.empty:
        return pd.DataFrame(
            index=df.index,
            columns=["LineUp", "LineDown", "Data", "Max", "Min"],
            dtype=float,
        )

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float).to_numpy()

    # Rolling extrema (past window including current bar)
    llv_s = low.rolling(window=period_lows, min_periods=period_lows).min()
    hhv_s = high.rolling(window=period_high, min_periods=period_high).max()

    llv = llv_s.to_numpy()
    hhv = hhv_s.to_numpy()

    n = len(df)
    data = np.full(n, np.nan, dtype=float)
    lineup = np.full(n, np.nan, dtype=float)
    linedown = np.full(n, np.nan, dtype=float)

    start = max(period_high, period_lows) - 1
    prev_default = 0.0  # mimic MT4 buffers' default initialization

    for t in range(start, n):
        prev = data[t - 1] if t > 0 and np.isfinite(data[t - 1]) else prev_default

        # Default carry-over if no direction change
        data[t] = prev

        c = close[t]
        # Compare and switch regime
        if c > prev:
            data[t] = llv[t]
            lineup[t] = llv[t]
            linedown[t] = np.nan
        elif c < prev:
            data[t] = hhv[t]
            linedown[t] = hhv[t]
            lineup[t] = np.nan
        else:
            # equal: keep previous data; no line on this bar
            lineup[t] = np.nan
            linedown[t] = np.nan

    out = pd.DataFrame(
        {
            "LineUp": lineup,
            "LineDown": linedown,
            "Data": data,
            "Max": hhv,
            "Min": llv,
        },
        index=df.index,
    )
    return out


# === END SherifHilo_calc.py ===


# === BEGIN Silence_calc.py ===
def Silence(
    df: pd.DataFrame,
    my_period: int = 12,
    buff_size: int = 96,
    point: float = 0.0001,
    redraw: bool = True,
) -> pd.DataFrame:
    """Silence indicator (Aggressiveness, Volatility) scaled 0..100 (reversed), aligned to df.index.
    - my_period: lookback for bar calculations (default 12)
    - buff_size: lookback for normalization window (default 96)
    - point: instrument tick size (default 0.0001)
    - redraw: if False, last bar equals previous (freeze current bar)
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    # Aggressiveness raw: rolling sum over my_period of sign(candle) * (close - prev_close)
    sgn = np.where(c > o, 1.0, -1.0)
    dclose = c.diff()
    b_contrib = pd.Series(sgn, index=df.index) * dclose
    aggress_raw = b_contrib.rolling(window=my_period, min_periods=my_period).sum() / (
        point * my_period
    )

    # Volatility raw: (highest high - lowest low) over my_period
    hh = h.rolling(window=my_period, min_periods=my_period).max()
    ll = l.rolling(window=my_period, min_periods=my_period).min()
    vol_raw = (hh - ll) / (point * my_period)

    # Normalization over last buff_size values (map [MIN,MAX] -> [100,0])
    def normalize(raw: pd.Series) -> pd.Series:
        roll_max = raw.rolling(window=buff_size, min_periods=buff_size).max()
        roll_min = raw.rolling(window=buff_size, min_periods=buff_size).min()
        denom = roll_max - roll_min
        out = 100.0 * (roll_max - raw) / denom
        out = out.where(denom != 0, 1e7)  # match MQL behavior when range == 0
        return out

    aggress = normalize(aggress_raw)
    vol = normalize(vol_raw)

    out = pd.DataFrame(
        {
            "Aggressiveness": aggress,
            "Volatility": vol,
        },
        index=df.index,
    )

    # Freeze last bar if redraw is False
    if not redraw and len(out) >= 2:
        out.iloc[-1] = out.iloc[-2]

    return out


# === END Silence_calc.py ===


# === BEGIN Sinewma_calc.py ===
def Sinewma(df: pd.DataFrame, length: int = 20, price: int = 0) -> pd.Series:
    """Return the Sine Weighted Moving Average (SineWMA) aligned to df.index.
    - length: window size (default 20, must be >=1)
    - price: applied price mapping:
        0 Close, 1 Open, 2 High, 3 Low, 4 Median(HL/2), 5 Typical(HLC/3), 6 Weighted(HLCC/4)
    Vectorized with numpy; preserves length with NaNs for warmup or incomplete windows; handles NaNs.
    """
    if length < 1:
        raise ValueError("length must be >= 1")

    # Applied price selection
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    if price == 0:
        x = c
    elif price == 1:
        x = o
    elif price == 2:
        x = h
    elif price == 3:
        x = l
    elif price == 4:
        x = (h + l) / 2.0
    elif price == 5:
        x = (h + l + c) / 3.0
    elif price == 6:
        x = (h + l + 2.0 * c) / 4.0
    else:
        x = c  # fallback to Close

    x_arr = x.to_numpy(dtype=float)
    n = x_arr.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=df.index, name=f"SineWMA_{length}")

    # Sine weights
    w = np.sin(np.pi * (np.arange(length) + 1.0) / (length + 1.0))
    w_sum = w.sum()
    w_rev = w[::-1]

    # Handle NaNs: require full window of valid values
    valid = np.isfinite(x_arr).astype(float)
    x_filled = np.where(np.isfinite(x_arr), x_arr, 0.0)

    num_full = np.convolve(x_filled, w_rev, mode="full")
    cnt_full = np.convolve(valid, np.ones(length, dtype=float), mode="full")

    end_idx_start = length - 1
    end_idx_stop = n  # inclusive of last index n-1
    vals = num_full[end_idx_start:end_idx_stop] / w_sum
    cnts = cnt_full[end_idx_start:end_idx_stop]

    out[end_idx_start:end_idx_stop] = np.where(cnts == length, vals, np.nan)

    return pd.Series(out, index=df.index, name=f"SineWMA_{length}")


# === END Sinewma_calc.py ===


# === BEGIN SmoothedMomentum_calc.py ===
def SmoothedMomentum(
    df: pd.DataFrame,
    momentum_length: int = 12,
    use_smoothing: bool = True,
    smoothing_method: int = 0,
    smoothing_length: int = 20,
    price: int = 0,
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.

    Parameters:
    - momentum_length: int = 12
    - use_smoothing: bool = True
    - smoothing_method: int = 0  (0=SMA, 1=EMA, 2=SMMA/Wilder, 3=LWMA)
    - smoothing_length: int = 20
    - price: int = 0  (0=Close, 1=Open, 2=High, 3=Low, 4=Median, 5=Typical, 6=Weighted)
    """
    n = max(int(momentum_length), 1)
    m = max(int(smoothing_length), 1)

    # Applied price selection
    if price == 0:
        p = df["Close"].astype(float)
    elif price == 1:
        p = df["Open"].astype(float)
    elif price == 2:
        p = df["High"].astype(float)
    elif price == 3:
        p = df["Low"].astype(float)
    elif price == 4:
        p = (df["High"] + df["Low"]) / 2.0
    elif price == 5:
        p = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif price == 6:
        p = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        p = df["Close"].astype(float)
    p = p.astype(float)

    # Momentum: 100 * price / price.shift(n); avoid div by zero
    denom = p.shift(n)
    denom = denom.where(denom != 0, np.nan)
    momentum = 100.0 * p / denom

    # Smoothing
    if use_smoothing:
        if m <= 1:
            sm = momentum.copy()
        else:
            if smoothing_method == 0:  # SMA
                sm = momentum.rolling(window=m, min_periods=m).mean()
            elif smoothing_method == 1:  # EMA
                sm = momentum.ewm(span=m, adjust=False).mean()
            elif smoothing_method == 2:  # SMMA (Wilder's RMA)
                sm = momentum.ewm(alpha=1.0 / m, adjust=False).mean()
            elif smoothing_method == 3:  # LWMA
                w = np.arange(1, m + 1, dtype=float)
                w_sum = w.sum()
                sm = momentum.rolling(window=m, min_periods=m).apply(
                    lambda a: np.dot(a, w) / w_sum, raw=True
                )
            else:  # default to SMA
                sm = momentum.rolling(window=m, min_periods=m).mean()
    else:
        sm = momentum.copy()

    out = pd.DataFrame({"SM": sm, "Momentum": momentum}, index=df.index)
    return out


# === END SmoothedMomentum_calc.py ===


# === BEGIN Smoothstep_calc.py ===
def Smoothstep(
    df: pd.DataFrame, period: int = 32, price: str = "close"
) -> pd.DataFrame:
    """
    SmoothStep indicator (mladen, 2022) converted to pandas/numpy.

    Parameters
    - df: DataFrame with ['Timestamp','Open','High','Low','Close','Volume']
    - period: rolling window length (default 32, coerced to >=1)
    - price: one of {'close','open','high','low','median','typical','weighted','lowhigh'}

    Returns
    - DataFrame with columns:
        'SmoothStep'        : main smoothstep line
        'SmoothStepDownA'   : alternating down-trend overlay segment A
        'SmoothStepDownB'   : alternating down-trend overlay segment B
      Aligned to df.index with NaNs for warmup.
    """
    period = max(int(period), 1)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    p = c.copy()
    ph = p.copy()
    pl = p.copy()

    mode = str(price).lower()
    if mode == "open":
        p = ph = pl = o
    elif mode == "high":
        p = ph = pl = h
    elif mode == "low":
        p = ph = pl = l
    elif mode == "median":
        p = ph = pl = (h + l) / 2.0
    elif mode == "typical":
        p = ph = pl = (h + l + c) / 3.0
    elif mode == "weighted":
        p = ph = pl = (h + l + 2.0 * c) / 4.0
    elif mode == "lowhigh":
        p = c
        ph = h
        pl = l
    else:  # "close" or default
        p = ph = pl = c

    low_roll = pl.rolling(window=period, min_periods=period).min()
    high_roll = ph.rolling(window=period, min_periods=period).max()
    denom = high_roll - low_roll

    raw = (p - low_roll) / denom
    # If denom == 0 and not NaN -> 0; if denom is NaN -> NaN
    raw = raw.where(denom != 0, 0.0)

    val = raw * raw * (3.0 - 2.0 * raw)

    delta = val.diff()
    valc = pd.Series(
        np.where(delta > 0, 1.0, np.where(delta < 0, 2.0, np.nan)), index=df.index
    )
    # Do not carry trend through NaN warmup; ffill afterwards
    valc = valc.ffill()

    down_mask = (valc == 2.0) & val.notna()

    run_start = down_mask & (~down_mask.shift(1, fill_value=False))
    rid = run_start.cumsum()
    odd = rid % 2 == 1

    a_mask = down_mask & odd
    b_mask = down_mask & (~odd)

    # Include previous bar at the start of each down run (to mimic iPlotPoint connectivity)
    a_mask = a_mask | (run_start & odd).shift(-1, fill_value=False)
    b_mask = b_mask | (run_start & (~odd)).shift(-1, fill_value=False)

    out = pd.DataFrame(index=df.index)
    out["SmoothStep"] = val
    out["SmoothStepDownA"] = val.where(a_mask)
    out["SmoothStepDownB"] = val.where(b_mask)
    return out


# === END Smoothstep_calc.py ===


# === BEGIN StiffnessIndicator_calc.py ===
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
                raise ValueError(
                    "Unknown MA method code. Use 0:SMA, 1:EMA, 2:SMMA, 3:LWMA."
                )
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
            raise ValueError(
                "Unknown MA method. Use 'sma','ema','smma','lwma' or 0..3."
            )
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
            return s.rolling(window=n, min_periods=n).apply(
                lambda x: np.dot(x, w) / w_sum, raw=True
            )
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


# === END StiffnessIndicator_calc.py ===


# === BEGIN Supertrend_calc.py ===
def Supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Jason Robinson (2008) SuperTrend.
    Returns a DataFrame with columns:
      - 'Supertrend_Up': line when in uptrend (else NaN)
      - 'Supertrend_Down': line when in downtrend (else NaN)
      - 'Supertrend': merged primary line (Up or Down)
      - 'Trend': +1 for uptrend, -1 for downtrend (NaN during warmup)
    Vectorized where possible; uses Wilder ATR (ewm alpha=1/period). Preserves index and length with NaNs in warmup.
    """
    if df.empty:
        return pd.DataFrame(
            index=df.index,
            columns=["Supertrend_Up", "Supertrend_Down", "Supertrend", "Trend"],
            dtype="float64",
        )

    h = df["High"].to_numpy(dtype="float64")
    l = df["Low"].to_numpy(dtype="float64")
    c = df["Close"].to_numpy(dtype="float64")

    # True Range
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    if len(c) > 1:
        prev_c[1:] = c[:-1]
    tr1 = h - l
    tr2 = np.abs(h - prev_c)
    tr3 = np.abs(l - prev_c)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)
    tr_series = pd.Series(tr, index=df.index)

    # Wilder ATR via ewm(alpha=1/period), with warmup NaNs
    atr = (
        tr_series.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period)
        .mean()
        .to_numpy()
    )

    median = (h + l) / 2.0
    up_base = median + multiplier * atr
    dn_base = median - multiplier * atr

    n = len(df)
    up = np.full(n, np.nan, dtype="float64")
    dn = np.full(n, np.nan, dtype="float64")
    trend = np.full(n, np.nan, dtype="float64")
    line_up = np.full(n, np.nan, dtype="float64")
    line_dn = np.full(n, np.nan, dtype="float64")

    started = False
    up_prev = np.nan
    dn_prev = np.nan
    trend_prev = 1  # default initial trend as in MQL

    for i in range(n):
        ub = up_base[i]
        db = dn_base[i]
        ci = c[i]

        # Require current bar values to proceed
        if np.isnan(ub) or np.isnan(db) or np.isnan(ci) or np.isnan(median[i]):
            continue

        if not started:
            # Seed with first available values
            trend_prev = 1
            up_prev = ub
            dn_prev = db
            trend[i] = 1
            dn[i] = db
            up[i] = ub
            line_up[i] = dn[i]
            started = True
            continue

        change = 0
        if ci > up_prev:
            curr_trend = 1
            if trend_prev == -1:
                change = 1
        elif ci < dn_prev:
            curr_trend = -1
            if trend_prev == 1:
                change = 1
        else:
            curr_trend = trend_prev
            change = 0

        flag = 1 if (curr_trend < 0 and trend_prev > 0) else 0
        flagh = 1 if (curr_trend > 0 and trend_prev < 0) else 0

        up_i = ub
        dn_i = db

        if curr_trend > 0 and dn_i < dn_prev:
            dn_i = dn_prev
        if curr_trend < 0 and up_i > up_prev:
            up_i = up_prev

        if flag == 1:
            up_i = ub
        if flagh == 1:
            dn_i = db

        if curr_trend == 1:
            line_up[i] = dn_i
            if change == 1 and i - 1 >= 0:
                line_up[i - 1] = line_dn[i - 1]
        else:
            line_dn[i] = up_i
            if change == 1 and i - 1 >= 0:
                line_dn[i - 1] = line_up[i - 1]

        trend[i] = curr_trend
        up[i] = up_i
        dn[i] = dn_i
        up_prev = up_i
        dn_prev = dn_i
        trend_prev = curr_trend

    supertrend = np.where(np.isnan(line_dn), line_up, line_dn)

    out = pd.DataFrame(
        {
            "Supertrend_Up": line_up,
            "Supertrend_Down": line_dn,
            "Supertrend": supertrend,
            "Trend": trend,
        },
        index=df.index,
    )
    return out


# === END Supertrend_calc.py ===


# === BEGIN T3MA_calc.py ===
def T3MA(
    df: pd.DataFrame, length: int = 10, b: float = 0.88, price: int = 0
) -> pd.Series:
    """Return the T3 moving average as a pandas Series aligned to df.index; vectorized, handles NaNs, stable defaults.

    Parameters:
    - length: smoothing length (int > 0), default 10
    - b: T3 'b' coefficient (float), default 0.88
    - price: applied price selector (int), default 0
        0=Close, 1=Open, 2=High, 3=Low, 4=Median(HL2), 5=Typical(HLC3), 6=Weighted(HLCC4)
    """
    if length <= 0:
        raise ValueError("length must be a positive integer")
    cols = df.columns
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(cols)):
        raise ValueError(f"df must contain columns: {sorted(required)}")

    if price == 0:
        pr = df["Close"]
    elif price == 1:
        pr = df["Open"]
    elif price == 2:
        pr = df["High"]
    elif price == 3:
        pr = df["Low"]
    elif price == 4:
        pr = (df["High"] + df["Low"]) / 2.0
    elif price == 5:
        pr = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif price == 6:
        pr = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        raise ValueError("price must be an int in {0,1,2,3,4,5,6}")

    w1 = 4.0 / (3.0 + float(length))

    e1 = pr.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e2 = e1.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e3 = e2.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e4 = e3.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e5 = e4.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()
    e6 = e5.ewm(alpha=w1, adjust=False, min_periods=1, ignore_na=True).mean()

    b2 = b * b
    b3 = b2 * b
    c1 = -b3
    c2 = 3.0 * (b2 + b3)
    c3 = -3.0 * (2.0 * b2 + b + b3)
    c4 = 1.0 + 3.0 * b + b3 + 3.0 * b2

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    t3.name = "T3MA"
    return t3


# === END T3MA_calc.py ===


# === BEGIN TetherLine_calc.py ===
def TetherLine(df: pd.DataFrame, length: int = 55) -> pd.DataFrame:
    """Return all indicator buffers as columns aligned to df.index:
    - AboveCenter: midpoint of rolling [Highest High + Lowest Low]/2 when Close > midpoint
    - BelowCenter: midpoint when Close < midpoint
    - ArrowUp, ArrowDown: placeholders (NaN) per original script
    """
    length = max(1, int(length))
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    hh = high.rolling(window=length, min_periods=length).max()
    ll = low.rolling(window=length, min_periods=length).min()
    center = (hh + ll) / 2.0

    above = pd.Series(
        np.where(close > center, center, np.nan), index=df.index, name="AboveCenter"
    )
    below = pd.Series(
        np.where(close < center, center, np.nan), index=df.index, name="BelowCenter"
    )

    up = pd.Series(np.nan, index=df.index, name="ArrowUp")
    down = pd.Series(np.nan, index=df.index, name="ArrowDown")

    return pd.DataFrame(
        {"AboveCenter": above, "BelowCenter": below, "ArrowUp": up, "ArrowDown": down}
    )


# === END TetherLine_calc.py ===


# === BEGIN Theturtletradingchannel_calc.py ===
def TheTurtleTradingChannel(
    df: pd.DataFrame,
    trade_period: int = 20,
    stop_period: int = 10,
    strict: bool = False,
) -> pd.DataFrame:
    """Return Turtle Trading Channel lines aligned to df.index.

    Parameters:
    - trade_period: Donchian channel period for trading signals (default 20)
    - stop_period: Donchian channel period for exit signals (default 10)
    - strict: If True, use intrabar HIGH/LOW breakouts; otherwise use CLOSE-based (except last bar)
    """
    if df.empty:
        return pd.DataFrame(
            index=df.index,
            columns=[
                "UpperLine",
                "LowerLine",
                "LongsStopLine",
                "ShortsStopLine",
                "BullChange",
                "BearChange",
            ],
            dtype=float,
        )

    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    C = df["Close"].astype(float)
    n = len(df)

    # Donchian channels excluding current bar
    rhigh = H.rolling(trade_period, min_periods=trade_period).max().shift(1)
    rlow = L.rolling(trade_period, min_periods=trade_period).min().shift(1)
    shigh = H.rolling(stop_period, min_periods=stop_period).max().shift(1)
    slow = L.rolling(stop_period, min_periods=stop_period).min().shift(1)

    # Gating close-based signals on all bars except the last (i > 0 in MQL)
    not_last = np.ones(n, dtype=bool)
    if n > 0:
        not_last[-1] = False
    not_last_s = pd.Series(not_last, index=df.index)

    if strict:
        up_break = ((C > rhigh) & not_last_s) | (H > rhigh)
        dn_break = ((C < rlow) & not_last_s) | (L < rlow)
    else:
        up_break = (C > rhigh) & not_last_s
        dn_break = (C < rlow) & not_last_s

    # Prioritize up over down on simultaneous triggers (as in MQL's if/else if)
    sig = np.zeros(n, dtype=float)
    up_idx = up_break.fillna(False).to_numpy()
    dn_idx = dn_break.fillna(False).to_numpy()
    sig[up_idx] = 1.0
    sig[~up_idx & dn_idx] = -1.0

    # Regime/state: last non-zero signal carried forward
    state = pd.Series(np.where(sig != 0.0, sig, np.nan), index=df.index).ffill()

    prev_state = state.shift(1)

    # Lines
    upper_line = pd.Series(
        np.where(state.eq(1.0), rlow.to_numpy(), np.nan), index=df.index
    )
    lower_line = pd.Series(
        np.where(state.eq(-1.0), rhigh.to_numpy(), np.nan), index=df.index
    )
    longs_stop = pd.Series(
        np.where(state.eq(1.0), slow.to_numpy(), np.nan), index=df.index
    )
    shorts_stop = pd.Series(
        np.where(state.eq(-1.0), shigh.to_numpy(), np.nan), index=df.index
    )

    # Trend change markers
    bull_change = pd.Series(
        np.where(
            (sig == 1.0) & (~prev_state.eq(1.0).fillna(True)), rlow.to_numpy(), np.nan
        ),
        index=df.index,
    )
    bear_change = pd.Series(
        np.where(
            (sig == -1.0) & (~prev_state.eq(-1.0).fillna(True)),
            rhigh.to_numpy(),
            np.nan,
        ),
        index=df.index,
    )

    out = pd.DataFrame(
        {
            "UpperLine": upper_line,
            "LowerLine": lower_line,
            "LongsStopLine": longs_stop,
            "ShortsStopLine": shorts_stop,
            "BullChange": bull_change,
            "BearChange": bear_change,
        },
        index=df.index,
    )
    return out


# === END Theturtletradingchannel_calc.py ===


# === BEGIN TII_calc.py ===
def TII(
    df: pd.DataFrame,
    length: int = 30,
    ma_length: int = 60,
    ma_method: int = 0,  # 0=SMA, 1=EMA, 2=SMMA, 3=LWMA
    price: int = 0,  # 0=Close,1=Open,2=High,3=Low,4=Median,5=Typical,6=Weighted
) -> pd.Series:
    """Trend Intensity Index (TII) as in MQL4 TII.mq4.
    Returns a Series aligned to df.index with NaNs for warmup.
    """
    if length <= 0 or ma_length <= 0:
        return pd.Series(np.nan, index=df.index, name="TII")

    # Applied price
    if price == 0:
        pr = df["Close"].astype(float)
    elif price == 1:
        pr = df["Open"].astype(float)
    elif price == 2:
        pr = df["High"].astype(float)
    elif price == 3:
        pr = df["Low"].astype(float)
    elif price == 4:
        pr = ((df["High"] + df["Low"]) / 2.0).astype(float)  # Median Price (HL/2)
    elif price == 5:
        pr = ((df["High"] + df["Low"] + df["Close"]) / 3.0).astype(
            float
        )  # Typical Price (HLC/3)
    elif price == 6:
        pr = ((df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0).astype(
            float
        )  # Weighted Price (HLCC/4)
    else:
        pr = df["Close"].astype(float)

    # Moving average helpers
    def _sma(x: pd.Series, p: int) -> pd.Series:
        return x.rolling(window=p, min_periods=p).mean()

    def _ema(x: pd.Series, p: int) -> pd.Series:
        y = x.ewm(span=p, adjust=False).mean()
        mask = x.notna().rolling(window=p, min_periods=p).count() >= p
        return y.where(mask)

    def _smma(x: pd.Series, p: int) -> pd.Series:
        # Wilder/Smoothed MA with alpha=1/p; mask first p-1 values
        y = x.ewm(alpha=1.0 / p, adjust=False).mean()
        mask = x.notna().rolling(window=p, min_periods=p).count() >= p
        return y.where(mask)

    def _lwma(x: pd.Series, p: int) -> pd.Series:
        w = np.arange(1, p + 1, dtype=float)
        w_sum = w.sum()

        def func(a: np.ndarray) -> float:
            if np.isnan(a).any():
                return np.nan
            return float(np.dot(a, w) / w_sum)

        return x.rolling(window=p, min_periods=p).apply(func, raw=True)

    def _ma(x: pd.Series, p: int, method: int) -> pd.Series:
        if method == 0:
            return _sma(x, p)
        elif method == 1:
            return _ema(x, p)
        elif method == 2:
            return _smma(x, p)
        elif method == 3:
            return _lwma(x, p)
        else:
            return _sma(x, p)

    ma = _ma(pr, ma_length, ma_method)

    diff = pr - ma
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)

    pos = up.rolling(window=length, min_periods=length).mean()
    neg = down.rolling(window=length, min_periods=length).mean()

    den = pos + neg
    tii = (100.0 * pos / den).where(den != 0)
    tii.name = "TII"
    return tii


# === END TII_calc.py ===


# === BEGIN TopBottomNR_calc.py ===
def TopBottomNR(df: pd.DataFrame, per: int = 14) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index. Vectorized, handles NaNs, stable defaults."""
    if per < 1:
        raise ValueError("per must be >= 1")

    low = df["Low"]
    high = df["High"]

    prev_low_min = low.shift(1).rolling(window=per, min_periods=per).min()
    prev_high_max = high.shift(1).rolling(window=per, min_periods=per).max()

    valid_long = prev_low_min.notna()
    valid_short = prev_high_max.notna()

    reset_long = (low < prev_low_min).fillna(False)
    reset_short = (high > prev_high_max).fillna(False)

    def _run_length(reset: pd.Series, valid: pd.Series) -> pd.Series:
        n = len(reset)
        idx = np.arange(n, dtype=np.int64)
        reset_arr = reset.to_numpy(dtype=bool)
        valid_arr = valid.to_numpy(dtype=bool)
        boundary = reset_arr | (~valid_arr)
        last_boundary = np.maximum.accumulate(np.where(boundary, idx, -1))
        res = (idx - last_boundary).astype(float)
        res[~valid_arr] = np.nan
        return pd.Series(res, index=reset.index)

    long_signal = _run_length(reset_long, valid_long)
    short_signal = _run_length(reset_short, valid_short)

    return pd.DataFrame(
        {"LongSignal": long_signal, "ShortSignal": short_signal}, index=df.index
    )


# === END TopBottomNR_calc.py ===


# === BEGIN TP_calc.py ===
def TP(df: pd.DataFrame, length: int = 14, show_updn: bool = False) -> pd.DataFrame:
    """Advance Trend Pressure (TP), Up and Dn lines.
    - Returns DataFrame with columns ['TP','Up','Dn'] aligned to df.index.
    - Vectorized; uses rolling sums over 'length' bars.
    - Up/Dn columns are NaN if show_updn is False.
    """
    length = int(max(1, length))

    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    # Per-bar contributions (exclude bars where Close == Open -> 0 contribution)
    up_contrib = c - l
    dn_contrib = h - c

    eq_mask = c == o
    up_contrib = up_contrib.where(~eq_mask, 0.0)
    dn_contrib = dn_contrib.where(~eq_mask, 0.0)

    # Preserve NaNs if any input OHLC is NaN
    valid_mask = o.notna() & h.notna() & l.notna() & c.notna()
    up_contrib = up_contrib.where(valid_mask, np.nan)
    dn_contrib = dn_contrib.where(valid_mask, np.nan)

    up = up_contrib.rolling(window=length, min_periods=length).sum()
    dn = dn_contrib.rolling(window=length, min_periods=length).sum()
    tp = up - dn

    if not show_updn:
        up_out = pd.Series(np.nan, index=df.index)
        dn_out = pd.Series(np.nan, index=df.index)
    else:
        up_out = up
        dn_out = dn

    return pd.DataFrame({"TP": tp, "Up": up_out, "Dn": dn_out}, index=df.index)


# === END TP_calc.py ===


# === BEGIN TRENDAKKAM_calc.py ===
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
        return pd.DataFrame(
            {"TrStop": pd.Series(dtype=float), "ATR": pd.Series(dtype=float)}
        )

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    open_ = pd.to_numeric(df["Open"], errors="coerce")

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    # Wilder's ATR (RMA with alpha = 1/period); NaN until warmup
    atr = tr.ewm(
        alpha=(1.0 / float(akk_range)), adjust=False, min_periods=akk_range
    ).mean()

    # EMA of ATR on array (iMAOnArray with MODE_EMA). Period 1 => identity.
    if ima_range <= 1:
        ma_atr = atr.copy()
    else:
        ma_atr = atr.ewm(span=ima_range, adjust=False, min_periods=ima_range).mean()

    # DeltaStop series
    if mode == 0:
        delta = ma_atr * float(akk_factor)
    else:
        delta = pd.Series(
            float(delta_price) * float(point), index=df.index, dtype=float
        )

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

        if not (
            np.isfinite(prev_T)
            and np.isfinite(prev_O)
            and np.isfinite(curr_O)
            and np.isfinite(curr_D)
        ):
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
                tr_rev[j] = (
                    (curr_O - curr_D) if (curr_O > prev_T) else (curr_O + curr_D)
                )

    trstop = pd.Series(tr_rev[::-1], index=df.index, name="TrStop")

    out = pd.DataFrame({"TrStop": trstop, "ATR": atr.rename("ATR")}, index=df.index)

    return out


# === END TRENDAKKAM_calc.py ===


# === BEGIN Trendcontinuation2_calc.py ===
def Trendcontinuation2(
    df: pd.DataFrame, n: int = 20, t3_period: int = 5, b: float = 0.618
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Vectorized implementation of 'Trend continuation factor 2' (MT4) using pandas only.
    Params:
      - n: lookback length for change aggregation (default 20). Effective window is n+1.
      - t3_period: T3 smoothing period (default 5).
      - b: T3 smoothing factor (default 0.618).
    """
    close = df["Close"].astype(float)

    # 1) One-bar changes split into positive/negative components (strict >0 / <0 as in MQL4)
    delta = close.diff()
    pos = delta.where(
        delta > 0, 0.0
    )  # positive changes; zeros elsewhere; NaNs preserved
    neg = (-delta).where(
        delta < 0, 0.0
    )  # magnitude of negative changes; zeros elsewhere

    # 2) CF_p / CF_n: cumulative sums within consecutive runs of pos/neg, resetting on non-pos/non-neg
    mask_pos = pos > 0
    grp_pos = (~mask_pos).cumsum()
    CF_p = pos.fillna(0.0).groupby(grp_pos).cumsum()
    CF_p = CF_p.where(mask_pos, 0.0)

    mask_neg = neg > 0
    grp_neg = (~mask_neg).cumsum()
    CF_n = neg.fillna(0.0).groupby(grp_neg).cumsum()
    CF_n = CF_n.where(mask_neg, 0.0)

    # 3) Rolling sums over [i-n, i] => window length n+1
    win = max(int(n), 0) + 1
    ch_p = pos.rolling(window=win, min_periods=win).sum()
    ch_n = neg.rolling(window=win, min_periods=win).sum()
    cff_p = CF_p.rolling(window=win, min_periods=win).sum()
    cff_n = CF_n.rolling(window=win, min_periods=win).sum()

    k_p = ch_p - cff_n
    k_n = ch_n - cff_p

    # 4) T3 smoothing (Tillson T3) with MetaTrader-style alpha adjustment
    b2 = b * b
    b3 = b2 * b
    c1 = -b3
    c2 = 3.0 * (b2 + b3)
    c3 = -3.0 * (2.0 * b2 + b + b3)
    c4 = 1.0 + 3.0 * b + b3 + 3.0 * b2

    n1 = max(int(t3_period), 1)
    n1 = 1.0 + 0.5 * (n1 - 1.0)
    alpha = 2.0 / (n1 + 1.0)

    def t3_filter(x: pd.Series) -> pd.Series:
        e1 = x.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e2 = e1.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e3 = e2.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e4 = e3.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e5 = e4.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        e6 = e5.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    out_pos = t3_filter(k_p)
    out_neg = t3_filter(k_n)

    return pd.DataFrame(
        {
            "TrendContinuation2_Pos": out_pos,
            "TrendContinuation2_Neg": out_neg,
        },
        index=df.index,
    )


# === END Trendcontinuation2_calc.py ===


# === BEGIN TrendDirectionForceIndexSmoothed4_calc.py ===
def TrendDirectionForceIndexSmoothed4(
    df: pd.DataFrame,
    trend_period: int = 20,
    trend_method: str = "ema",
    price: str = "close",
    trigger_up: float = 0.05,
    trigger_down: float = -0.05,
    smooth_length: float = 5.0,
    smooth_phase: float = 0.0,
    color_change_on_zero_cross: bool = False,
    point: float = 1.0,
) -> pd.DataFrame:
    """
    Return ALL lines as columns with clear names; aligned to df.index.
    Vectorized where possible; recursive parts are looped as needed. Handles NaNs.
    Supported trend_method: sma, ema, dsema, dema, tema, smma, lwma, pwma, vwma, hull, tma, sine, mcg, zlma, lead, ssm, smoo, linr, ilinr, ie2, nlma
    Supported price: close, open, high, low, median, typical, weighted, average, medianb, tbiased,
                     haclose, haopen, hahigh, halow, hamedian, hatypical, haweighted, haaverage, hamedianb, hatbiased
    """
    o = pd.Series(df["Open"].to_numpy(dtype=float), index=df.index)
    h = pd.Series(df["High"].to_numpy(dtype=float), index=df.index)
    l = pd.Series(df["Low"].to_numpy(dtype=float), index=df.index)
    c = pd.Series(df["Close"].to_numpy(dtype=float), index=df.index)
    v = pd.Series(df["Volume"].to_numpy(dtype=float), index=df.index)
    n = len(df)

    # Price selection
    def heikin_ashi(Open, High, Low, Close):
        ha_close = (Open + High + Low + Close) / 4.0
        ha_open = np.empty(n)
        ha_high = np.empty(n)
        ha_low = np.empty(n)
        ha_open[:] = np.nan
        ha_high[:] = np.nan
        ha_low[:] = np.nan
        for i in range(n):
            if i == 0 or not np.isfinite(ha_open[i - 1]):
                prev_ha_open = (Open.iloc[i] + Close.iloc[i]) / 2.0
                prev_ha_close = (
                    Open.iloc[i] + High.iloc[i] + Low.iloc[i] + Close.iloc[i]
                ) / 4.0
            else:
                prev_ha_open = ha_open[i - 1]
                prev_ha_close = ha_close.iloc[i - 1]
            hao = 0.5 * (prev_ha_open + prev_ha_close)
            hac = ha_close.iloc[i]
            hah = max(High.iloc[i], hao, hac)
            hal = min(Low.iloc[i], hao, hac)
            ha_open[i] = hao
            ha_high[i] = hah
            ha_low[i] = hal
        ha = pd.DataFrame(
            {
                "ha_open": pd.Series(ha_open, index=Open.index),
                "ha_high": pd.Series(ha_high, index=Open.index),
                "ha_low": pd.Series(ha_low, index=Open.index),
                "ha_close": ha_close,
            }
        )
        return ha

    def get_price(price_key: str) -> pd.Series:
        key = price_key.lower()
        if key.startswith("ha"):
            ha = heikin_ashi(o, h, l, c)
            if key == "haclose":
                p = ha["ha_close"]
            elif key == "haopen":
                p = ha["ha_open"]
            elif key == "hahigh":
                p = ha["ha_high"]
            elif key == "halow":
                p = ha["ha_low"]
            elif key == "hamedian":
                p = (ha["ha_high"] + ha["ha_low"]) / 2.0
            elif key == "hatypical":
                p = (ha["ha_high"] + ha["ha_low"] + ha["ha_close"]) / 3.0
            elif key == "haweighted":
                p = (ha["ha_high"] + ha["ha_low"] + 2.0 * ha["ha_close"]) / 4.0
            elif key == "haaverage":
                p = (
                    ha["ha_high"] + ha["ha_low"] + ha["ha_close"] + ha["ha_open"]
                ) / 4.0
            elif key == "hamedianb":
                p = (ha["ha_open"] + ha["ha_close"]) / 2.0
            elif key == "hatbiased":
                cond = ha["ha_close"] > ha["ha_open"]
                p = pd.Series(
                    np.where(
                        cond,
                        (ha["ha_high"] + ha["ha_close"]) / 2.0,
                        (ha["ha_low"] + ha["ha_close"]) / 2.0,
                    ),
                    index=ha.index,
                )
            else:
                p = c.copy()
        else:
            if key == "close":
                p = c
            elif key == "open":
                p = o
            elif key == "high":
                p = h
            elif key == "low":
                p = l
            elif key == "median":
                p = (h + l) / 2.0
            elif key == "typical":
                p = (h + l + c) / 3.0
            elif key == "weighted":
                p = (h + l + 2.0 * c) / 4.0
            elif key == "average":
                p = (h + l + c + o) / 4.0
            elif key == "medianb":
                p = (o + c) / 2.0
            elif key == "tbiased":
                p = pd.Series(
                    np.where(c > o, (h + c) / 2.0, (l + c) / 2.0), index=c.index
                )
            else:
                p = c
        return p.astype(float)

    x = get_price(price)

    # Rolling weighted helpers
    def _lwma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        w = np.arange(1.0, length + 1.0)
        denom = w.sum()
        return series.rolling(length, min_periods=length).apply(
            lambda a: float(np.dot(a, w) / denom), raw=True
        )

    def _pwma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        w = np.arange(1.0, length + 1.0) ** 2
        denom = w.sum()
        return series.rolling(length, min_periods=length).apply(
            lambda a: float(np.dot(a, w) / denom), raw=True
        )

    def _sma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        return series.rolling(length, min_periods=length).mean()

    def _ema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        return series.ewm(span=length, adjust=False).mean()

    def _smma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr, dtype=float)
        out[:] = np.nan
        alpha = 1.0 / float(length)
        prev = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if not np.isfinite(prev):
                prev = xi
                out[i] = prev
            else:
                prev = prev + alpha * (xi - prev)
                out[i] = prev
        return pd.Series(out, index=series.index)

    def _vwma(series: pd.Series, vol: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        num = (series * vol).rolling(length, min_periods=length).sum()
        den = vol.rolling(length, min_periods=length).sum()
        return num / den

    def _tema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        e1 = _ema(series, length)
        e2 = _ema(e1, length)
        e3 = _ema(e2, length)
        return 3.0 * (e1 - e2) + e3

    def _dema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        e1 = _ema(series, length)
        e2 = _ema(e1, length)
        return 2.0 * e1 - e2

    def _dsema(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        L = max(length, 1.0)
        alpha = 2.0 / (1.0 + np.sqrt(L))
        arr = series.to_numpy(copy=True)
        e1 = np.empty_like(arr)
        e2 = np.empty_like(arr)
        e1[:] = np.nan
        e2[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                e1[i] = np.nan
                e2[i] = np.nan
                continue
            if i == 0 or not np.isfinite(e1[i - 1]):
                e1[i] = xi
                e2[i] = xi
            else:
                e1[i] = e1[i - 1] + alpha * (xi - e1[i - 1])
                e2[i] = e2[i - 1] + alpha * (e1[i] - e2[i - 1])
        return pd.Series(e2, index=series.index)

    def _tma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        n1 = int(np.ceil((length + 1) / 2.0))
        n2 = int(np.floor((length + 1) / 2.0))
        return _sma(_sma(series, n1), n2)

    def _hull(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        n1 = int(np.floor(length / 2.0))
        n2 = int(length)
        n3 = int(max(np.floor(np.sqrt(length)), 1))
        w1 = _lwma(series, n1)
        w2 = _lwma(series, n2)
        diff = 2.0 * w1 - w2
        return _lwma(diff, n3)

    def _sinewma(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        k = np.arange(1.0, length + 1.0)
        w = np.sin(np.pi * k / (length + 1.0))
        denom = w.sum()
        return series.rolling(length, min_periods=length).apply(
            lambda a: float(np.dot(a, w) / denom), raw=True
        )

    def _mcg(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i == 0 or not np.isfinite(out[i - 1]) or out[i - 1] == 0:
                out[i] = xi
            else:
                denom = length * (xi / out[i - 1]) ** 4 / 2.0
                out[i] = out[i - 1] + (xi - out[i - 1]) / denom
        return pd.Series(out, index=series.index)

    def _zlma(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        L = int((length - 1.0) // 2)
        alpha = 2.0 / (1.0 + length)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i <= L:
                out[i] = xi
            else:
                if int(length) % 2 == 0:
                    median = (arr[i - L] + arr[i - L - 1]) / 2.0
                else:
                    median = arr[i - L]
                prev = out[i - 1] if np.isfinite(out[i - 1]) else xi
                out[i] = prev + alpha * (2.0 * xi - median - prev)
        return pd.Series(out, index=series.index)

    def _leader(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        alpha = 2.0 / (length + 1.0)
        arr = series.to_numpy(copy=True)
        e1 = np.empty_like(arr)
        e2 = np.empty_like(arr)
        e1[:] = np.nan
        e2[:] = np.nan
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i == 0 or not np.isfinite(e1[i - 1]):
                e1[i] = xi
                e2[i] = xi
            else:
                e1[i] = e1[i - 1] + alpha * (xi - e1[i - 1])
                e2[i] = e2[i - 1] + alpha * (xi - e1[i] - e2[i - 1])
            out[i] = e1[i] + e2[i]
        return pd.Series(out, index=series.index)

    def _ssm(series: pd.Series, length: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2.0 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1.0 - c2 - c3
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i < 2 or not np.isfinite(out[i - 1]) or not np.isfinite(out[i - 2]):
                out[i] = xi
            else:
                out[i] = (
                    c1 * (xi + arr[i - 1]) / 2.0 + c2 * out[i - 1] + c3 * out[i - 2]
                )
        return pd.Series(out, index=series.index)

    def _smooth_simple(series: pd.Series, length: int) -> pd.Series:
        # iSmooth(price,int length,...) simple variant used by "smoo"
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        s0 = np.empty_like(arr)
        s1 = np.empty_like(arr)
        s2 = np.empty_like(arr)
        s3 = np.empty_like(arr)
        s4 = np.empty_like(arr)
        s0[:] = np.nan
        s1[:] = np.nan
        s2[:] = np.nan
        s3[:] = np.nan
        s4[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                s4[i] = np.nan
                continue
            if i <= 2:
                s0[i] = xi
                s2[i] = xi
                s4[i] = xi
                s1[i] = 0.0
                s3[i] = 0.0
            else:
                beta = 0.45 * (length - 1.0) / (0.45 * (length - 1.0) + 2.0)
                alpha = beta
                s0[i] = xi + alpha * (s0[i - 1] - xi)
                s1[i] = (xi - s0[i]) * (1 - alpha) + alpha * s1[i - 1]
                s2[i] = s0[i] + s1[i]
                s3[i] = (s2[i] - s4[i - 1]) * (1 - alpha) ** 2 + (alpha**2) * s3[i - 1]
                s4[i] = s3[i] + s4[i - 1]
        return pd.Series(s4, index=series.index)

    def _linr(series: pd.Series, length: int) -> pd.Series:
        # 3*LWMA - 2*SMA approximation per MQL implementation
        L = _lwma(series, length)
        S = _sma(series, length)
        return 3.0 * L - 2.0 * S

    def _ilinr(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        sma = np.empty_like(arr)
        sma[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i + 1 >= length:
                window = arr[i - length + 1 : i + 1]
                idx = np.arange(length, dtype=float)
                sumx = idx.sum()
                sumxx = (idx**2).sum()
                sumy = np.nansum(window)
                sum1 = np.nansum(window * idx)
                slope = 0.0
                den = sumx * sumx - length * sumxx
                if den != 0:
                    slope = (sum1 * length - sumx * sumy) / den
                sma[i] = np.nanmean(window)
                out[i] = sma[i] + slope
            else:
                out[i] = xi
        return pd.Series(out, index=series.index)

    def _ie2(series: pd.Series, length: int) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            xi = arr[i]
            if not np.isfinite(xi):
                out[i] = np.nan
                continue
            if i + 1 >= length:
                window = arr[i - length + 1 : i + 1]
                idx = np.arange(length, dtype=float)
                sumx = idx.sum()
                sumxx = (idx**2).sum()
                sumxy = np.nansum(window * idx)
                sumy = np.nansum(window)
                den = sumx * sumx - length * sumxx
                tslope = (length * sumxy - sumx * sumy) / den if den != 0 else 0.0
                average = sumy / length
                out[i] = ((average + tslope) + (sumy + tslope * sumx) / length) / 2.0
            else:
                out[i] = xi
        return pd.Series(out, index=series.index)

    def _nlma(series: pd.Series, length: float) -> pd.Series:
        L = float(length)
        if L < 3 or n == 0:
            return series.astype(float)
        Cycle = 4.0
        Coeff = 3.0 * np.pi
        Phase = int(L - 1)
        total_len = int(L * 4 + Phase)
        # Precompute alphas/weights
        alphas = np.empty(total_len)
        for k in range(total_len):
            if k <= Phase - 1:
                t = 1.0 * k / max(Phase - 1, 1)
            else:
                t = 1.0 + (k - Phase + 1) * (2.0 * Cycle - 1.0) / (Cycle * L - 1.0)
            beta = np.cos(np.pi * t)
            g = 1.0 if t <= 0.5 else 1.0 / (Coeff * t + 1.0)
            alphas[k] = g * beta
        wsum = alphas.sum()
        arr = series.to_numpy(copy=True)
        out = np.empty_like(arr)
        out[:] = np.nan
        for i in range(n):
            if i == 0 or not np.isfinite(arr[i]):
                out[i] = arr[i] if np.isfinite(arr[i]) else np.nan
                continue
            acc = 0.0
            kk = 0
            j = i
            while kk < total_len and j >= 0:
                val = arr[j]
                if np.isfinite(val):
                    acc += alphas[kk] * val
                kk += 1
                j -= 1
            if wsum != 0:
                out[i] = acc / wsum
            else:
                out[i] = 0.0
        return pd.Series(out, index=series.index)

    ma_key = trend_method.lower()
    if ma_key == "sma":
        mma = _sma(x, int(trend_period))
    elif ma_key == "ema":
        mma = _ema(x, float(trend_period))
    elif ma_key == "dsema":
        mma = _dsema(x, float(trend_period))
    elif ma_key == "dema":
        mma = _dema(x, float(trend_period))
    elif ma_key == "tema":
        mma = _tema(x, float(trend_period))
    elif ma_key == "smma":
        mma = _smma(x, int(trend_period))
    elif ma_key == "lwma":
        mma = _lwma(x, int(trend_period))
    elif ma_key == "pwma":
        mma = _pwma(x, int(trend_period))
    elif ma_key == "vwma":
        mma = _vwma(x, v, int(trend_period))
    elif ma_key == "hull":
        mma = _hull(x, int(trend_period))
    elif ma_key == "tma":
        mma = _tma(x, int(trend_period))
    elif ma_key == "sine":
        mma = _sinewma(x, int(trend_period))
    elif ma_key == "mcg":
        mma = _mcg(x, float(trend_period))
    elif ma_key == "zlma":
        mma = _zlma(x, float(trend_period))
    elif ma_key == "lead":
        mma = _leader(x, float(trend_period))
    elif ma_key == "ssm":
        mma = _ssm(x, float(trend_period))
    elif ma_key == "smoo":
        mma = _smooth_simple(x, int(trend_period))
    elif ma_key == "linr":
        mma = _linr(x, int(trend_period))
    elif ma_key == "ilinr":
        mma = _ilinr(x, int(trend_period))
    elif ma_key == "ie2":
        mma = _ie2(x, int(trend_period))
    elif ma_key == "nlma":
        mma = _nlma(x, float(trend_period))
    else:
        raise ValueError(f"Unsupported trend_method '{trend_method}'")

    # SMMA of MMA with alpha = 2/(trend_period+1) (EMA-like on MMA)
    alpha = 2.0 / (float(trend_period) + 1.0) if trend_period > 0 else 1.0
    smma2 = mma.ewm(alpha=alpha, adjust=False).mean()

    impet_mma = mma.diff()
    impet_smma = smma2.diff()
    divma = (mma - smma2).abs() / float(point)
    averimpet = (impet_mma + impet_smma) / (2.0 * float(point))
    tdf_raw = divma * (averimpet**3)

    # Normalize by rolling max(abs) over trend_period*3
    norm_len = max(int(trend_period * 3), 1)
    absmax = tdf_raw.abs().rolling(norm_len, min_periods=1).max()
    normalized = pd.Series(
        np.where(absmax.to_numpy() > 0, (tdf_raw / absmax).to_numpy(), 0.0),
        index=tdf_raw.index,
    )

    # Final smoothing with variable-phase iSmooth (length, phase)
    def _iSmooth_variable(series: pd.Series, length: float, phase: float) -> pd.Series:
        if length <= 1:
            return series.astype(float)
        arr = series.to_numpy(copy=True)
        # wrk columns: [0..4] + [bsmax, bsmin, volty, vsum, avolty] = 10 total
        wrk = np.zeros((n, 10), dtype=float)
        out = np.empty(n, dtype=float)
        out[:] = np.nan
        for r in range(n):
            price_r = arr[r]
            if not np.isfinite(price_r):
                out[r] = np.nan
                if r > 0:
                    wrk[r] = wrk[r - 1]
                continue
            if r == 0:
                wrk[r, 0:7] = price_r
                wrk[r, 7:] = 0.0
                out[r] = price_r
                continue
            len1 = max(np.log(np.sqrt(0.5 * (length - 1))) / np.log(2.0) + 2.0, 0.0)
            pow1 = max(len1 - 2.0, 0.5)
            del1 = price_r - wrk[r - 1, 5]  # bsmax
            del2 = price_r - wrk[r - 1, 6]  # bsmin
            div = 1.0 / (10.0 + 10.0 * (min(max(length - 10.0, 0.0), 100.0)) / 100.0)
            forBar = min(r, 10)
            volty = max(abs(del1), abs(del2))
            wrk[r, 7] = wrk[r - 1, 7] + (volty - wrk[r - forBar, 7]) * div  # vsum
            av_prev = wrk[r - 1, 9]
            wrk[r, 9] = av_prev + (2.0 / (max(4.0 * length, 30.0) + 1.0)) * (
                wrk[r, 7] - av_prev
            )
            dVolty = 0.0
            if wrk[r, 9] > 0:
                dVolty = volty / wrk[r, 9]
            dVolty = min(max(dVolty, 1.0), len1 ** (1.0 / pow1))
            pow2 = dVolty**pow1
            len2 = np.sqrt(0.5 * (length - 1.0)) * len1
            Kv = (len2 / (len2 + 1.0)) ** np.sqrt(pow2)
            wrk[r, 5] = price_r if del1 > 0 else price_r - Kv * del1  # bsmax
            wrk[r, 6] = price_r if del2 < 0 else price_r - Kv * del2  # bsmin
            R = np.clip(phase, -100.0, 100.0) / 100.0 + 1.5
            beta = 0.45 * (length - 1.0) / (0.45 * (length - 1.0) + 2.0)
            alpha_v = beta**pow2
            wrk[r, 0] = price_r + alpha_v * (wrk[r - 1, 0] - price_r)
            wrk[r, 1] = (price_r - wrk[r, 0]) * (1.0 - beta) + beta * wrk[r - 1, 1]
            wrk[r, 2] = wrk[r, 0] + R * wrk[r, 1]
            wrk[r, 3] = (wrk[r, 2] - wrk[r - 1, 4]) * (1.0 - alpha_v) ** 2 + (
                alpha_v**2
            ) * wrk[r - 1, 3]
            wrk[r, 4] = wrk[r - 1, 4] + wrk[r, 3]
            out[r] = wrk[r, 4]
        return pd.Series(out, index=series.index)

    tdf_smoothed = _iSmooth_variable(
        normalized, float(smooth_length), float(smooth_phase)
    )

    # Trend states
    trend = np.zeros(n, dtype=float)
    tb = tdf_smoothed.to_numpy(copy=True)
    for i in range(n):
        prev = trend[i - 1] if i > 0 else 0.0
        val = tb[i]
        if not np.isfinite(val):
            trend[i] = prev
            continue
        if color_change_on_zero_cross:
            if val > 0:
                trend[i] = 1.0
            elif val < 0:
                trend[i] = -1.0
            else:
                trend[i] = prev
        else:
            if val > trigger_up:
                trend[i] = 1.0
            elif val < trigger_down:
                trend[i] = -1.0
            else:
                trend[i] = 0.0

    # Segmented up/down plot arrays (emulate PlotPoint backward)
    def _plot_segments(trend_arr: np.ndarray, src: np.ndarray):
        up_a = np.full(n, np.nan)
        up_b = np.full(n, np.nan)
        dn_a = np.full(n, np.nan)
        dn_b = np.full(n, np.nan)

        def plot_point(i, first, second, source):
            if i >= n - 2:
                return
            if np.isnan(first[i + 1]):
                if np.isnan(first[i + 2]):
                    first[i] = source[i]
                    first[i + 1] = source[i + 1]
                    second[i] = np.nan
                else:
                    second[i] = source[i]
                    second[i + 1] = source[i + 1]
                    first[i] = np.nan
            else:
                first[i] = source[i]
                second[i] = np.nan

        for i in range(n - 1, -1, -1):
            if not np.isfinite(src[i]):
                continue
            if trend_arr[i] == 1.0:
                plot_point(i, up_a, up_b, src)
            elif trend_arr[i] == -1.0:
                plot_point(i, dn_a, dn_b, src)
        return up_a, up_b, dn_a, dn_b

    up_a, up_b, dn_a, dn_b = _plot_segments(trend, tb)

    out = pd.DataFrame(
        {
            "TDF": tdf_smoothed,
            "TriggerUp": pd.Series(np.full(n, trigger_up), index=df.index),
            "TriggerDown": pd.Series(np.full(n, trigger_down), index=df.index),
            "UpA": pd.Series(up_a, index=df.index),
            "UpB": pd.Series(up_b, index=df.index),
            "DownA": pd.Series(dn_a, index=df.index),
            "DownB": pd.Series(dn_b, index=df.index),
            "Trend": pd.Series(trend, index=df.index),
        },
        index=df.index,
    )
    return out


# === END TrendDirectionForceIndexSmoothed4_calc.py ===


# === BEGIN TrendLordNrpIndicator_calc.py ===
def TrendLordNrpIndicator(
    df: pd.DataFrame,
    length: int = 12,
    mode: str = "smma",
    price: str = "close",
    show_high_low: bool = False,
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Computes the Trend Lord (non-repainting display logic) using two-stage MA:
    MA(length, mode) on price, then MA(sqrt(length), mode) on the first MA.
    Columns:
      - Buy: slot at Low/High (or MA) depending on up/down state
      - Sell: second-stage MA (trend line)
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=["Buy", "Sell"], dtype=float)

    n = max(int(length), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)

    # Applied price
    p = price.lower()
    if p == "close":
        src = df["Close"]
    elif p == "open":
        src = df["Open"]
    elif p == "high":
        src = df["High"]
    elif p == "low":
        src = df["Low"]
    elif p == "median":
        src = (df["High"] + df["Low"]) / 2.0
    elif p == "typical":
        src = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif p == "weighted":
        src = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        raise ValueError(
            "price must be one of: close, open, high, low, median, typical, weighted"
        )

    m = mode.lower()

    def _ma(series: pd.Series, period: int, mth: str) -> pd.Series:
        if period <= 1:
            return series.astype(float)
        if mth in ("sma",):
            return series.rolling(window=period, min_periods=period).mean()
        elif mth in ("ema",):
            return series.ewm(span=period, adjust=False, min_periods=period).mean()
        elif mth in ("smma", "rma", "wilder"):
            alpha = 1.0 / period
            return series.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        elif mth in ("lwma", "wma"):
            weights = np.arange(1, period + 1, dtype=float)
            wsum = weights.sum()
            return series.rolling(window=period, min_periods=period).apply(
                lambda x: np.dot(x, weights) / wsum, raw=True
            )
        else:
            raise ValueError("mode must be one of: sma, ema, smma (rma/wilder), lwma")

    ma1 = _ma(src.astype(float), n, m)
    ma2 = _ma(ma1, sqrt_n, m)  # Array1 in MQL

    # Slots for Buy line
    if show_high_low:
        slot_ll = df["Low"].to_numpy(dtype=float)
        slot_hh = df["High"].to_numpy(dtype=float)
    else:
        slot_ll = ma1.to_numpy(dtype=float)
        slot_hh = ma1.to_numpy(dtype=float)

    # Direction based on current vs previous (MQL compares i to i+1)
    up = ma2 > ma2.shift(1)
    down = ma2 < ma2.shift(1)

    buy = np.full(len(df), np.nan, dtype=float)
    up_idx = up.fillna(False).to_numpy()
    dn_idx = down.fillna(False).to_numpy()
    buy[up_idx] = slot_ll[up_idx]
    buy[dn_idx] = slot_hh[dn_idx]

    sell = ma2.to_numpy(dtype=float)

    out = pd.DataFrame({"Buy": buy, "Sell": sell}, index=df.index)
    return out


# === END TrendLordNrpIndicator_calc.py ===


# === BEGIN Trimagen_calc.py ===
def Trimagen(
    df: pd.DataFrame, period: int = 20, applied_price: str = "close"
) -> pd.Series:
    """Return the primary indicator line(s) aligned to df.index; vectorized, handle NaNs, stable defaults. For the other parameters, use what is specifically required for this specific indicator."""
    p = int(period) if period is not None else 20
    if p < 1:
        p = 1

    len1 = int(np.floor((p + 1.0) / 2.0))
    len2 = int(np.ceil((p + 1.0) / 2.0))

    ap = (applied_price or "close").lower()
    if ap == "close":
        price = df["Close"]
    elif ap == "open":
        price = df["Open"]
    elif ap == "high":
        price = df["High"]
    elif ap == "low":
        price = df["Low"]
    elif ap == "median":
        price = (df["High"] + df["Low"]) / 2.0
    elif ap == "typical":
        price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    elif ap == "weighted":
        price = (df["High"] + df["Low"] + 2.0 * df["Close"]) / 4.0
    else:
        price = df["Close"]

    sma1 = price.rolling(window=len1, min_periods=len1).mean()
    trimagen = sma1.rolling(window=len2, min_periods=len2).mean()
    trimagen.name = "TriMAgen"
    return trimagen


# === END Trimagen_calc.py ===


# === BEGIN TrueStrengthIndex_calc.py ===
def TrueStrengthIndex(
    df: pd.DataFrame, first_r: int = 5, second_s: int = 8
) -> pd.Series:
    """Return the True Strength Index (TSI) as a pandas Series aligned to df.index.
    Uses nested EMAs of momentum and its absolute value:
        TSI = 100 * EMA(EMA(mtm, first_r), second_s) / EMA(EMA(|mtm|, first_r), second_s)
    Defaults match the provided MQL4 script. Vectorized, NaNs preserved during warmup.
    """
    if first_r <= 0 or second_s <= 0:
        raise ValueError("first_r and second_s must be positive integers.")

    close = df["Close"].astype(float)

    mtm = close.diff()  # Close - Close.shift(1)
    abs_mtm = mtm.abs()

    ema_mtm = mtm.ewm(span=first_r, adjust=False, min_periods=first_r).mean()
    ema_abs_mtm = abs_mtm.ewm(span=first_r, adjust=False, min_periods=first_r).mean()

    ema2_mtm = ema_mtm.ewm(span=second_s, adjust=False, min_periods=second_s).mean()
    ema2_abs_mtm = ema_abs_mtm.ewm(
        span=second_s, adjust=False, min_periods=second_s
    ).mean()

    tsi_values = np.divide(
        100.0 * ema2_mtm,
        ema2_abs_mtm,
        out=np.full_like(ema2_mtm, np.nan),
        where=ema2_abs_mtm != 0,
    )

    tsi = pd.Series(tsi_values, index=df.index, name=f"TSI_{first_r}_{second_s}")
    return tsi


# === END TrueStrengthIndex_calc.py ===


# === BEGIN TTF_calc.py ===
def TTF(
    df: pd.DataFrame,
    ttf_bars: int = 8,
    top_line: float = 75.0,
    bottom_line: float = -75.0,
    t3_period: int = 3,
    b: float = 0.7,
) -> pd.DataFrame:
    """Trend Trigger Factor (TTF) with T3 smoothing.
    Returns a DataFrame with columns ['TTF','Signal'] aligned to df.index.
    Vectorized using pandas/numpy, NaNs for warmup.
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=["TTF", "Signal"], dtype=float)

    n = max(int(ttf_bars), 1)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Recent and older rolling extrema
    hh_recent = high.rolling(window=n, min_periods=n).max()
    ll_recent = low.rolling(window=n, min_periods=n).min()
    hh_older = hh_recent.shift(n)
    ll_older = ll_recent.shift(n)

    buy_power = hh_recent - ll_older
    sell_power = hh_older - ll_recent

    den = 0.5 * (buy_power + sell_power)
    den = den.replace(0.0, np.nan)
    ttf_raw = (buy_power - sell_power) / den * 100.0

    # T3 smoothing parameters
    r = float(max(t3_period, 1))
    r = 1.0 + 0.5 * (r - 1.0)  # as in the MQL4 code
    alpha = 2.0 / (r + 1.0)

    b2 = b * b
    b3 = b2 * b
    c1 = -b3
    c2 = 3.0 * (b2 + b3)
    c3 = -3.0 * (2.0 * b2 + b + b3)
    c4 = 1.0 + 3.0 * b + b3 + 3.0 * b2

    def ema_zero_seed(x: pd.Series, a: float) -> pd.Series:
        """EMA with initial previous value = 0 (zero-seeded), vectorized via ewm."""
        y = x.copy()
        fv = y.first_valid_index()
        if fv is not None:
            y.loc[fv] = y.loc[fv] * a  # ensures y_fv = a * x_fv (zero seed)
        return y.ewm(alpha=a, adjust=False, ignore_na=True).mean()

    e1 = ema_zero_seed(ttf_raw, alpha)
    e2 = ema_zero_seed(e1, alpha)
    e3 = ema_zero_seed(e2, alpha)
    e4 = ema_zero_seed(e3, alpha)
    e5 = ema_zero_seed(e4, alpha)
    e6 = ema_zero_seed(e5, alpha)

    ttf_main = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    signal = pd.Series(
        np.where(ttf_main >= 0.0, top_line, bottom_line), index=df.index, dtype=float
    )
    signal = signal.where(ttf_main.notna())

    return pd.DataFrame({"TTF": ttf_main, "Signal": signal}, index=df.index)


# === END TTF_calc.py ===


# === BEGIN TTMS_calc.py ===
def TTMS(
    df: pd.DataFrame,
    bb_length: int = 20,
    bb_deviation: float = 2.0,
    keltner_length: int = 20,
    keltner_smooth_length: int = 20,
    keltner_smooth_method: int = 0,  # 0-SMA, 1-EMA, 2-SMMA(Wilder), 3-LWMA
    keltner_deviation: float = 2.0,
) -> pd.DataFrame:
    """Return TTMS buffers as columns aligned to df.index."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=n).mean()

    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def _rma(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(alpha=1.0 / max(n, 1), adjust=False, min_periods=n).mean()

    def _lwma(s: pd.Series, n: int) -> pd.Series:
        if n <= 0:
            return pd.Series(np.nan, index=s.index)
        w = np.arange(1, n + 1, dtype=float)
        w_sum = w.sum()
        return s.rolling(n, min_periods=n).apply(
            lambda x: np.dot(x, w) / w_sum, raw=True
        )

    def _ma(s: pd.Series, n: int, method: int) -> pd.Series:
        if method == 0:
            return _sma(s, n)
        elif method == 1:
            return _ema(s, n)
        elif method == 2:
            return _rma(s, n)
        elif method == 3:
            return _lwma(s, n)
        else:
            return _sma(s, n)

    # Keltner center line MA
    ma_keltner = _ma(close, keltner_length, keltner_smooth_method)

    # ATR (Wilder)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = _rma(tr, keltner_smooth_length)

    # Keltner bands
    keltner_high = ma_keltner + atr * keltner_deviation
    keltner_low = ma_keltner - atr * keltner_deviation

    # Bollinger middle and std (population, MODE_SMA)
    bb_mid = _sma(close, bb_length)
    bb_std = close.rolling(bb_length, min_periods=bb_length).std(ddof=0)
    bb_top = bb_mid + bb_deviation * bb_std
    bb_bot = bb_mid - bb_deviation * bb_std

    # TTMS value
    denom = bb_top - bb_bot
    ttms_raw = (keltner_high - keltner_low) / denom - 1.0
    ttms = ttms_raw.where(denom != 0)

    # Split into Up/Dn histograms based on slope vs previous
    prev_ttms = ttms.shift(1)
    cond_up = ttms > prev_ttms
    ttms_up = ttms.where(cond_up, 0.0)
    ttms_dn = ttms.where(~cond_up, 0.0)
    valid_ttms = ttms.notna()
    ttms_up = ttms_up.where(valid_ttms)
    ttms_dn = ttms_dn.where(valid_ttms)

    # Alerts (squeeze on when BB is inside Keltner)
    valid_bands = (
        keltner_high.notna() & keltner_low.notna() & bb_top.notna() & bb_bot.notna()
    )
    squeeze_on = (bb_top < keltner_high) & (bb_bot > keltner_low) & valid_bands
    alert = pd.Series(np.where(squeeze_on, 1e-5, np.nan), index=df.index)
    noalert = pd.Series(
        np.where((~squeeze_on) & valid_bands, 1e-5, np.nan), index=df.index
    )

    out = pd.DataFrame(
        {
            "TTMS_Up": ttms_up,
            "TTMS_Dn": ttms_dn,
            "Alert": alert,
            "NoAlert": noalert,
        },
        index=df.index,
    )
    return out


# === END TTMS_calc.py ===


# === BEGIN VIDYA_calc.py ===
def VIDYA(df: pd.DataFrame, period: int = 9, histper: int = 30) -> pd.Series:
    """Return the VIDYA line aligned to df.index; vectorized where possible, stable defaults.
    Parameters:
    - period: int = 9
    - histper: int = 30
    Uses Close price only (as per source)."""
    close = pd.to_numeric(df["Close"], errors="coerce")
    n = len(close)
    if n == 0:
        return pd.Series([], index=df.index, dtype=float, name="VIDYA")

    period = max(1, int(period))
    histper = max(1, int(histper))

    std_short = close.rolling(window=period, min_periods=period).std(ddof=0)
    std_long = close.rolling(window=histper, min_periods=histper).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = std_short / std_long

    sc = 2.0 / (period + 1.0)
    alpha = k * sc  # may exceed 1; matching source behavior

    c = close.to_numpy(dtype=float, copy=False)
    a = alpha.to_numpy(dtype=float, copy=False)
    out = np.full(n, np.nan, dtype=float)

    if histper <= n:
        out[:histper] = c[:histper]
        start = histper
    else:
        out[:] = c
        return pd.Series(out, index=df.index, name="VIDYA")

    for i in range(start, n):
        prev = out[i - 1]
        x = c[i]
        ai = a[i]
        if not np.isfinite(ai):
            ai = 0.0
        if np.isnan(prev):
            out[i] = x
        else:
            out[i] = ai * x + (1.0 - ai) * prev

    return pd.Series(out, index=df.index, name="VIDYA")


# === END VIDYA_calc.py ===


# === BEGIN VolatilityRatio_calc.py ===
def VolatilityRatio(
    df: pd.DataFrame, period: int = 25, price: str = "Close"
) -> pd.DataFrame:
    """Return ALL lines as columns aligned to df.index:
    - VR: volatility ratio (std / SMA(std))
    - VR_below1_a: alternating segments of VR where VR < 1 (set A)
    - VR_below1_b: alternating segments of VR where VR < 1 (set B)
    Vectorized, handles NaNs, and preserves length with NaNs for warmup."""
    if period < 1:
        period = 1

    h, l, c = df["High"], df["Low"], df["Close"]
    p_lower = str(price).lower()
    if p_lower == "close":
        p = c
    elif p_lower == "open":
        p = df["Open"]
    elif p_lower == "high":
        p = h
    elif p_lower == "low":
        p = l
    elif p_lower in ("hl2", "median"):
        p = (h + l) / 2.0
    elif p_lower in ("hlc3", "typical"):
        p = (h + l + c) / 3.0
    elif p_lower in ("ohlc4",):
        p = (df["Open"] + h + l + c) / 4.0
    elif p_lower in ("weighted", "wclose", "wclose4"):
        p = (h + l + 2.0 * c) / 4.0
    else:
        # Fallback to close if unknown
        p = c

    # Population std over 'period' and its SMA over 'period'
    std = p.rolling(window=period, min_periods=period).std(ddof=0)
    ma_std = std.rolling(window=period, min_periods=period).mean()

    vr = std / ma_std
    vr = vr.where(ma_std.ne(0.0), 1.0)

    # Build alternating below-1 segments into two buffers
    mask = vr < 1.0
    run_start = mask & ~mask.shift(fill_value=False)
    run_id = run_start.cumsum()  # increases only at starts of True runs
    # For non-True positions, set run_id to 0 to simplify parity checks
    run_id_arr = np.where(mask.to_numpy(), run_id.to_numpy(), 0)

    is_a = mask & (run_id_arr % 2 == 1)
    is_b = mask & (run_id_arr % 2 == 0) & (run_id_arr != 0)

    vr_below_a = vr.where(is_a)
    vr_below_b = vr.where(is_b)

    out = pd.DataFrame(
        {
            "VR": vr.astype(float),
            "VR_below1_a": vr_below_a.astype(float),
            "VR_below1_b": vr_below_b.astype(float),
        },
        index=df.index,
    )
    return out


# === END VolatilityRatio_calc.py ===


# === BEGIN VortexIndicator_calc.py ===
def VortexIndicator(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Compute the Vortex Indicator (VI+ and VI-) over 'length' periods.
    Returns a DataFrame with columns ['VI_plus_{length}', 'VI_minus_{length}'] aligned to df.index.
    Vectorized using pandas; preserves length with NaNs for warmup periods.
    """
    high = df["High"].astype("float64")
    low = df["Low"].astype("float64")
    close = df["Close"].astype("float64")

    prev_low = low.shift(1)
    prev_high = high.shift(1)
    prev_close = close.shift(1)

    plus_vm = (high - prev_low).abs()
    minus_vm = (low - prev_high).abs()

    tr_hl = (high - low).abs()
    tr_hc = (high - prev_close).abs()
    tr_lc = (low - prev_close).abs()
    tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)

    sum_plus_vm = plus_vm.rolling(length, min_periods=length).sum()
    sum_minus_vm = minus_vm.rolling(length, min_periods=length).sum()
    sum_tr = tr.rolling(length, min_periods=length).sum().replace(0, np.nan)

    vi_plus = sum_plus_vm / sum_tr
    vi_minus = sum_minus_vm / sum_tr

    out = pd.DataFrame(
        {f"VI_plus_{length}": vi_plus, f"VI_minus_{length}": vi_minus}, index=df.index
    )

    return out


# === END VortexIndicator_calc.py ===


# === BEGIN WilliamVixFix_calc.py ===
def WilliamVixFix(df: pd.DataFrame, period: int = 22) -> pd.Series:
    """Return the William VIX-FIX primary line aligned to df.index; vectorized, handles NaNs, stable defaults."""
    close = df["Close"]
    low = df["Low"]
    max_close = close.rolling(window=period, min_periods=period).max()
    wvf = 100.0 * (max_close - low) / max_close
    wvf = wvf.where(~(max_close <= 0), 0.0)
    wvf.name = f"WilliamVixFix_{period}"
    return wvf


# === END WilliamVixFix_calc.py ===


# === BEGIN WprMaAlerts_calc.py ===
def WprMaAlerts(
    df: pd.DataFrame,
    wpr_period: int = 35,
    signal_period: int = 21,
    ma_method: str = "smma",
) -> pd.DataFrame:
    """Williams %R with signal MA and cross state.

    Parameters:
    - df: DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume']
    - wpr_period: int, lookback for Williams %R (default 35)
    - signal_period: int, lookback for signal MA (default 21)
    - ma_method: str, one of {'sma','ema','smma','lwma'} (default 'smma')

    Returns:
    - DataFrame with columns ['WPR','Signal','Cross'], aligned to df.index
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # Williams %R
    hh = high.rolling(window=int(max(1, wpr_period)), min_periods=1).max()
    ll = low.rolling(window=int(max(1, wpr_period)), min_periods=1).min()
    denom = (hh - ll).to_numpy()
    wpr_vals = np.where(
        denom != 0, -100.0 * (hh.to_numpy() - close.to_numpy()) / denom, 0.0
    )
    wpr = pd.Series(wpr_vals, index=df.index)

    # Signal MA helpers
    def _lwma(series: pd.Series, period: int) -> pd.Series:
        period = max(1, int(period))
        s = series.to_numpy(dtype=float)
        n = s.shape[0]
        out = np.full(n, np.nan, dtype=float)

        # Early ramp: variable window sizes 1..min(period-1, n)
        early = min(period - 1, n)
        for t in range(early):
            w = t + 1
            window = s[:w]
            mask = ~np.isnan(window)
            if mask.any():
                wts = np.arange(1, w + 1, dtype=float)[mask]
                out[t] = np.dot(window[mask], wts) / wts.sum()

        # Steady state: full window = period
        if n >= period:
            sw = sliding_window_view(
                s, window_shape=period
            )  # shape (n-period+1, period)
            mask = ~np.isnan(sw)
            wts = np.arange(1, period + 1, dtype=float)
            denom = (wts * mask).sum(axis=1)
            num = np.nansum(sw * wts, axis=1)
            vals = num / denom
            out[period - 1 :] = vals

        return pd.Series(out, index=series.index)

    mm = str(ma_method).strip().lower()
    p = max(1, int(signal_period))
    if p <= 1:
        sig = wpr.copy()
    elif mm == "sma":
        sig = wpr.rolling(window=p, min_periods=1).mean()
    elif mm == "ema":
        sig = wpr.ewm(alpha=2.0 / (p + 1.0), adjust=False, min_periods=1).mean()
    elif mm == "smma":
        sig = wpr.ewm(alpha=1.0 / p, adjust=False, min_periods=1).mean()
    elif mm == "lwma":
        sig = _lwma(wpr, p)
    else:
        # Fallback: no smoothing
        sig = wpr.copy()

    # Cross state: 1 if WPR > Signal, -1 if WPR < Signal, else carry previous; NaN where diff is NaN
    diff = wpr - sig
    raw = np.sign(diff.to_numpy())
    raw[np.isnan(diff.to_numpy())] = np.nan
    cross = pd.Series(raw, index=df.index)
    cross_filled = cross.replace(0.0, np.nan).ffill()
    cross_final = cross_filled.where(~diff.isna(), np.nan).fillna(0.0).astype(float)
    # Cast to int where not NaN; keep NaN as NaN
    cross_int = pd.Series(
        np.where(cross_final.notna(), cross_final.astype(int), np.nan), index=df.index
    )

    out = pd.DataFrame(
        {
            "WPR": wpr,
            "Signal": sig,
            "Cross": cross_int,
        },
        index=df.index,
    )
    return out


# === END WprMaAlerts_calc.py ===


# === BEGIN XmaColouredUpdatedForNnfx_calc.py ===
def XmaColouredUpdatedForNnfx(
    df: pd.DataFrame,
    period: int = 12,
    porog: int = 3,
    metod: str | int = "ema",
    metod2: str | int = "ema",
    price: str = "close",
    tick_size: float = 0.0001,
) -> pd.DataFrame:
    """Return ALL lines as columns with clear denoted names; aligned to df.index.
    Parameters:
      - period: MA length (default 12)
      - porog: threshold in 'points' (integer), multiplied by tick_size (default 3)
      - metod/metod2: MA methods ('sma','ema','smma','lwma' or 0/1/2/3 respectively; default 'ema')
      - price: applied price ('close','open','high','low','median','typical','weighted'; default 'close')
      - tick_size: price per point (default 0.0001)
    """

    def _applied_price(data: pd.DataFrame, p: str) -> pd.Series:
        p = str(p).lower()
        if p == "close":
            return data["Close"].astype(float)
        if p == "open":
            return data["Open"].astype(float)
        if p == "high":
            return data["High"].astype(float)
        if p == "low":
            return data["Low"].astype(float)
        if p == "median":
            return (data["High"] + data["Low"]) / 2.0
        if p == "typical":
            return (data["High"] + data["Low"] + data["Close"]) / 3.0
        if p == "weighted":
            return (data["High"] + data["Low"] + 2.0 * data["Close"]) / 4.0
        # default to close
        return data["Close"].astype(float)

    def _normalize_method(m):
        m_map = {
            0: "sma",
            1: "ema",
            2: "smma",
            3: "lwma",
            "sma": "sma",
            "ema": "ema",
            "smma": "smma",
            "rma": "smma",
            "lwma": "lwma",
            "wma": "lwma",
        }
        key = m if isinstance(m, (int, np.integer)) else str(m).lower()
        return m_map.get(key, "ema")

    def _ma(x: pd.Series, length: int, method: str) -> pd.Series:
        method = _normalize_method(method)
        if method == "sma":
            return x.rolling(length, min_periods=length).mean()
        if method == "ema":
            return x.ewm(span=length, adjust=False, min_periods=length).mean()
        if method == "smma":  # Wilder's / RMA
            alpha = 1.0 / float(length)
            return x.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
        if method == "lwma":
            w = np.arange(1, length + 1, dtype=float)
            w_sum = w.sum()
            return x.rolling(length, min_periods=length).apply(
                lambda a: np.dot(a, w) / w_sum, raw=True
            )
        # fallback
        return x.ewm(span=length, adjust=False, min_periods=length).mean()

    p = _applied_price(df, price)
    close = df["Close"].astype(float)

    ma1 = _ma(p, period, metod)
    ma2 = _ma(p, period, metod2).shift(1)

    threshold = float(porog) * float(tick_size)
    cond = ma1.subtract(ma2).abs() >= threshold

    # Build Signal: when cond True use ma2, else carry forward previous Signal
    signal_candidates = ma2.where(cond)
    signal = signal_candidates.ffill()

    # Initialize output columns
    up = pd.Series(np.nan, index=df.index, dtype=float)
    dn = pd.Series(np.nan, index=df.index, dtype=float)
    fl = pd.Series(np.nan, index=df.index, dtype=float)

    # Masks
    event = cond & signal.notna()
    non_event = (~cond) & signal.notna()

    up_mask = event & (close >= signal)
    dn_mask = event & (close <= signal)
    fl_mask_event = event & (close != signal)

    up.loc[up_mask] = signal.loc[up_mask]
    dn.loc[dn_mask] = signal.loc[dn_mask]
    fl.loc[fl_mask_event] = signal.loc[fl_mask_event]
    fl.loc[non_event] = signal.loc[non_event]

    out = pd.DataFrame(
        {
            "Signal": signal,
            "Fl": fl,
            "Up": up,
            "Dn": dn,
        },
        index=df.index,
    )
    return out


# === END XmaColouredUpdatedForNnfx_calc.py ===


# === BEGIN ZerolagMACDMq4_calc.py ===
def ZerolagMACDMq4(
    df: pd.DataFrame, fast: int = 12, slow: int = 24, signal: int = 9
) -> pd.DataFrame:
    """Return ZeroLag MACD and Signal lines as columns aligned to df.index.
    Uses MT4-style EMA seeding (first value is SMA over the period); NaNs preserved during warmup.
    """
    close = pd.Series(
        df["Close"].astype(float).to_numpy(), index=df.index, dtype="float64"
    )

    def _ema_mt4(s: pd.Series, period: int) -> pd.Series:
        if period is None or period <= 1:
            # EMA(1) equals the series itself
            out = s.astype("float64").copy()
            return out

        a = 2.0 / (period + 1.0)
        # Raw EMA (adjust=False => recursive form)
        ema_raw = s.ewm(alpha=a, adjust=False).mean()

        # Align to MT4 seeding: first EMA value equals SMA over the first 'period' valid observations
        roll = s.rolling(window=period, min_periods=period).mean()
        mask_valid = roll.notna().to_numpy()
        if not mask_valid.any():
            # Not enough data for initial SMA
            return pd.Series(np.nan, index=s.index, dtype="float64")

        pos0 = int(np.argmax(mask_valid))  # first index where SMA is available
        sma0 = float(roll.iloc[pos0])
        ema0 = float(ema_raw.iloc[pos0])
        delta = sma0 - ema0

        n = len(s)
        ema_adj = ema_raw.to_numpy(dtype="float64").copy()
        # NaN out everything before first valid EMA
        if pos0 > 0:
            ema_adj[:pos0] = np.nan
        # Exponential correction to enforce EMA[pos0] == SMA[pos0]
        decay = 1.0 - a
        powers = np.power(decay, np.arange(n - pos0, dtype="float64"))
        ema_adj[pos0:] = ema_adj[pos0:] + delta * powers

        return pd.Series(ema_adj, index=s.index, dtype="float64")

    # First-level EMAs
    ema_fast = _ema_mt4(close, fast)
    ema_slow = _ema_mt4(close, slow)

    # Zero-lag EMAs for fast and slow: 2*EMA - EMA(EMA)
    ema_fast2 = _ema_mt4(ema_fast, fast)
    ema_slow2 = _ema_mt4(ema_slow, slow)
    zl_fast = 2.0 * ema_fast - ema_fast2
    zl_slow = 2.0 * ema_slow - ema_slow2

    zl_macd = zl_fast - zl_slow

    # Signal line: zero-lag EMA of MACD
    sig_ema = _ema_mt4(zl_macd, signal)
    sig_ema2 = _ema_mt4(sig_ema, signal)
    zl_signal = 2.0 * sig_ema - sig_ema2

    out = pd.DataFrame(
        {
            "ZL_MACD": zl_macd.to_numpy(dtype="float64"),
            "ZL_Signal": zl_signal.to_numpy(dtype="float64"),
        },
        index=df.index,
    )
    return out


# === END ZerolagMACDMq4_calc.py ===
