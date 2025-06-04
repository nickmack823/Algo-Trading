import logging
from typing import Callable, TypedDict

import pandas as pd
import talib as ta

from scripts.indicator_functions import (
    ALMA,
    EHLERS,
    HMA,
    J_TPO,
    KASE,
    KVO,
    LSMA,
    LWPI,
    SSL,
    TCF,
    TDFI,
    TTF,
    UF2018,
    VIDYA,
    WAE,
    AcceleratorLSMA,
    BraidFilter,
    BraidFilterHist,
    BullsBearsImpulse,
    CenterOfGravity,
    Coral,
    FantailVMA,
    FilteredATR,
    Fisher,
    Gen3MA,
    GruchaIndex,
    HalfTrend,
    KalmanFilter,
    KijunSen,
    Laguerre,
    MACDZeroLag,
    McGinleyDI,
    NormalizedVolume,
    RecursiveMA,
    SchaffTrendCycle,
    SmoothStep,
    SuperTrend,
    TopTrend,
    TrendLord,
    TwiggsMF,
    VolatilityRatio,
    Vortex,
)
from scripts.signal_functions import *

logging.basicConfig(
    level=20, datefmt="%m/%d/%Y %H:%M:%S", format="[%(asctime)s] %(message)s"
)


# Indicator Config for indicator parameters in strategies
class IndicatorConfig(TypedDict):
    name: str
    function: Callable[[pd.DataFrame], pd.Series]
    description: str
    signal_function: Callable
    raw_function: Callable
    parameters: dict
    parameter_space = dict


# Utility to find config by name in a pool
def find_indicator_config(name, params) -> IndicatorConfig:
    for config in all_indicators:
        if config["name"] == name:
            return {
                "name": name,
                "function": config["function"],
                "signal_function": config["signal_function"],
                "raw_function": config.get("raw_function", config["function"]),
                "description": config.get("description", ""),
                "parameters": params,
            }
    raise ValueError(f"Indicator {name} not found in provided pool.")


def candle_2crows_func(df):
    return ta.CDL2CROWS(df["Open"], df["High"], df["Low"], df["Close"])


def candle_3blackcrows_func(df):
    return ta.CDL3BLACKCROWS(df["Open"], df["High"], df["Low"], df["Close"])


def candle_3inside_func(df):
    return ta.CDL3INSIDE(df["Open"], df["High"], df["Low"], df["Close"])


def candle_3linestrike_func(df):
    return ta.CDL3LINESTRIKE(df["Open"], df["High"], df["Low"], df["Close"])


def candle_3starsinsouth_func(df):
    return ta.CDL3STARSINSOUTH(df["Open"], df["High"], df["Low"], df["Close"])


def candle_3whitesoldiers_func(df):
    return ta.CDL3WHITESOLDIERS(df["Open"], df["High"], df["Low"], df["Close"])


def candle_abandonedbaby_func(df):
    return ta.CDLABANDONEDBABY(df["Open"], df["High"], df["Low"], df["Close"])


def candle_advanceblock_func(df):
    return ta.CDLADVANCEBLOCK(df["Open"], df["High"], df["Low"], df["Close"])


def candle_belthold_func(df):
    return ta.CDLBELTHOLD(df["Open"], df["High"], df["Low"], df["Close"])


def candle_breakaway_func(df):
    return ta.CDLBREAKAWAY(df["Open"], df["High"], df["Low"], df["Close"])


def candle_closingmarubozu_func(df):
    return ta.CDLCLOSINGMARUBOZU(df["Open"], df["High"], df["Low"], df["Close"])


def candle_concealbabyswall_func(df):
    return ta.CDLCONCEALBABYSWALL(df["Open"], df["High"], df["Low"], df["Close"])


def candle_counterattack_func(df):
    return ta.CDLCOUNTERATTACK(df["Open"], df["High"], df["Low"], df["Close"])


def candle_darkcloudcover_func(df):
    return ta.CDLDARKCLOUDCOVER(df["Open"], df["High"], df["Low"], df["Close"])


def candle_doji_func(df):
    return ta.CDLDOJI(df["Open"], df["High"], df["Low"], df["Close"])


def candle_dojistar_func(df):
    return ta.CDLDOJISTAR(df["Open"], df["High"], df["Low"], df["Close"])


def candle_dragonflydoji_func(df):
    return ta.CDLDRAGONFLYDOJI(df["Open"], df["High"], df["Low"], df["Close"])


def candle_engulfing_func(df):
    return ta.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])


def candle_eveningdojistar_func(df):
    return ta.CDLEVENINGDOJISTAR(df["Open"], df["High"], df["Low"], df["Close"])


def candle_eveningstar_func(df):
    return ta.CDLEVENINGSTAR(df["Open"], df["High"], df["Low"], df["Close"])


def candle_gapsidesidewhite_func(df):
    return ta.CDLGAPSIDESIDEWHITE(df["Open"], df["High"], df["Low"], df["Close"])


def candle_gravestonedoji_func(df):
    return ta.CDLGRAVESTONEDOJI(df["Open"], df["High"], df["Low"], df["Close"])


def candle_hammer_func(df):
    return ta.CDLHAMMER(df["Open"], df["High"], df["Low"], df["Close"])


def candle_hangingman_func(df):
    return ta.CDLHANGINGMAN(df["Open"], df["High"], df["Low"], df["Close"])


def candle_harami_func(df):
    return ta.CDLHARAMI(df["Open"], df["High"], df["Low"], df["Close"])


def candle_haramicross_func(df):
    return ta.CDLHARAMICROSS(df["Open"], df["High"], df["Low"], df["Close"])


def candle_highwave_func(df):
    return ta.CDLHIGHWAVE(df["Open"], df["High"], df["Low"], df["Close"])


def candle_hikkake_func(df):
    return ta.CDLHIKKAKE(df["Open"], df["High"], df["Low"], df["Close"])


def candle_hikkakemod_func(df):
    return ta.CDLHIKKAKEMOD(df["Open"], df["High"], df["Low"], df["Close"])


def candle_homingpigeon_func(df):
    return ta.CDLHOMINGPIGEON(df["Open"], df["High"], df["Low"], df["Close"])


def candle_identical3crows_func(df):
    return ta.CDLIDENTICAL3CROWS(df["Open"], df["High"], df["Low"], df["Close"])


def candle_inneck_func(df):
    return ta.CDLINNECK(df["Open"], df["High"], df["Low"], df["Close"])


def candle_invertedhammer_func(df):
    return ta.CDLINVERTEDHAMMER(df["Open"], df["High"], df["Low"], df["Close"])


def candle_kicking_func(df):
    return ta.CDLKICKING(df["Open"], df["High"], df["Low"], df["Close"])


def candle_kickingbylength_func(df):
    return ta.CDLKICKINGBYLENGTH(df["Open"], df["High"], df["Low"], df["Close"])


def candle_ladderbottom_func(df):
    return ta.CDLLADDERBOTTOM(df["Open"], df["High"], df["Low"], df["Close"])


def candle_longleggeddoji_func(df):
    return ta.CDLLONGLEGGEDDOJI(df["Open"], df["High"], df["Low"], df["Close"])


def candle_longline_func(df):
    return ta.CDLLONGLINE(df["Open"], df["High"], df["Low"], df["Close"])


def candle_marubozu_func(df):
    return ta.CDLMARUBOZU(df["Open"], df["High"], df["Low"], df["Close"])


def candle_matchinglow_func(df):
    return ta.CDLMATCHINGLOW(df["Open"], df["High"], df["Low"], df["Close"])


def candle_mathold_func(df):
    return ta.CDLMATHOLD(df["Open"], df["High"], df["Low"], df["Close"])


def candle_morningdojistar_func(df):
    return ta.CDLMORNINGDOJISTAR(df["Open"], df["High"], df["Low"], df["Close"])


def candle_morningstar_func(df):
    return ta.CDLMORNINGSTAR(df["Open"], df["High"], df["Low"], df["Close"])


def candle_onneck_func(df):
    return ta.CDLONNECK(df["Open"], df["High"], df["Low"], df["Close"])


def candle_piercing_func(df):
    return ta.CDLPIERCING(df["Open"], df["High"], df["Low"], df["Close"])


def candle_rickshawman_func(df):
    return ta.CDLRICKSHAWMAN(df["Open"], df["High"], df["Low"], df["Close"])


def candle_risefall3methods_func(df):
    return ta.CDLRISEFALL3METHODS(df["Open"], df["High"], df["Low"], df["Close"])


def candle_separatinglines_func(df):
    return ta.CDLSEPARATINGLINES(df["Open"], df["High"], df["Low"], df["Close"])


def candle_shootingstar_func(df):
    return ta.CDLSHOOTINGSTAR(df["Open"], df["High"], df["Low"], df["Close"])


def candle_shortline_func(df):
    return ta.CDLSHORTLINE(df["Open"], df["High"], df["Low"], df["Close"])


def candle_spinningtop_func(df):
    return ta.CDLSPINNINGTOP(df["Open"], df["High"], df["Low"], df["Close"])


def candle_stalledpattern_func(df):
    return ta.CDLSTALLEDPATTERN(df["Open"], df["High"], df["Low"], df["Close"])


def candle_sticksandwich_func(df):
    return ta.CDLSTICKSANDWICH(df["Open"], df["High"], df["Low"], df["Close"])


def candle_takuri_func(df):
    return ta.CDLTAKURI(df["Open"], df["High"], df["Low"], df["Close"])


def candle_tasukigap_func(df):
    return ta.CDLTASUKIGAP(df["Open"], df["High"], df["Low"], df["Close"])


def candle_thrusting_func(df):
    return ta.CDLTHRUSTING(df["Open"], df["High"], df["Low"], df["Close"])


def candle_tristar_func(df):
    return ta.CDLTRISTAR(df["Open"], df["High"], df["Low"], df["Close"])


def candle_unique3river_func(df):
    return ta.CDLUNIQUE3RIVER(df["Open"], df["High"], df["Low"], df["Close"])


def candle_upsidegap2crows_func(df):
    return ta.CDLUPSIDEGAP2CROWS(df["Open"], df["High"], df["Low"], df["Close"])


def candle_xsidegap3methods_func(df):
    return ta.CDLXSIDEGAP3METHODS(df["Open"], df["High"], df["Low"], df["Close"])


def ad_func(df, **kwargs):
    return ta.AD(df["High"], df["Low"], df["Close"], df["Volume"], **kwargs)


def adx_func(df, **kwargs):
    return ta.ADX(df["High"], df["Low"], df["Close"], **kwargs)


def adxr_func(df, **kwargs):
    return ta.ADXR(df["High"], df["Low"], df["Close"], **kwargs)


def adosc_func(df, **kwargs):
    return ta.ADOSC(df["High"], df["Low"], df["Close"], df["Volume"], **kwargs)


def obv_func(df, **kwargs):
    return ta.OBV(df["Close"], df["Volume"], **kwargs)


def aroon_func(df, **kwargs):
    return ta.AROON(df["High"], df["Low"], **kwargs)


def aroonosc_func(df, **kwargs):
    return ta.AROONOSC(df["High"], df["Low"], **kwargs)


def apo_func(df, fastperiod=12, slowperiod=26, matype=0):
    return ta.APO(
        df["Close"], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype
    )


def cci_func(df, **kwargs):
    return ta.CCI(df["High"], df["Low"], df["Close"], **kwargs)


def macd_func(df, **kwargs):
    return ta.MACD(df["Close"], **kwargs)


def macdext_func(df, **kwargs):
    return ta.MACDEXT(df["Close"], **kwargs)


def macdfix_func(df, **kwargs):
    return ta.MACDFIX(df["Close"], **kwargs)


def mama_func(df, **kwargs):
    return ta.MAMA(df["Close"], **kwargs)


def ppo_func(df, **kwargs):
    return ta.PPO(df["Close"], **kwargs)


def roc_func(df, **kwargs):
    return ta.ROC(df["Close"], **kwargs)


def rocp_func(df, **kwargs):
    return ta.ROCP(df["Close"], **kwargs)


def rocr_func(df, **kwargs):
    return ta.ROCR(df["Close"], **kwargs)


def rocr100_func(df, **kwargs):
    return ta.ROCR100(df["Close"], **kwargs)


def rsi_func(df, **kwargs):
    return ta.RSI(df["Close"], **kwargs)


def stoch_slow_func(df, **kwargs):
    return ta.STOCH(df["High"], df["Low"], df["Close"], **kwargs)


def stochf_fast_func(df, **kwargs):
    return ta.STOCHF(df["High"], df["Low"], df["Close"], **kwargs)


def stochrsi_func(df, **kwargs):
    return ta.STOCHRSI(df["Close"], **kwargs)


def tsf_func(df, **kwargs):
    return ta.TSF(df["Close"], **kwargs)


def ultosc_func(df, **kwargs):
    return ta.ULTOSC(df["High"], df["Low"], df["Close"], **kwargs)


def atr_func(df, **kwargs):
    return ta.ATR(df["High"], df["Low"], df["Close"], **kwargs)


def natr_func(df, **kwargs):
    return ta.NATR(df["High"], df["Low"], df["Close"], **kwargs)


def trange_func(df, **kwargs):
    return ta.TRANGE(df["High"], df["Low"], df["Close"], **kwargs)


def bollingerbands_func(df, **kwargs):
    return ta.BBANDS(df["Close"], **kwargs)


def stddev_func(df, **kwargs):
    return ta.STDDEV(df["Close"], **kwargs)


def var_func(df, **kwargs):
    return ta.VAR(df["Close"], **kwargs)


def avgprice_func(df):
    return ta.AVGPRICE(df["Open"], df["High"], df["Low"], df["Close"])


def medprice_func(df):
    return ta.MEDPRICE(df["High"], df["Low"])


def max_func(df, timeperiod=14):
    return ta.MAX(df["Close"], timeperiod=timeperiod)


def maxindex_func(df, timeperiod=14):
    return ta.MAXINDEX(df["Close"], timeperiod=timeperiod)


def midpoint_func(df, timeperiod=14):
    return ta.MIDPOINT(df["Close"], timeperiod=timeperiod)


def midprice_func(df, timeperiod=14):
    return ta.MIDPRICE(df["High"], df["Low"], timeperiod=timeperiod)


def min_func(df, timeperiod=14):
    return ta.MIN(df["Close"], timeperiod=timeperiod)


def minindex_func(df, timeperiod=14):
    return ta.MININDEX(df["Close"], timeperiod=timeperiod)


def minmax_func(df, timeperiod=14):
    return ta.MINMAX(df["Close"], timeperiod=timeperiod)


def sum_func(df, timeperiod=14):
    return ta.SUM(df["Close"], timeperiod=timeperiod)


def typprice_func(df):
    return ta.TYPPRICE(df["High"], df["Low"], df["Close"])


def wclprice_func(df):
    return ta.WCLPRICE(df["High"], df["Low"], df["Close"])


ta_lib_candlestick = [
    {
        "name": "CANDLE_2CROWS",
        "function": candle_2crows_func,
        "description": "Two Crows: A bearish reversal pattern where two consecutive bearish candles signal a potential trend change from bullish to bearish.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3BLACKCROWS",
        "function": candle_3blackcrows_func,
        "description": "Three Black Crows: A strong bearish reversal pattern characterized by three long bearish candles with short shadows, suggesting heavy selling.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3INSIDE",
        "function": candle_3inside_func,
        "description": "Three Inside: A pattern where a small candle is engulfed by its neighbors, hinting at potential reversal or pause in the current trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3LINESTRIKE",
        "function": candle_3linestrike_func,
        "description": "Three Line Strike: A reversal pattern where three candles are followed by a candle that ‘strikes’ through them, indicating a possible trend reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3STARSINSOUTH",
        "function": candle_3starsinsouth_func,
        "description": "Three Stars in the South: A bullish reversal pattern with three small candles after a downtrend, suggesting emerging buying pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_3WHITESOLDIERS",
        "function": candle_3whitesoldiers_func,
        "description": "Three White Soldiers: A bullish reversal pattern marked by three consecutive strong white candles, indicating sustained upward momentum.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ABANDONEDBABY",
        "function": candle_abandonedbaby_func,
        "description": "Abandoned Baby: A rare and powerful reversal pattern featuring an isolated doji with gaps on both sides, signaling a sharp change in trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ADVANCEBLOCK",
        "function": candle_advanceblock_func,
        "description": "Advance Block: A bearish reversal pattern that appears after an uptrend when bullish candles become constrained, hinting at exhaustion.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_BELTHOLD",
        "function": candle_belthold_func,
        "description": "Belt Hold: A reversal signal where a long candle opens at an extreme (low for bullish, high for bearish) and moves strongly, indicating momentum.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_BREAKAWAY",
        "function": candle_breakaway_func,
        "description": "Breakaway: A pattern that forms after a gap and a series of candles, signaling the end of a trend and the potential start of a reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_CLOSINGMARUBOZU",
        "function": candle_closingmarubozu_func,
        "description": "Closing Marubozu: A candlestick with little or no shadows that shows strong momentum in the direction of the close.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_CONCEALBABYSWALL",
        "function": candle_concealbabyswall_func,
        "description": "Concealing Baby Swallow: A bearish reversal pattern where a small candle is engulfed, suggesting a shift in market sentiment.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_COUNTERATTACK",
        "function": candle_counterattack_func,
        "description": "Counterattack: A pattern where a gap in one direction is met by a candle in the opposite direction, implying a potential reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DARKCLOUDCOVER",
        "function": candle_darkcloudcover_func,
        "description": "Dark Cloud Cover: A bearish reversal pattern where a bearish candle opens above and closes below the midpoint of a preceding bullish candle.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DOJI",
        "function": candle_doji_func,
        "description": "Doji: Represents indecision in the market as the open and close are nearly equal; it may signal a reversal when seen with other indicators.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DOJISTAR",
        "function": candle_dojistar_func,
        "description": "Doji Star: A variation of the doji that appears with a gap, highlighting indecision and a possible imminent reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_DRAGONFLYDOJI",
        "function": candle_dragonflydoji_func,
        "description": "Dragonfly Doji: A bullish reversal signal with a long lower shadow and no upper shadow, indicating that buyers may step in.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ENGULFING",
        "function": candle_engulfing_func,
        "description": "Engulfing: A reversal pattern where one candle completely engulfs the previous candle’s body, suggesting a strong change in sentiment.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_EVENINGDOJISTAR",
        "function": candle_eveningdojistar_func,
        "description": "Evening Doji Star: A bearish reversal pattern where a doji follows a bullish candle, signaling potential weakness.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_EVENINGSTAR",
        "function": candle_eveningstar_func,
        "description": "Evening Star: A classic bearish reversal pattern formed by three candles, marking a potential shift from uptrend to downtrend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_GAPSIDESIDEWHITE",
        "function": candle_gapsidesidewhite_func,
        "description": "Gap Side-by-Side White Lines: Typically a continuation pattern where similar white candles with gaps indicate trend persistence.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_GRAVESTONEDOJI",
        "function": candle_gravestonedoji_func,
        "description": "Gravestone Doji: A bearish reversal signal with a long upper shadow and no lower shadow, suggesting a potential market top.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HAMMER",
        "function": candle_hammer_func,
        "description": "Hammer: A bullish reversal pattern in a downtrend, characterized by a small body and a long lower shadow indicating buying support.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HANGINGMAN",
        "function": candle_hangingman_func,
        "description": "Hanging Man: A bearish reversal pattern in an uptrend with a small body and long lower shadow, hinting at potential weakness.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HARAMI",
        "function": candle_harami_func,
        "description": "Harami: A reversal pattern where a small candle is contained within the previous candle’s body, suggesting a possible trend change.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HARAMICROSS",
        "function": candle_haramicross_func,
        "description": "Harami Cross: Similar to the Harami pattern but with a doji, indicating uncertainty and a potential reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HIGHWAVE",
        "function": candle_highwave_func,
        "description": "High-Wave Candle: Features long shadows with a small body, reflecting high volatility and market indecision.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HIKKAKE",
        "function": candle_hikkake_func,
        "description": "Hikkake: A trap pattern that identifies false breakouts and signals a potential reversal when the breakout fails.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HIKKAKEMOD",
        "function": candle_hikkakemod_func,
        "description": "Modified Hikkake: A refined version of the Hikkake pattern that offers a clearer reversal signal after a false breakout.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_HOMINGPIGEON",
        "function": candle_homingpigeon_func,
        "description": "Homing Pigeon: A bullish reversal pattern where a small bullish candle appears in a downtrend, suggesting a potential bottom.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_IDENTICAL3CROWS",
        "function": candle_identical3crows_func,
        "description": "Identical Three Crows: A bearish reversal pattern with three similarly shaped bearish candles, indicating strong selling pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_INNECK",
        "function": candle_inneck_func,
        "description": "In-Neck: A bearish reversal pattern where the close is near the previous candle’s low, hinting at continuing downward momentum.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_INVERTEDHAMMER",
        "function": candle_invertedhammer_func,
        "description": "Inverted Hammer: A bullish reversal pattern after a downtrend, marked by a small body with a long upper shadow that signals buying interest.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_KICKING",
        "function": candle_kicking_func,
        "description": "Kicking: A reversal pattern marked by a gap and a strong candle in the opposite direction, indicating a sharp change in market sentiment.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_KICKINGBYLENGTH",
        "function": candle_kickingbylength_func,
        "description": "Kicking by Length: A variant of the Kicking pattern that emphasizes longer candles, suggesting a more powerful reversal signal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_LADDERBOTTOM",
        "function": candle_ladderbottom_func,
        "description": "Ladder Bottom: A bullish reversal pattern that forms near support, indicating the potential end of a downtrend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_LONGLEGGEDDOJI",
        "function": candle_longleggeddoji_func,
        "description": "Long-Legged Doji: Exhibits a wide range with a nearly equal open and close, signifying indecision that may lead to a reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_LONGLINE",
        "function": candle_longline_func,
        "description": "Long Line Candle: A candle with a long body, indicating strong momentum in the prevailing trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MARUBOZU",
        "function": candle_marubozu_func,
        "description": "Marubozu: A candlestick with no shadows that reflects dominance by buyers or sellers, supporting trend continuation.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MATCHINGLOW",
        "function": candle_matchinglow_func,
        "description": "Matching Low: A bullish reversal pattern where the current low matches the previous low, hinting at a price bottom.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MATHOLD",
        "function": candle_mathold_func,
        "description": "Mat Hold: A bullish continuation pattern that shows a pause before an uptrend resumes, suggesting temporary consolidation.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MORNINGDOJISTAR",
        "function": candle_morningdojistar_func,
        "description": "Morning Doji Star: A bullish reversal pattern where a doji appears between a bearish and a bullish candle, indicating a potential bottom.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_MORNINGSTAR",
        "function": candle_morningstar_func,
        "description": "Morning Star: A well-known bullish reversal pattern formed by three candles that signals the end of a downtrend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_ONNECK",
        "function": candle_onneck_func,
        "description": "On-Neck: A bearish reversal pattern where the second candle closes near the first candle’s neck, suggesting emerging selling pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_PIERCING",
        "function": candle_piercing_func,
        "description": "Piercing: A bullish reversal pattern where a bearish candle is followed by a bullish candle that closes above the midpoint of the previous candle.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_RICKSHAWMAN",
        "function": candle_rickshawman_func,
        "description": "Rickshaw Man: A pattern similar to a doji that reflects indecision, potentially signaling an upcoming reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_RISEFALL3METHODS",
        "function": candle_risefall3methods_func,
        "description": "Rising/Falling Three Methods: A continuation pattern where a group of candles indicates that the current trend is still intact despite minor pullbacks.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SEPARATINGLINES",
        "function": candle_separatinglines_func,
        "description": "Separating Lines: A continuation pattern where similar candles appear consecutively, suggesting the trend will persist.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SHOOTINGSTAR",
        "function": candle_shootingstar_func,
        "description": "Shooting Star: A bearish reversal pattern in an uptrend, marked by a small body and long upper shadow, indicating potential trend weakness.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SHORTLINE",
        "function": candle_shortline_func,
        "description": "Short Line Candle: A candle with a narrow range that reflects low volatility and market indecision.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_SPINNINGTOP",
        "function": candle_spinningtop_func,
        "description": "Spinning Top: Characterized by a small body with long shadows, this pattern signals indecision and may precede a reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_STALLEDPATTERN",
        "function": candle_stalledpattern_func,
        "description": "Stalled Pattern: A reversal signal in an uptrend that indicates a loss of momentum and a potential market top.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_STICKSANDWICH",
        "function": candle_sticksandwich_func,
        "description": "Stick Sandwich: A continuation pattern where a small candle is ‘sandwiched’ between two similar candles, suggesting the trend will continue.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_TAKURI",
        "function": candle_takuri_func,
        "description": "Takuri: A pattern akin to a dragonfly doji with a long lower shadow, indicating a potential reversal when buyers gain control.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_TASUKIGAP",
        "function": candle_tasukigap_func,
        "description": "Tasuki Gap: A continuation pattern where a gap between candles is partially filled, supporting the ongoing trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_THRUSTING",
        "function": candle_thrusting_func,
        "description": "Thrusting: A bearish reversal pattern where a candle opens within the previous candle’s body and closes near its low, indicating selling pressure.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_TRISTAR",
        "function": candle_tristar_func,
        "description": "Tristar: A reversal pattern composed of three small candles that signal indecision and a potential change in trend.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_UNIQUE3RIVER",
        "function": candle_unique3river_func,
        "description": "Unique 3 River: A less common reversal pattern indicating a marked shift in momentum over three consecutive candles.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_UPSIDEGAP2CROWS",
        "function": candle_upsidegap2crows_func,
        "description": "Upside Gap Two Crows: A bearish reversal pattern where an initial gap up is followed by two bearish candles, suggesting a trend reversal.",
        "signal_function": interpret_candlestick,
    },
    {
        "name": "CANDLE_XSIDEGAP3METHODS",
        "function": candle_xsidegap3methods_func,
        "description": "X-Side Gap Three Methods: A continuation pattern identified by a gap and three supportive candles, indicating the current trend is likely to persist.",
        "signal_function": interpret_candlestick,
    },
]
ta_lib_volume = [
    {
        "name": "AD",
        "function": ad_func,
        "description": "Accumulation/Distribution (AD): Measures the cumulative flow of money into and out of a security based on price and volume.",
        "signal_function": signal_ad,
        "raw_function": ta.AD,
        "parameters": {},
        "parameter_space": {},
    },
    {
        "name": "ADX",
        "function": adx_func,
        "description": "Average Directional Index (ADX): Measures the strength of a trend but not its direction.",
        "signal_function": signal_adx,
        "raw_function": ta.ADX,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    },
    {
        "name": "ADXR",
        "function": adxr_func,
        "description": "Average Directional Movement Rating (ADXR): Smoothed version of ADX to confirm trend strength.",
        "signal_function": signal_adx,
        "raw_function": ta.ADXR,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    },
    {
        "name": "ADOSC",
        "function": adosc_func,
        "description": "Accumulation/Distribution Oscillator (ADOSC): A momentum oscillator based on the AD line to detect volume strength shifts.",
        "signal_function": signal_adosc,
        "raw_function": ta.ADOSC,
        "parameters": {"fastperiod": 3, "slowperiod": 10},
        "parameter_space": {
            "fastperiod": [3, 4, 5, 6, 7],
            "slowperiod": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        },
    },
    {
        "name": "OBV",
        "function": obv_func,
        "description": "On-Balance Volume (OBV): Tracks cumulative buying/selling pressure by adding volume on up days and subtracting it on down days.",
        "signal_function": signal_obv,
        "raw_function": ta.OBV,
        "parameters": {},
        "parameter_space": {},
    },
]
additional_volume = [
    {
        "name": "FantailVMA",
        "function": FantailVMA,
        "description": "Fantail VMA: A smoothed volume-weighted average that responds to price volatility and ADX to infer market momentum.",
        "signal_function": signal_fantail_vma,
        "raw_function": FantailVMA,
        "parameters": {"adx_length": 2, "weighting": 2.0, "ma_length": 1},
        "parameter_space": {
            "adx_length": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "weighting": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "ma_length": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
    {
        "name": "KVO",
        "function": KVO,
        "description": "Klinger Volume Oscillator (KVO): Combines price movements and volume to identify long-term trends of money flow.",
        "signal_function": signal_kvo,
        "raw_function": KVO,
        "parameters": {"fast_ema": 34, "slow_ema": 55, "signal_ema": 13},
        "parameter_space": {
            "fast_ema": [5, 10, 15, 20, 25, 30, 35, 40, 45],
            "slow_ema": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            "signal_ema": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        },
    },
    {
        "name": "LWPI",
        "function": LWPI,
        "description": "Larry Williams Proxy Index (LWPI): An oscillator evaluating buying/selling strength using price and volatility.",
        "signal_function": signal_lwpi,
        "raw_function": LWPI,
        "parameters": {"period": 8},
        "parameter_space": {"period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    },
    {
        "name": "NormalizedVolume",
        "function": NormalizedVolume,
        "description": "Normalized Volume: Expresses current volume as a percentage of the average over a given period.",
        "signal_function": signal_normalized_volume,
        "raw_function": NormalizedVolume,
        "parameters": {"period": 14},
        "parameter_space": {"period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    },
    {
        "name": "TwiggsMF",
        "function": TwiggsMF,
        "description": "Twiggs Money Flow: Combines price and volume to estimate the strength of accumulation or distribution.",
        "signal_function": signal_twiggs_mf,
        "raw_function": TwiggsMF,
        "parameters": {"period": 21},
        "parameter_space": {"period": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]},
    },
]
volume_indicators = ta_lib_volume + additional_volume
ta_lib_trend = [
    {
        "name": "AROON",
        "function": aroon_func,
        "description": "Aroon Down: Indicates how recently the lowest low occurred during a given period.",
        "signal_function": signal_aroon,
        "raw_function": ta.AROON,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [5, 7, 10, 12, 15, 17, 20, 22, 25, 30]},
    },
    {
        "name": "AROONOSC",
        "function": aroonosc_func,
        "description": "Aroon Oscillator: Measures the difference between Aroon Up and Down to gauge trend direction.",
        "signal_function": signal_aroonosc,
        "raw_function": ta.AROONOSC,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [5, 7, 10, 12, 15, 17, 20, 22, 25, 30]},
    },
]
additional_trend = [
    {
        "name": "KASE",
        "function": KASE,
        "description": "KASE Permission Stochastic: Smoothed stochastic oscillator used to assess trade permission based on trend momentum.",
        "signal_function": signal_kase,
        "raw_function": KASE,
        "parameters": {"pstLength": 9, "pstX": 5, "pstSmooth": 3, "smoothPeriod": 10},
        "parameter_space": {
            "pstLength": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            "pstX": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            "pstSmooth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "smoothPeriod": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        },
    },
    {
        "name": "KijunSen",
        "function": KijunSen,
        "description": "Kijun-sen (Base Line): Part of Ichimoku, provides key support/resistance levels and trend direction.",
        "signal_function": signal_kijunsen,
        "raw_function": KijunSen,
        "parameters": {"period": 26, "shift": 9},
        "parameter_space": {
            "period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "shift": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
    {
        "name": "KalmanFilter",
        "function": KalmanFilter,
        "description": "Kalman Filter: A statistical filter used to smooth price data and reveal trend dynamics.",
        "signal_function": signal_kalman_filter,
        "raw_function": KalmanFilter,
        "parameters": {"k": 1, "sharpness": 1},
        "parameter_space": {
            "k": [1, 2, 5, 10, 20, 50, 100],
            "sharpness": [1, 2, 5, 10, 20, 50, 100],
        },
    },
    {
        "name": "SSL",
        "function": SSL,
        "description": "SSL Channel: A trend-following indicator that uses smoothed high/low averages to generate crossovers.",
        "signal_function": signal_ssl,
        "raw_function": SSL,
        "parameters": {"period": 10},
        "parameter_space": {"period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    },
    {
        "name": "SuperTrend",
        "function": SuperTrend,
        "description": "SuperTrend: A trailing stop and trend direction indicator based on ATR.",
        "signal_function": signal_supertrend,
        "raw_function": SuperTrend,
        "parameters": {"period": 10, "multiplier": 3.0},
        "parameter_space": {
            "period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "multiplier": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
    {
        "name": "TrendLord",
        "function": TrendLord,
        "description": "TrendLord: A smoothed oscillator-based trend detector showing bullish/bearish strength zones.",
        "signal_function": signal_trendlord,
        "raw_function": TrendLord,
        "parameters": {
            "period": 12,
            "ma_method": "smma",
            "applied_price": "close",
            "show_high_low": False,
            "signal_bar": 1,
        },
        "parameter_space": {
            "period": [3, 5, 7, 9, 11, 13, 15, 17, 19],
            "ma_method": ["sma", "ema", "smma", "lwma"],
            "signal_bar": [1, 2, 3, 4, 5],
        },
    },
    {
        "name": "UF2018",
        "function": UF2018,
        "description": "UF2018: A visual zig-zag style trend-following indicator that shows direction based on local highs/lows.",
        "signal_function": signal_uf2018,
        "raw_function": UF2018,
        "parameters": {"period": 54},
        "parameter_space": {"period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    },
    {
        "name": "CenterOfGravity",
        "function": CenterOfGravity,
        "description": "Center of Gravity (COG): A zero-lag oscillator that forecasts turning points using weighted averages.",
        "signal_function": signal_center_of_gravity,
        "raw_function": CenterOfGravity,
        "parameters": {"period": 10},
        "parameter_space": {"period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    },
    {
        "name": "GruchaIndex",
        "function": GruchaIndex,
        "description": "Grucha Index: A visual trend indicator showing buyer vs. seller dominance over recent candles.",
        "signal_function": signal_grucha_index,
        "raw_function": GruchaIndex,
        "parameters": {"period": 10, "ma_period": 10},
        "parameter_space": {
            "period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "ma_period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        },
    },
    {
        "name": "HalfTrend",
        "function": HalfTrend,
        "description": "HalfTrend: A non-lagging trend reversal indicator that reacts only after sustained movement.",
        "signal_function": signal_halftrend,
        "raw_function": HalfTrend,
        "parameters": {"amplitude": 2},
        "parameter_space": {"amplitude": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    },
    {
        "name": "TopTrend",
        "function": TopTrend,
        "description": "TopTrend: Uses modified Bollinger logic to indicate trend reversals and trailing stop levels.",
        "signal_function": signal_top_trend,
        "raw_function": TopTrend,
        "parameters": {"period": 20, "deviation": 2, "money_risk": 1.0},
        "parameter_space": {
            "period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "deviation": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "money_risk": [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
        },
    },
]
trend_indicators = ta_lib_trend + additional_trend
ta_lib_momentum = [
    {
        "name": "APO",
        "function": apo_func,
        "description": "APO (Absolute Price Oscillator): Measures the difference between fast and slow EMAs of price to indicate momentum direction and strength.",
        "signal_function": signal_apo,
        "raw_function": ta.APO,
        "parameters": {"fastperiod": 12, "slowperiod": 26, "matype": 0},
        "parameter_space": {
            "fastperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "slowperiod": [16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            "matype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    {
        "name": "CCI",
        "function": cci_func,
        "description": "CCI (Commodity Channel Index): A momentum oscillator that measures deviation from a moving average to detect overbought or oversold conditions.",
        "signal_function": signal_cci,
        "raw_function": ta.CCI,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]},
    },
    {
        "name": "MACD",
        "function": macd_func,
        "description": "MACD: Combines the MACD line, signal line, and histogram to track momentum and detect entry signals via crossovers.",
        "signal_function": signal_macd,
        "raw_function": ta.MACD,
        "parameters": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "parameter_space": {
            "fastperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "slowperiod": [16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            "signalperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        },
    },
    {
        "name": "MACDEXT",
        "function": macdext_func,
        "description": "MACDEXT: An extended version of the MACD indicator with customizable moving average types.",
        "signal_function": signal_macdext,
        "raw_function": ta.MACDEXT,
        "parameters": {
            "fastperiod": 12,
            "fastmatype": 0,
            "slowperiod": 26,
            "slowmatype": 0,
            "signalperiod": 9,
            "signalmatype": 0,
        },
        "parameter_space": {
            "fastperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "fastmatype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "slowperiod": [16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            "slowmatype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "signalperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "signalmatype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    {
        "name": "MACDFIX",
        "function": macdfix_func,
        "description": "MACDFIX: A simplified MACD that uses a fixed signal period (typically 9), useful for consistent momentum detection.",
        "signal_function": signal_macdfix,
        "raw_function": ta.MACDFIX,
        "parameters": {"signalperiod": 9},
        "parameter_space": {"signalperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
    },
    {
        "name": "MAMA",
        "function": mama_func,
        "description": "MAMA: The MESA Adaptive Moving Average is designed to react faster to price changes using adaptive cycle techniques.",
        "signal_function": signal_mama,
        "raw_function": ta.MAMA,
        "parameters": {"fastlimit": 0.5, "slowlimit": 0.05},
        "parameter_space": {
            "fastlimit": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
            "slowlimit": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        },
    },
    {
        "name": "PPO",
        "function": ppo_func,
        "description": "PPO (Percentage Price Oscillator): A normalized MACD that measures momentum as a percentage rather than absolute price.",
        "signal_function": signal_ppo,
        "raw_function": ta.PPO,
        "parameters": {"fastperiod": 12, "slowperiod": 26, "matype": 0},
        "parameter_space": {
            "fastperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "slowperiod": [16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
            "matype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    {
        "name": "ROC",
        "function": roc_func,
        "description": "ROC (Rate of Change): Measures the percentage change in price over a defined period to assess momentum shifts.",
        "signal_function": signal_roc,
        "raw_function": ta.ROC,
        "parameters": {"timeperiod": 10},
        "parameter_space": {"timeperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
    },
    {
        "name": "ROCP",
        "function": rocp_func,
        "description": "ROCP (Rate of Change Percent): Measures the percent change in price over a specified period to detect momentum shifts.",
        "signal_function": signal_rocp,
        "raw_function": ta.ROCP,
        "parameters": {"timeperiod": 10},
        "parameter_space": {"timeperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
    },
    {
        "name": "ROCR",
        "function": rocr_func,
        "description": "ROCR (Rate of Change Ratio): Expresses price change as a ratio relative to a previous period, highlighting acceleration or deceleration.",
        "signal_function": signal_rocr,
        "raw_function": ta.ROCR,
        "parameters": {"timeperiod": 10},
        "parameter_space": {"timeperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
    },
    {
        "name": "ROCR100",
        "function": rocr100_func,
        "description": "ROCR100: Like ROCR, but scaled to 100. A value >100 indicates upward momentum; <100 suggests decline.",
        "signal_function": signal_rocr100,
        "raw_function": ta.ROCR100,
        "parameters": {"timeperiod": 10},
        "parameter_space": {"timeperiod": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
    },
    {
        "name": "RSI",
        "function": rsi_func,
        "description": "RSI (Relative Strength Index): Measures recent price gains versus losses to identify overbought or oversold conditions.",
        "signal_function": signal_rsi,
        "raw_function": ta.RSI,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]},
    },
    {
        "name": "STOCH_Slow",
        "function": stoch_slow_func,
        "description": "Stochastic Oscillator (Slow): Uses smoothed %K and %D crossovers to identify potential momentum reversals.",
        "signal_function": signal_stoch,
        "raw_function": ta.STOCH,
        "parameters": {
            "fastk_period": 5,
            "slowk_period": 3,
            "slowk_matype": 0,
            "slowd_period": 3,
            "slowd_matype": 0,
        },
        "parameter_space": {
            "fastk_period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "slowk_period": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "slowk_matype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "slowd_period": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "slowd_matype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    {
        "name": "STOCHF_Fast",
        "function": stochf_fast_func,
        "description": "Stochastic Oscillator (Fast): Uses raw %K and %D crossovers for faster, more sensitive momentum shifts.",
        "signal_function": signal_stoch,
        "raw_function": ta.STOCHF,
        "parameters": {"fastk_period": 5, "fastd_period": 3, "fastd_matype": 0},
        "parameter_space": {
            "fastk_period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "fastd_period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "fastd_matype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    {
        "name": "STOCHRSI",
        "function": stochrsi_func,
        "description": "Stochastic RSI: Applies stochastic logic to RSI values, enhancing detection of overbought/oversold extremes.",
        "signal_function": signal_stoch,
        "raw_function": ta.STOCHRSI,
        "parameters": {
            "timeperiod": 14,
            "fastk_period": 5,
            "fastd_period": 3,
            "fastd_matype": 0,
        },
        "parameter_space": {
            "timeperiod": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "fastk_period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "fastd_period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "fastd_matype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    {
        "name": "TSF",
        "function": tsf_func,
        "description": "TSF (Time Series Forecast): Projects a linear regression forward, estimating future price direction.",
        "signal_function": signal_tsf,
        "raw_function": ta.TSF,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]},
    },
    {
        "name": "ULTOSC",
        "function": ultosc_func,
        "description": "ULTOSC (Ultimate Oscillator): Combines multiple timeframes of momentum into one oscillator to reduce false signals.",
        "signal_function": signal_ultosc,
        "raw_function": ta.ULTOSC,
        "parameters": {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28},
        "parameter_space": {
            "timeperiod1": [3, 4, 5, 6, 7, 8, 9, 10],
            "timeperiod2": [12, 14, 16, 18, 20, 22, 24],
            "timeperiod3": [26, 28, 30, 32, 34, 36, 38],
        },
    },
]
additional_momentum = [
    {
        "name": "MACDZeroLag",
        "function": MACDZeroLag,
        "description": "MACDZeroLag: A variation of MACD designed to reduce lag by applying zero-lag moving averages for better signal timing.",
        "signal_function": signal_macd_zero_lag,
        "raw_function": MACDZeroLag,
        "parameters": {"short_period": 12, "long_period": 26, "signal_period": 9},
        "parameter_space": {
            "short_period": [1, 4, 7, 10, 13, 16, 19, 22],
            "long_period": [24, 27, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56],
            "signal_period": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        },
    },
    {
        "name": "Fisher",
        "function": Fisher,
        "description": "Fisher Transform: Sharpens turning points in price action using a mathematical transformation of normalized prices.",
        "signal_function": signal_fisher,
        "raw_function": Fisher,
        "parameters": {
            "range_periods": 10,
            "price_smoothing": 0.3,
            "index_smoothing": 0.3,
        },
        "parameter_space": {
            "range_periods": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "price_smoothing": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "index_smoothing": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    {
        "name": "BullsBearsImpulse",
        "function": BullsBearsImpulse,
        "description": "Bulls/Bears Impulse: Measures the relative strength of bullish and bearish forces to highlight dominant sentiment.",
        "signal_function": signal_bulls_bears_impulse,
        "raw_function": BullsBearsImpulse,
        "parameters": {"ma_period": 13},
        "parameter_space": {"ma_period": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]},
    },
    {
        "name": "J_TPO",
        "function": J_TPO,
        "description": "J_TPO: A custom oscillator derived from time-price opportunity modeling to reflect short-term velocity and acceleration.",
        "signal_function": signal_j_tpo,
        "raw_function": J_TPO,
        "parameters": {"period": 14},
        "parameter_space": {"period": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]},
    },
    {
        "name": "Laguerre",
        "function": Laguerre,
        "description": "Laguerre Filter: A smooth oscillator designed to track price momentum while minimizing whipsaws.",
        "signal_function": signal_laguerre,
        "raw_function": Laguerre,
        "parameters": {"gamma": 0.7},
        "parameter_space": {"gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    },
    {
        "name": "SchaffTrendCycle",
        "function": SchaffTrendCycle,
        "description": "Schaff Trend Cycle: Combines MACD and cycle theory to create a responsive momentum oscillator.",
        "signal_function": signal_schaff_trend_cycle,
        "raw_function": SchaffTrendCycle,
        "parameters": {
            "period": 10,
            "fast_ma_period": 23,
            "slow_ma_period": 50,
            "signal_period": 3,
        },
        "parameter_space": {
            "period": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "fast_ma_period": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28],
            "slow_ma_period": [29, 32, 35, 38, 41, 44, 47, 50, 53, 56],
            "signal_period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
    {
        "name": "TDFI",
        "function": TDFI,
        "description": "TDFI (Trend Direction & Force Index): Captures the intensity and direction of price moves for momentum analysis.",
        "signal_function": signal_tdfi,
        "raw_function": TDFI,
        "parameters": {"period": 13},
        "parameter_space": {"period": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]},
    },
    {
        "name": "TTF",
        "function": TTF,
        "description": "TTF (Trend Trigger Factor): A smoothed oscillator designed to confirm momentum reversals through threshold crossings.",
        "signal_function": signal_ttf,
        "raw_function": TTF,
        "parameters": {
            "period": 8,
            "top_line": 75,
            "bottom_line": -75,
            "t3_period": 3,
            "b": 0.7,
        },
        "parameter_space": {
            "period": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28],
            "top_line": [50, 60, 70, 80, 90, 100],
            "bottom_line": [-50, -60, -70, -80, -90, -100],
            "t3_period": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    {
        "name": "AcceleratorLSMA",
        "function": AcceleratorLSMA,
        "description": "Accelerator LSMA: Combines LSMA smoothing with acceleration logic to detect momentum shifts early.",
        "signal_function": signal_accelerator_lsma,
        "raw_function": AcceleratorLSMA,
        "parameters": {"long_period": 30, "short_period": 10},
        "parameter_space": {
            "long_period": [24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
            "short_period": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        },
    },
]
momentum_indicators = ta_lib_momentum + additional_momentum
ta_lib_volatility = [
    {
        "name": "TRANGE",
        "function": trange_func,
        "description": "True Range (TRANGE): Raw measure of price range and gap movement, used as a base for ATR.",
        "signal_function": signal_volatility_line,
        "raw_function": ta.TRANGE,
        "parameters": {},
        "parameter_space": {},
    },
    {
        "name": "BollingerBands",
        "function": bollingerbands_func,
        "description": "Bollinger Bands: A volatility-based envelope plotted at standard deviations above and below a moving average. Returns upper, middle, and lower bands.",
        "signal_function": signal_bollinger,
        "raw_function": ta.BBANDS,
        "parameters": {"timeperiod": 5, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
        "parameter_space": {
            "timeperiod": [1, 3, 5, 7, 10, 12, 15, 18, 20, 22],
            "nbdevup": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "nbdevdn": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "matype": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },
    {
        "name": "STDDEV",
        "function": stddev_func,
        "description": "Standard Deviation (STDDEV): Measures price dispersion from the mean to gauge volatility.",
        "signal_function": signal_stddev,
        "raw_function": ta.STDDEV,
        "parameters": {"timeperiod": 5, "nbdev": 1},
        "parameter_space": {
            "timeperiod": [1, 3, 5, 7, 10, 12, 15, 18, 20, 22],
            "nbdev": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
    {
        "name": "VAR",
        "function": var_func,
        "description": "Variance (VAR): Square of standard deviation; tracks volatility by measuring price fluctuation strength.",
        "signal_function": signal_var,
        "raw_function": ta.VAR,
        "parameters": {"timeperiod": 5, "nbdev": 1},
        "parameter_space": {
            "timeperiod": [1, 3, 5, 7, 10, 12, 15, 18, 20, 22],
            "nbdev": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    },
]
additional_volatility = [
    {
        "name": "VolatilityRatio",
        "function": VolatilityRatio,
        "description": "Volatility Ratio: Compares recent price deviation against a longer-period range to measure relative volatility shifts.",
        "signal_function": signal_volatility_ratio,
        "raw_function": VolatilityRatio,
        "parameters": {"period": 25, "inp_price": "Close"},
        "parameter_space": {
            "period": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        },
    },
    {
        "name": "WAE",
        "function": WAE,
        "description": "Waddah Attar Explosion (WAE): Combines MACD-based momentum with Bollinger Band expansion to highlight explosive volatility phases.",
        "signal_function": signal_wae,
        "raw_function": WAE,
        "parameters": {"minutes": 0, "sensitivity": 150, "dead_zone_pip": 15},
        "parameter_space": {
            "minutes": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            "sensitivity": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
            "dead_zone_pip": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        },
    },
]

atr_indicators = [
    {
        "name": "ATR",
        "function": atr_func,
        "description": "Average True Range (ATR): Measures absolute volatility by accounting for gaps and high-low range.",
        "signal_function": signal_volatility_line,
        "raw_function": ta.ATR,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]},
    },
    {
        "name": "NATR",
        "function": natr_func,
        "description": "Normalized ATR (NATR): ATR expressed as a percentage of price, useful for cross-asset volatility comparison.",
        "signal_function": signal_volatility_line,
        "raw_function": ta.NATR,
        "parameters": {"timeperiod": 14},
        "parameter_space": {"timeperiod": [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]},
    },
    {
        "name": "FilteredATR",
        "function": FilteredATR,
        "description": "Filtered ATR: A smoothed version of the ATR to reduce noise and better reflect sustained volatility.",
        "signal_function": signal_filtered_atr,
        "raw_function": FilteredATR,
        "parameters": {"period": 34, "ma_period": 34, "ma_shift": 0},
        "parameter_space": {
            "period": [26, 28, 30, 32, 34, 36, 38, 40, 42, 44],
            "ma_period": [24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
            "ma_shift": [-2, -1, 0, 1, 2],
        },
    },
]

volatility_indicators = ta_lib_volatility + additional_volatility

ta_lib_price = [
    {
        "name": "AVGPRICE",
        "function": avgprice_func,
        "description": "Average Price: The average of the open, high, low, and close prices. A smoothed central value.",
        "signal_function": signal_avgprice,
    },
    {
        "name": "MEDPRICE",
        "function": medprice_func,
        "description": "Median Price: The midpoint between the high and low price.",
        "signal_function": signal_medprice,
    },
    {
        "name": "MAX",
        "function": max_func,
        "description": "Maximum Price: Highest close price over the given period.",
        "signal_function": signal_max,
    },
    {
        "name": "MAXINDEX",
        "function": maxindex_func,
        "description": "Max Index: Index of the maximum closing price over the time period.",
        "signal_function": signal_maxindex,
    },
    {
        "name": "MIDPOINT",
        "function": midpoint_func,
        "description": "Midpoint: Average of highest and lowest close price over a time period.",
        "signal_function": signal_midpoint,
    },
    {
        "name": "MIDPRICE",
        "function": midprice_func,
        "description": "Mid Price: Average of high and low prices over the specified period.",
        "signal_function": signal_midprice,
    },
    {
        "name": "MIN",
        "function": min_func,
        "description": "Minimum Price: Lowest close price over the given period.",
        "signal_function": signal_min,
    },
    {
        "name": "MININDEX",
        "function": minindex_func,
        "description": "Min Index: Index of the minimum close price over the time period.",
        "signal_function": signal_minindex,
    },
    {
        "name": "MINMAX",
        "function": minmax_func,
        "description": "MINMAX: Returns the lowest and highest close price over a given period, useful for detecting range and breakouts.",
        "signal_function": signal_minmax,
    },
    {
        "name": "SUM",
        "function": sum_func,
        "description": "Sum: Total of close prices over the defined time period.",
        "signal_function": signal_sum,
    },
    {
        "name": "TYPPRICE",
        "function": typprice_func,
        "description": "Typical Price: Average of high, low, and close. Reflects a representative transaction price.",
        "signal_function": signal_typprice,
    },
    {
        "name": "WCLPRICE",
        "function": wclprice_func,
        "description": "Weighted Close Price: Weighted average price placing more emphasis on the close.",
        "signal_function": signal_wclprice,
    },
]
price_indicators = ta_lib_price
baseline_indicators = [
    {
        "name": "ALMA",
        "function": ALMA,
        "description": "ALMA (Arnaud Legoux Moving Average): A Gaussian-weighted moving average designed to reduce lag while smoothing price.",
        "signal_function": signal_baseline_standard,
        "raw_function": ALMA,
        "parameters": {"period": 9, "sigma": 6, "offset": 0.85},
        "parameter_space": {
            "period": [3, 5, 7, 9, 11, 13, 15, 17, 19],
            "sigma": [3, 4, 5, 6, 7, 8, 9, 10, 11],
            "offset": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        },
    },
    {
        "name": "HMA",
        "function": HMA,
        "description": "HMA (Hull Moving Average): Uses weighted moving averages to minimize lag and smooth price action responsively.",
        "signal_function": signal_baseline_standard,
        "raw_function": HMA,
        "parameters": {"period": 13},
        "parameter_space": {
            "period": [5, 7, 9, 11, 13, 15, 17, 19, 21],
        },
    },
    {
        "name": "RecursiveMA",
        "function": RecursiveMA,
        "description": "Recursive MA: Applies repeated exponential smoothing to create a stable baseline for trend analysis.",
        "signal_function": signal_recursive_ma,
        "raw_function": RecursiveMA,
        "parameters": {"period": 2, "recursions": 20},
        "parameter_space": {
            "period": [1, 2, 3, 4, 5],
            "recursions": [10, 15, 20, 25, 30],
        },
    },
    {
        "name": "LSMA",
        "function": LSMA,
        "description": "LSMA (Least Squares Moving Average): A regression-based average used to project smoothed directional bias.",
        "signal_function": signal_lsma,
        "raw_function": LSMA,
        "parameters": {"period": 14, "shift": 0},
        "parameter_space": {
            "period": [3, 5, 7, 9, 11, 13, 15, 17, 19],
            "shift": [0],
        },
    },
    {
        "name": "VIDYA",
        "function": VIDYA,
        "description": "VIDYA (Variable Index Dynamic Average): Adaptive moving average that responds to volatility by adjusting smoothing.",
        "signal_function": signal_baseline_standard,
        "raw_function": VIDYA,
        "parameters": {"period": 9, "histper": 30},
        "parameter_space": {
            "period": [3, 5, 7, 9, 11, 13, 15],
            "histper": [5, 10, 15, 20, 25],
        },
    },
    {
        "name": "Gen3MA",
        "function": Gen3MA,
        "description": "Gen3MA: A third-generation moving average that combines multi-scale smoothing and sampling.",
        "signal_function": signal_gen3ma,
        "raw_function": Gen3MA,
        "parameters": {"period": 220, "sampling_period": 50},
        "parameter_space": {
            "period": [50, 100, 150, 200, 250],
            "sampling_period": [10, 20, 30, 40, 50],
        },
    },
    {
        "name": "TrendLord",
        "function": TrendLord,
        "description": "TrendLord: A smoothed MA-based trend filter that dynamically reacts to shifts using layered moving average logic.",
        "signal_function": signal_trendlord,
        "raw_function": TrendLord,
        "parameters": {
            "period": 12,
            "ma_method": "smma",
            "applied_price": "close",
            "show_high_low": False,
            "signal_bar": 1,
        },
        "parameter_space": {
            "period": [3, 5, 7, 9, 11, 13, 15, 17, 19],
            "ma_method": ["sma", "ema", "smma", "lwma"],
            "signal_bar": [1, 2, 3, 4, 5],
        },
    },
    {
        "name": "BraidFilter",
        "function": BraidFilter,
        "description": "Braid Filter: Combines multiple moving averages and separation conditions to confirm trend stability or congestion.",
        "signal_function": signal_braidfilter,
        "raw_function": BraidFilter,
        "parameters": {
            "period1": 5,
            "period2": 8,
            "period3": 20,
            "pips_min_sep_percent": 0.5,
        },
        "parameter_space": {
            "period1": [3, 4, 5, 6],
            "period2": [7, 8, 9, 10],
            "period3": [14, 16, 18, 20, 22, 24, 26],
            "pips_min_sep_percent": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    },
    {
        "name": "AcceleratorLSMA",
        "function": AcceleratorLSMA,
        "description": "Accelerator LSMA: Applies differential velocity of LSMA to identify acceleration or deceleration in trend bias.",
        "signal_function": signal_accelerator_lsma,
        "raw_function": AcceleratorLSMA,
        "parameters": {"long_period": 21, "short_period": 9},
        "parameter_space": {
            "long_period": [15, 18, 21, 24, 27, 30],
            "short_period": [3, 5, 7, 9, 11, 13],
        },
    },
    {
        "name": "McGinleyDI",
        "function": McGinleyDI,
        "description": "McGinley Dynamic Index: An adaptive moving average that self-adjusts for speed and volatility.",
        "signal_function": signal_baseline_standard,
        "raw_function": McGinleyDI,
        "parameters": {"period": 12, "mcg_constant": 5},
        "parameter_space": {
            "period": [3, 5, 7, 9, 11, 13, 15],
            "mcg_constant": [3, 4, 5, 6, 7],
        },
    },
]
all_indicators = (
    ta_lib_candlestick
    + volume_indicators
    + trend_indicators
    + momentum_indicators
    + volatility_indicators
    + price_indicators
    + baseline_indicators
    + atr_indicators
)
