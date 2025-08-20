import logging
from typing import Callable, TypedDict

import pandas as pd
import talib as ta

from scripts.indicators.calculation_functions import (
    ALMA,
    ASO,
    HMA,
    J_TPO,
    KASE,
    KVO,
    LSMA,
    LWPI,
    RWIBTF,
    SSL,
    T3MA,
    TDFI,
    TII,
    TP,
    TRENDAKKAM,
    TTF,
    TTMS,
    UF2018,
    VIDYA,
    WAE,
    AcceleratorLSMA,
    AdaptiveSmootherTriggerlinesMtfAlertsNmc,
    ATRBasedEMAVariant1,
    BamsBung3,
    BandPassFilter,
    BraidFilter,
    BullsBearsImpulse,
    CenterOfGravity,
    Chandelierexit,
    Coral,
    CVIMulti,
    CyberCycle,
    DecyclerOscillator,
    DetrendedSyntheticPriceGoscillators,
    DodaStochasticModified,
    DorseyInertia,
    DpoHistogramIndicator,
    EhlersDELIDetrendedLeadingIndicator,
    EhlersEarlyOnsetTrend,
    EhlersReverseEMA,
    EhlersRoofingFilterA,
    EhlersTwoPoleSuperSmootherFilter,
    ErgodicTVI,
    FantailVMA,
    FilteredATR,
    Fisher,
    Forecast,
    FramaIndicator,
    Gchannel,
    Gd,
    Gen3MA,
    GeominMA,
    GlitchIndexFixed,
    GruchaIndex,
    Hacolt202Lines,
    HalfTrend,
    Hlctrend,
    ISCalculation,
    JpOscillator,
    KalmanFilter,
    KijunSen,
    Laguerre,
    MACDZeroLag,
    McGinleyDI,
    McginleyDynamic23,
    MetroAdvanced,
    METROFixed,
    MomentumCandlesModifiedWAtr,
    MomentumCandlesWATR,
    NormalizedVolume,
    PrecisionTrendHistogram,
    PriceMomentumOscillator,
    QQEWithAlerts,
    RangeFilterModified,
    RecursiveMA,
    SchaffTrendCycle,
    SecondOrderGaussianHighPassFilterMtfZones,
    SherifHilo,
    Silence,
    Sinewma,
    SmoothedMomentum,
    Smoothstep,
    StiffnessIndicator,
    SuperTrend,
    TetherLine,
    TheTurtleTradingChannel,
    ThirdGenMA,
    TopBottomNR,
    TopTrend,
    Trendcontinuation2,
    TrendDirectionForceIndexSmoothed4,
    TrendLord,
    TrendLordNrpIndicator,
    Trimagen,
    TrueStrengthIndex,
    TwiggsMF,
    VolatilityRatio,
    VortexIndicator,
    WilliamVixFix,
    WprMaAlerts,
    XmaColouredUpdatedForNnfx,
    ZerolagMACDMq4,
)
from scripts.indicators.signal_functions import *

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


def cmo_func(df, **kwargs):
    return ta.CMO(df["Close"], **kwargs)


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
        "parameter_space": {"period": [30, 34, 38, 42, 46, 50, 54, 58, 62, 66]},
    },
    {
        "name": "CenterOfGravity",
        "function": CenterOfGravity,
        "description": "Center of Gravity (COG): A zero-lag oscillator that forecasts turning points using weighted averages.",
        "signal_function": signal_center_of_gravity,
        "raw_function": CenterOfGravity,
        "parameters": {"period": 10},
        "parameter_space": {"period": [6, 8, 10, 12, 14, 16, 18, 20]},
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
            # "deviation": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # "money_risk": [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
        },
    },
    {
        "name": "2nd Order Gaussian High Pass Filter MTF Zones",
        "function": SecondOrderGaussianHighPassFilterMtfZones,
        "signal_function": signal_2nd_order_gaussian_high_pass_filter_mtf_zones,
        "raw_function": SecondOrderGaussianHighPassFilterMtfZones,
        "description": "* A high-pass filter using Gaussian coefficients to remove low-frequency (trend) components and emphasise short-term fluctuations.",
        "parameters": {
            "alpha": 0.14,
            "timeframe": None,
            "interpolate": True,
            "maxbars": 2000,
        },
        "parameter_space": {
            "alpha": [0.05, 0.1, 0.14, 0.2, 0.3],
            # "timeframe": [None, "5T", "15T", "1H", "4H", "1D"],
            # "interpolate": [True, False],
        },
    },
    {
        "name": "3rdgenma",
        "function": ThirdGenMA,
        "signal_function": signal_3rdgenma,
        "raw_function": ThirdGenMA,
        "description": "Third Generation Moving Average (Durschner). Two-pass MA with adaptive alpha to reduce lag while preserving smoothness. Supports SMA, EMA, SMMA, LWMA and multiple applied prices; outputs MA3G and MA1.",
        "parameters": {
            "ma_period": 220,
            "sampling_period": 50,
            "method": 1,
            "applied_price": 5,
        },
        "parameter_space": {
            "ma_period": [180, 220, 260, 300, 360],
            "sampling_period": [30, 40, 50, 60, 70],
            "method": [0, 1, 2, 3],
            # "applied_price": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "AdaptiveSmootherTriggerlinesMtfAlertsNmc",
        "function": AdaptiveSmootherTriggerlinesMtfAlertsNmc,
        "signal_function": signal_adaptive_smoother_triggerlines_mtf_alerts_nmc,
        "raw_function": AdaptiveSmootherTriggerlinesMtfAlertsNmc,
        "description": "Adaptive smoother (iSmooth) with dynamic period from std-dev ratio; outputs current lsma, previous-bar lsma (lwma), slope-based trends (+1/-1), and optional multi-color segmented down-trend series.",
        "parameters": {
            "LsmaPeriod": 50,
            "LsmaPrice": 0,
            "AdaptPeriod": 21,
            "MultiColor": True,
        },
        "parameter_space": {
            "LsmaPeriod": [10, 21, 34, 50, 100],
            # "LsmaPrice": [0, 1, 2, 3, 4, 5, 6],
            "AdaptPeriod": [10, 14, 21, 30, 50],
            # "MultiColor": [True, False],
        },
    },
    {
        "name": "ASO",
        "function": ASO,
        "signal_function": signal_aso,
        "raw_function": ASO,
        "description": "* This oscillator attempts to quantify market sentiment by averaging momentum readings from several timeframes.",
        "parameters": {"period": 10, "mode": 0, "bulls": True, "bears": True},
        "parameter_space": {
            "period": [6, 8, 10, 12, 14, 16, 18, 20],
            # "mode": [0, 1, 2],
            # "bulls": [True, False],
            # "bears": [True, False],
        },
    },
    {
        "name": "Band Pass Filter",
        "function": BandPassFilter,
        "signal_function": signal_band_pass_filter,
        "raw_function": BandPassFilter,
        "description": "* Passes price components within a target frequency band (cycle emphasis) while attenuating trend and high-frequency noise; includes slope/trend-segmented histogram outputs.",
        "parameters": {"period": 50, "price": "median", "delta": 0.1},
        "parameter_space": {
            "period": [20, 30, 40, 50, 60, 80],
            "price": [
                "median",
                "close",
                "typical",
                "weighted",
                "average",
                "ha_median",
                "ha_typical",
                "ha_weighted",
                "hab_median",
            ],
            # "delta": [0.05, 0.1, 0.15, 0.2],
        },
    },
    {
        "name": "ChandelierExit",
        "function": Chandelierexit,
        "signal_function": signal_chandelierexit,
        "raw_function": Chandelierexit,
        "description": "* A trailing stop indicator developed by Chuck LeBeau.  It places the stop a multiple of the Average True Range (ATR) away from the highest high (for long trades) or lowest low (for short trades).",
        "parameters": {"lookback": 7, "atr_period": 9, "atr_mult": 2.5, "shift": 0},
        "parameter_space": {
            "lookback": [5, 7, 9, 11, 13, 15],
            "atr_period": [7, 9, 14, 21],
            "atr_mult": [2.0, 2.5, 3.0, 3.5],
        },
    },
    {
        "name": "Coral",
        "function": Coral,
        "signal_function": signal_coral,
        "raw_function": Coral,
        "description": "* A linear lag removal (LLR) filter that produces a very smooth moving average with minimal delay. Often called the “Coral Trend Indicator.”",
        "parameters": {"length": 34, "coef": 0.4},
        "parameter_space": {"length": [21, 34, 55, 89], "coef": [0.3, 0.4, 0.5]},
    },
    {
        "name": "cyber-cycle",
        "function": CyberCycle,
        "signal_function": signal_cyber_cycle,
        "raw_function": CyberCycle,
        "description": "* John Ehlers’ Cyber Cycle indicator applies smoothing and differentiation to extract a cyclical component from price. It oscillates around zero.",
        "parameters": {"alpha": 0.07, "price": "hl2"},
        "parameter_space": {
            # "alpha": [0.03, 0.05, 0.07, 0.1, 0.15],
            "price": ["close", "hl2", "hlc3", "ohlc4", "wclose"],
        },
    },
    {
        "name": "Decycler Oscillator",
        "function": DecyclerOscillator,
        "signal_function": signal_decycler_oscillator,
        "raw_function": DecyclerOscillator,
        "description": "John Ehlers-style decycler oscillator (two-pass) that removes dominant cycle components and highlights trend; includes segmented down-trend buffers.",
        "parameters": {
            "hp_period": 125,
            "k": 1.0,
            "hp_period2": 100,
            "k2": 1.2,
            "price": "close",
        },
        "parameter_space": {
            "hp_period": [75, 100, 125, 150, 200],
            # "k": [0.5, 1.0, 1.5, 2.0],
            "hp_period2": [40, 60, 80, 100, 125],
            # "k2": [0.8, 1.0, 1.2, 1.5],
            "price": [
                "close",
                "median",
                "typical",
                "weighted",
                "average",
                "tbiased",
                "haclose",
                "hamedian",
                "haweighted",
            ],
        },
    },
    {
        "name": "DetrendedSyntheticPriceGoscillators",
        "function": DetrendedSyntheticPriceGoscillators,
        "signal_function": signal_detrended_synthetic_price_goscillators,
        "raw_function": DetrendedSyntheticPriceGoscillators,
        "description": "* This indicator subtracts a moving average from a “synthetic price” (a weighted price) to remove trend and highlight cycles.",
        "parameters": {
            "dsp_period": 14,
            "price_mode": "median",
            "signal_period": 9,
            "color_on": "outer",
        },
        "parameter_space": {
            "dsp_period": [7, 14, 21, 30],
            "price_mode": [
                "close",
                "open",
                "high",
                "low",
                "median",
                "medianb",
                "typical",
                "weighted",
                "average",
                "tbiased",
                "tbiased2",
                "haclose",
                "haopen",
                "hahigh",
                "halow",
                "hamedian",
                "hamedianb",
                "hatypical",
                "haweighted",
                "haaverage",
                "hatbiased",
                "hatbiased2",
            ],
            "signal_period": [5, 9, 13, 21],
            # "color_on": ["outer", "outer2", "zero", "slope"],
        },
    },
    {
        "name": "Doda-Stochastic-Modified",
        "function": DodaStochasticModified,
        "signal_function": signal_doda_stochastic_modified,
        "raw_function": DodaStochasticModified,
        "description": "A modified stochastic oscillator: EMA of Close -> stochastic (0-100) over Pds -> EMA (Slw) -> signal EMA (Slwsignal).",
        "parameters": {"Slw": 8, "Pds": 13, "Slwsignal": 9},
        "parameter_space": {
            "Slw": [5, 8, 13, 21],
            "Pds": [9, 13, 21, 34],
            "Slwsignal": [5, 9, 13, 18],
        },
    },
    {
        "name": "dorsey-inertia",
        "function": DorseyInertia,
        "signal_function": signal_dorsey_inertia,
        "raw_function": DorseyInertia,
        "description": "Dorsey Inertia (Mladen): EMA-averaged up/down rolling-std RVI components with final SMA smoothing.",
        "parameters": {"rvi_period": 10, "avg_period": 14, "smoothing_period": 20},
        "parameter_space": {
            "rvi_period": [5, 8, 10, 14, 20, 30],
            "avg_period": [8, 10, 14, 21, 30],
            "smoothing_period": [10, 14, 20, 30, 50],
        },
    },
    {
        "name": "Ehlers_DELI_(Detrended_Leading_Indicator)",
        "function": EhlersDELIDetrendedLeadingIndicator,
        "signal_function": signal_ehlers_deli__detrended_leading_indicator,
        "raw_function": EhlersDELIDetrendedLeadingIndicator,
        "description": "* A detrended oscillator by John Ehlers that leads price by removing lag and emphasising turning points.",
        "parameters": {"period": 14},
        "parameter_space": {"period": [7, 14, 21, 30]},
    },
    {
        "name": "Ehlers DELI (Detrended Leading Indicator)",
        "function": EhlersDELIDetrendedLeadingIndicator,
        "signal_function": signal_ehlers_deli_detrended_leading_indicator,
        "raw_function": EhlersDELIDetrendedLeadingIndicator,
        "description": "* A detrended oscillator by John Ehlers that leads price by removing lag and emphasising turning points.",
        "parameters": {"period": 14},
        "parameter_space": {"period": [7, 14, 21, 30]},
    },
    {
        "name": "Ehlers Early Onset Trend",
        "function": EhlersEarlyOnsetTrend,
        "signal_function": signal_ehlers_early_onset_trend,
        "raw_function": EhlersEarlyOnsetTrend,
        "description": "A John Ehlers indicator designed to detect the early onset of trends using high-pass filtering, adaptive smoothing, and a quotient transform. Returns two lines (EEOT_Q1, EEOT_Q2).",
        "parameters": {"period": 20, "q1": 0.8, "q2": 0.4},
        "parameter_space": {
            "period": [10, 20, 30, 50],
            # "q1": [0.6, 0.8, 0.9],
            # "q2": [0.2, 0.4, 0.6],
        },
    },
    {
        "name": "Ehlers Reverse EMA",
        "function": EhlersReverseEMA,
        "signal_function": signal_ehlers_reverse_ema,
        "raw_function": EhlersReverseEMA,
        "description": "* A variant of the exponential moving average that applies weighting in reverse order (most recent prices receive the least weight). This aims to anticipate turning points.",
        "parameters": {"alpha": 0.1},
        "parameter_space": {"alpha": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]},
    },
    {
        "name": "EhlersRoofingFilterA",
        "function": EhlersRoofingFilterA,
        "signal_function": signal_ehlersroofingfiltera,
        "raw_function": EhlersRoofingFilterA,
        "description": "John Ehlers' roofing filter combining a high-pass and low-pass to isolate tradable trends; returns rfilt, trigger, hp, up, down.",
        "parameters": {
            "hp_length": 80,
            "lp_length": 40,
            "arrow_distance": 100.0,
            "point": 1.0,
        },
        "parameter_space": {
            "hp_length": [40, 60, 80, 100, 120],
            "lp_length": [20, 30, 40, 50, 60],
        },
    },
    {
        "name": "forecast",
        "function": Forecast,
        "signal_function": signal_forecast,
        "raw_function": Forecast,
        "description": "* The Forecast Oscillator measures the difference between the actual price and its regression forecast value, expressed as a percentage of price.",
        "parameters": {"length": 20, "price": 0},
        "parameter_space": {
            "length": [10, 14, 20, 30, 50],
            "price": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "frama-indicator",
        "function": FramaIndicator,
        "signal_function": signal_frama_indicator,
        "raw_function": FramaIndicator,
        "description": "Developed by John Ehlers, the Fractal Adaptive Moving Average adapts its smoothing via the fractal dimension: faster in trends, slower in choppy markets.",
        "parameters": {"period": 10, "price_type": 0},
        "parameter_space": {
            "period": [5, 10, 14, 20, 30],
            "price_type": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "GeominMA",
        "function": GeominMA,
        "signal_function": signal_geomin_ma,
        "raw_function": GeominMA,
        "description": "Geometric Mean Moving Average indicator.",
        "parameters": {"length": 10, "price": 0},
        "parameter_space": {
            "length": [5, 10, 14, 20, 30, 50],
            "price": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "hacolt_2.02_lines",
        "function": Hacolt202Lines,
        "signal_function": signal_hacolt_2_02_lines,
        "raw_function": Hacolt202Lines,
        "description": "Long-term Heikin-Ashi Candlestick Oscillator (HACO lt) 2.02 lines by Sylvain Vervoort.",
        "parameters": {"Length": 55, "CandleSize": 1.1, "LtLength": 60},
        "parameter_space": {
            "Length": [34, 55, 70, 89],
            "CandleSize": [0.9, 1.0, 1.1, 1.2, 1.3],
            "LtLength": [40, 50, 60, 80, 100],
        },
    },
    {
        "name": "HLCTrend",
        "function": Hlctrend,
        "signal_function": signal_hlctrend,
        "raw_function": Hlctrend,
        "description": "Trend indicator using EMAs of Close, Low, and High; outputs two lines: EMA(Close)-EMA(High) and EMA(Low)-EMA(Close).",
        "parameters": {"close_period": 5, "low_period": 13, "high_period": 34},
        "parameter_space": {
            "close_period": [3, 5, 8],
            "low_period": [8, 13, 21],
            "high_period": [34, 55, 89],
        },
    },
    {
        "name": "Metro-Advanced",
        "function": MetroAdvanced,
        "signal_function": signal_metro_advanced,
        "raw_function": MetroAdvanced,
        "description": "RSI-based step oscillator with fast/slow step channels and adaptive levels; provides a simple metro-style trend/momentum display.",
        "parameters": {
            "period_rsi": 14,
            "rsi_type": "rsi",
            "price": "close",
            "step_size_fast": 5.0,
            "step_size_slow": 15.0,
            "over_sold": 10.0,
            "over_bought": 90.0,
            "minmax_period": 49,
        },
        "parameter_space": {
            "period_rsi": [7, 14, 21, 28],
            "rsi_type": ["rsi", "wilder", "rsx", "cutler"],
            "price": [
                "close",
                "open",
                "high",
                "low",
                "median",
                "typical",
                "weighted",
                "median_body",
                "average",
                "trend_biased",
                "volume",
            ],
            # "step_size_fast": [3.0, 5.0, 7.0],
            # "step_size_slow": [10.0, 15.0, 20.0, 25.0],
            "over_sold": [5.0, 10.0, 20.0],
            "over_bought": [80.0, 90.0, 95.0],
            "minmax_period": [21, 34, 49, 63],
        },
    },
    {
        "name": "Momentum Candles Modified w ATR",
        "function": MomentumCandlesModifiedWAtr,
        "signal_function": signal_momentum_candles_modified_w_atr,
        "raw_function": MomentumCandlesModifiedWAtr,
        "description": "ATR-normalized candle momentum with static thresholds at +/- (1/atr_multiplier). Returns Value, Threshold_Pos, Threshold_Neg.",
        "parameters": {"atr_period": 50, "atr_multiplier": 2.5},
        "parameter_space": {
            "atr_period": [14, 21, 50, 70],
            "atr_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0],
        },
    },
    {
        "name": "Momentum Candles w ATR",
        "function": MomentumCandlesWATR,
        "signal_function": signal_momentum_candles_w_atr,
        "raw_function": MomentumCandlesWATR,
        "description": "Colours candles based on momentum with an ATR filter: bullish if Close > Open and ATR(atr_period)/abs(Close-Open) < atr_multiplier; bearish if Close < Open under the same filter.",
        "parameters": {"atr_period": 50, "atr_multiplier": 2.5},
        "parameter_space": {
            "atr_period": [14, 21, 50, 70],
            "atr_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0],
        },
    },
    {
        "name": "precision-trend-histogram",
        "function": PrecisionTrendHistogram,
        "signal_function": signal_precision_trend_histogram,
        "raw_function": PrecisionTrendHistogram,
        "description": "A histogram that visualises trend strength and direction using an average range band and sequential trend state.",
        "parameters": {"avg_period": 30, "sensitivity": 3.0},
        "parameter_space": {
            "avg_period": [25, 30, 35, 45, 50],
            # "sensitivity": [1.0, 2.0, 3.0, 4.0],
        },
    },
    {
        "name": "QQEWithAlerts",
        "function": QQEWithAlerts,
        "signal_function": signal_qqe_with_alerts,
        "raw_function": QQEWithAlerts,
        "description": "QQE smooths RSI with an EMA and adds a dynamic trailing level to form a hybrid oscillator with trend-following characteristics.",
        "parameters": {"rsi_period": 14, "sf": 5},
        "parameter_space": {"rsi_period": [7, 14, 21, 30], "sf": [3, 5, 7, 9]},
    },
    {
        "name": "RWI BTF",
        "function": RWIBTF,
        "signal_function": signal_rwi_btf,
        "raw_function": RWIBTF,
        "description": "Random Walk Index (BTF-capable): measures deviation from a random walk by comparing price moves to ATR-scaled expected range.",
        "parameters": {"length": 2, "tf": None},
        "parameter_space": {
            "length": [2, 3, 4, 6, 8, 10, 14],
            # "tf": [None, "5T", "15T", "1H", "4H", "1D"],
        },
    },
    {
        "name": "SherifHilo",
        "function": SherifHilo,
        "signal_function": signal_sherif_hilo,
        "raw_function": SherifHilo,
        "description": "Rolling Hi-Lo regime indicator that switches Data to LLV or HHV based on close vs previous value; outputs LineUp/LineDown.",
        "parameters": {"period_high": 100, "period_lows": 100},
        "parameter_space": {
            "period_high": [20, 50, 100, 150, 200],
            "period_lows": [20, 50, 100, 150, 200],
        },
    },
    {
        "name": "TheTurtleTradingChannel",
        "function": TheTurtleTradingChannel,
        "signal_function": signal_theturtletradingchannel,
        "raw_function": TheTurtleTradingChannel,
        "description": "* Based on the Turtle Trading rules, this channel plots recent highs and lows (Donchian channels) to define breakout levels.",
        "parameters": {"trade_period": 20, "stop_period": 10, "strict": False},
        "parameter_space": {
            "trade_period": [20, 30, 40],
            "stop_period": [5, 10, 15],
            "strict": [False, True],
        },
    },
    {
        "name": "TII",
        "function": TII,
        "signal_function": signal_tii,
        "raw_function": TII,
        "description": "Trend Intensity Index (TII). Measures the strength of a trend relative to a moving average over a look-back window, ranging from 0 to 100.",
        "parameters": {"length": 30, "ma_length": 60, "ma_method": 0, "price": 0},
        "parameter_space": {
            "length": [14, 20, 30, 40, 50],
            "ma_length": [30, 60, 90, 120],
            "ma_method": [0, 1, 2, 3],
            "price": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "Top Bottom NR",
        "function": TopBottomNR,
        "signal_function": signal_top_bottom_nr,
        "raw_function": TopBottomNR,
        "description": "Computes run-length counts since the most recent breakout of lows/highs over the previous period, returning LongSignal and ShortSignal aligned to the input index.",
        "parameters": {"per": 14},
        "parameter_space": {"per": [7, 14, 21, 30]},
    },
    {
        "name": "TP",
        "function": TP,
        "signal_function": signal_tp,
        "raw_function": TP,
        "description": "Advance Trend Pressure (TP). Computes rolling sums of up/down contributions over 'length' bars; TP = Up - Dn. Optionally outputs Up and Dn lines when show_updn=True.",
        "parameters": {"length": 14, "show_updn": False},
        "parameter_space": {"length": [7, 14, 21, 30], "show_updn": [False, True]},
    },
    {
        "name": "TrendAkkam",
        "function": TRENDAKKAM,
        "signal_function": signal_trend_akkam,
        "raw_function": TRENDAKKAM,
        "description": "ATR-based trailing stop that flips with price and can use either ATR*factor or a fixed delta for stop distance.",
        "parameters": {
            "akk_range": 100,
            "ima_range": 1,
            "akk_factor": 6.0,
            "mode": 0,
            "delta_price": 30.0,
            "point": 1.0,
        },
        "parameter_space": {
            "akk_range": [14, 50, 100, 200],
            "ima_range": [1, 3, 5, 10],
            "akk_factor": [2.0, 3.0, 4.0, 6.0],
            "mode": [0, 1],
            "delta_price": [10.0, 20.0, 30.0, 50.0],
            "point": [0.01, 0.1, 1.0],
        },
    },
    {
        "name": "TrendDirectionForceIndexSmoothed4",
        "function": TrendDirectionForceIndexSmoothed4,
        "signal_function": signal_trend_direction_force_index_smoothed_4,
        "raw_function": TrendDirectionForceIndexSmoothed4,
        "description": "Trend Direction Force Index (Smoothed v4): compares a chosen moving average to its EMA-like smooth, normalizes and iSmooth-filters the result, and emits up/down triggers and a trend state.",
        "parameters": {
            "trend_period": 20,
            "trend_method": "ema",
            "price": "close",
            "trigger_up": 0.05,
            "trigger_down": -0.05,
            "smooth_length": 5.0,
            "smooth_phase": 0.0,
            "color_change_on_zero_cross": False,
            "point": 1.0,
        },
        "parameter_space": {
            "trend_period": [10, 20, 30, 50],
            "trend_method": [
                "sma",
                "ema",
                "dsema",
                "dema",
                "tema",
                "smma",
                "lwma",
                "pwma",
                "vwma",
                "hull",
                "tma",
                "sine",
                "mcg",
                "zlma",
                "lead",
                "ssm",
                "smoo",
                "linr",
                "ilinr",
                "ie2",
                "nlma",
            ],
            "price": [
                "close",
                "open",
                "high",
                "low",
                "median",
                "typical",
                "weighted",
                "average",
                "medianb",
                "tbiased",
                "haclose",
                "haopen",
                "hahigh",
                "halow",
                "hamedian",
                "hatypical",
                "haweighted",
                "haaverage",
                "hamedianb",
                "hatbiased",
            ],
            "trigger_up": [0.02, 0.05, 0.1],
            "trigger_down": [-0.02, -0.05, -0.1],
            "smooth_length": [3.0, 5.0, 8.0, 13.0],
            "smooth_phase": [-50.0, 0.0, 50.0],
            "color_change_on_zero_cross": [False, True],
            "point": [0.01, 0.1, 1.0],
        },
    },
    {
        "name": "trend-lord-nrp-indicator",
        "function": TrendLordNrpIndicator,
        "signal_function": signal_trend_lord_nrp_indicator,
        "raw_function": TrendLordNrpIndicator,
        "description": "Two-stage moving average (length and sqrt(length)) on selected price. Buy slots to Low/High (or MA) based on up/down; Sell is the second-stage MA.",
        "parameters": {
            "length": 12,
            "mode": "smma",
            "price": "close",
            "show_high_low": False,
        },
        "parameter_space": {
            "length": [8, 12, 16, 20, 24, 30],
            "mode": ["sma", "ema", "smma", "lwma"],
            "price": ["close", "open", "high", "low", "median", "typical", "weighted"],
            "show_high_low": [False, True],
        },
    },
    {
        "name": "TrendContinuation2",
        "function": Trendcontinuation2,
        "signal_function": signal_trendcontinuation2,
        "raw_function": Trendcontinuation2,
        "description": "* An indicator designed to confirm whether an existing trend is continuing or faltering by blending moving averages and momentum.",
        "parameters": {"n": 20, "t3_period": 5, "b": 0.618},
        "parameter_space": {
            "n": [10, 20, 30, 50],
            "t3_period": [3, 5, 8, 10],
            "b": [0.5, 0.618, 0.7, 0.8],
        },
    },
    {
        "name": "TTF",
        "function": TTF,
        "signal_function": signal_ttf,
        "raw_function": TTF,
        "description": "Trend Trigger Factor (TTF) from rolling highs/lows with T3 smoothing; returns TTF and a threshold Signal at top_line/bottom_line.",
        "parameters": {
            "ttf_bars": 8,
            "top_line": 75.0,
            "bottom_line": -75.0,
            "t3_period": 3,
            "b": 0.7,
        },
        "parameter_space": {
            "ttf_bars": [5, 8, 12, 20],
            "top_line": [60.0, 70.0, 75.0, 80.0],
            "bottom_line": [-80.0, -75.0, -70.0, -60.0],
            "t3_period": [1, 2, 3, 5],
            # "b": [0.5, 0.6, 0.7, 0.8],
        },
    },
    {
        "name": "VIDYA",
        "function": VIDYA,
        "signal_function": signal_vidya,
        "raw_function": VIDYA,
        "description": "VIDYA is an adaptive moving average that scales an EMA by a volatility ratio (short std divided by long std). Uses Close price only.",
        "parameters": {"period": 9, "histper": 30},
        "parameter_space": {"period": [5, 9, 14, 21], "histper": [20, 30, 50, 100]},
    },
    {
        "name": "Vortex_Indicator",
        "function": VortexIndicator,
        "signal_function": signal_vortex_indicator,
        "raw_function": VortexIndicator,
        "description": "* The Vortex Indicator plots two lines, VI+ and VI-, derived from up and down price movement. It measures the strength of positive and negative trends.",
        "parameters": {"length": 14},
        "parameter_space": {"length": [7, 14, 21, 30]},
    },
    {
        "name": "wpr + ma (alerts)",
        "function": WprMaAlerts,
        "signal_function": signal_wpr_ma_alerts,
        "raw_function": WprMaAlerts,
        "description": "Williams %R smoothed by a selectable moving average (sma/ema/smma/lwma), returning WPR, Signal, and Cross states.",
        "parameters": {"wpr_period": 35, "signal_period": 21, "ma_method": "smma"},
        "parameter_space": {
            "wpr_period": [14, 21, 35, 55],
            "signal_period": [5, 9, 13, 21, 34],
            "ma_method": ["sma", "ema", "smma", "lwma"],
        },
    },
    {
        "name": "XmaColouredUpdatedForNnfx",
        "function": XmaColouredUpdatedForNnfx,
        "signal_function": signal_xma_coloured_updated_for_nnfx,
        "raw_function": XmaColouredUpdatedForNnfx,
        "description": "Moving-average threshold indicator producing Signal, Fl, Up, and Dn series; uses two MAs with a point threshold on an applied price.",
        "parameters": {
            "period": 12,
            "porog": 3,
            "metod": "ema",
            "metod2": "ema",
            "price": "close",
            "tick_size": 0.0001,
        },
        "parameter_space": {
            "period": [8, 12, 20, 34, 50],
            "porog": [1, 2, 3, 5, 8],
            "metod": ["sma", "ema", "smma", "lwma"],
            "metod2": ["sma", "ema", "smma", "lwma"],
            "price": ["close", "open", "high", "low", "median", "typical", "weighted"],
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
    {
        "name": "CMO",
        "function": cmo_func,
        "signal_function": signal_cmo,
        "raw_function": ta.CMO,
        "description": "Tushar Chande's oscillator measures momentum by comparing the sum of up moves to the sum of down moves over a look-back period.",
        "parameters": {"timeperiod": 9},
        "parameter_space": {
            "timeperiod": [5, 9, 14, 20, 30],
            # "price": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"],
        },
    },
    {
        "name": "dpo-histogram-indicator",
        "function": DpoHistogramIndicator,
        "signal_function": signal_dpo_histogram_indicator,
        "raw_function": DpoHistogramIndicator,
        "description": "Detrended Price Oscillator with up/down histograms. Computes price minus a forward-shifted moving average; positive values map to DPO_Up and negatives to DPO_Dn.",
        "parameters": {"period": 14, "ma": "sma"},
        "parameter_space": {
            "period": [10, 14, 20, 30, 50],
            "ma": ["sma", "ema", "smma", "wma"],
        },
    },
    {
        "name": "ErgodicTVI",
        "function": ErgodicTVI,
        "signal_function": signal_ergodic_tvi,
        "raw_function": ErgodicTVI,
        "description": "* This oscillator combines price momentum and volatility information to produce an “ergodic” signal line with a smoothing component. It is similar to an MACD of the True Volume Indicator.",
        "parameters": {
            "Period1": 12,
            "Period2": 12,
            "Period3": 1,
            "EPeriod1": 5,
            "EPeriod2": 5,
            "EPeriod3": 5,
            "pip_size": 0.0001,
        },
        "parameter_space": {
            "Period1": [8, 12, 16, 20, 34],
            "Period2": [8, 12, 16, 20, 34],
            "Period3": [1, 2, 3, 5],
            "EPeriod1": [3, 5, 8, 13],
            "EPeriod2": [3, 5, 8, 13],
            "EPeriod3": [3, 5, 8, 13],
        },
    },
    {
        "name": "GlitchIndexFixed",
        "function": GlitchIndexFixed,
        "signal_function": signal_glitch_index_fixed,
        "raw_function": GlitchIndexFixed,
        "description": "* An obscure momentum indicator designed to detect unusual “glitches” or anomalies in price movement.  It acts as a smoothed oscillator.",
        "parameters": {
            "MaPeriod": 30,
            "MaMethod": "sma",
            "Price": "median",
            "level1": 1.0,
            "level2": 1.0,
        },
        "parameter_space": {
            "MaPeriod": [10, 20, 30, 50, 100],
            "MaMethod": [
                "sma",
                "ema",
                "smma",
                "lwma",
                "slwma",
                "dsema",
                "tema",
                "lsma",
                "nlma",
            ],
            "Price": ["close", "median", "typical", "weighted", "haaverage"],
            "level1": [0.5, 1.0, 1.5],
            "level2": [1.5, 2.0, 2.5],
        },
    },
    {
        "name": "IS Calculation",
        "function": ISCalculation,
        "signal_function": signal_is_calculation,
        "raw_function": ISCalculation,
        "description": "* A momentum indicator that applies additional smoothing to a traditional momentum calculation (price – price n periods ago).",
        "parameters": {"period": 10, "nbchandelier": 10, "lag": 0},
        "parameter_space": {
            "period": [5, 10, 14, 20, 30],
            "nbchandelier": [5, 10, 14, 20],
            "lag": [0, 1, 2, 3, 5],
        },
    },
    {
        "name": "JpOscillator",
        "function": JpOscillator,
        "signal_function": signal_jposcillator,
        "raw_function": JpOscillator,
        "description": "Forward-looking buffer 2*Close - 0.5*Close.shift(-1) - 0.5*Close.shift(-2) - Close.shift(-4), optionally smoothed with SMA/EMA/SMMA/LWMA; outputs Jp plus slope-segmented JpUp/JpDown.",
        "parameters": {"period": 5, "mode": 0, "smoothing": True},
        "parameter_space": {
            "period": [3, 5, 8, 13, 21],
            "mode": [0, 1, 2, 3],
            "smoothing": [True, False],
        },
    },
    {
        "name": "METROFixed",
        "function": METROFixed,
        "signal_function": signal_metro_fixed,
        "raw_function": METROFixed,
        "description": "Computes Wilder's RSI and StepRSI (fast/slow) via backward clamped recursion with adjustable step sizes. Outputs RSI, StepRSI_fast, and StepRSI_slow aligned to df.index.",
        "parameters": {"period_rsi": 14, "step_size_fast": 5.0, "step_size_slow": 15.0},
        "parameter_space": {
            "period_rsi": [7, 10, 14, 21, 28],
            "step_size_fast": [3.0, 5.0, 7.0, 10.0],
            "step_size_slow": [10.0, 15.0, 20.0, 30.0],
        },
    },
    {
        "name": "Price Momentum Oscillator",
        "function": PriceMomentumOscillator,
        "signal_function": signal_price_momentum_oscillator,
        "raw_function": PriceMomentumOscillator,
        "description": "A momentum oscillator that applies two-stage EMA smoothing to percentage price change (alpha=2/one then 2/two), scaled by 10; the signal is an EMA of the PMO. Oscillates around zero.",
        "parameters": {"one": 35, "two": 20, "period": 10},
        "parameter_space": {
            "one": [20, 25, 30, 35, 40, 45],
            "two": [10, 14, 20, 26, 30],
            "period": [5, 10, 14, 20],
        },
    },
    {
        "name": "SmoothedMomentum",
        "function": SmoothedMomentum,
        "signal_function": signal_smoothed_momentum,
        "raw_function": SmoothedMomentum,
        "description": "Percentage momentum (100 * price / price[n] ago) with optional smoothing via SMA/EMA/SMMA/LWMA on a chosen applied price; returns both smoothed (SM) and raw Momentum.",
        "parameters": {
            "momentum_length": 12,
            "use_smoothing": True,
            "smoothing_method": 0,
            "smoothing_length": 20,
            "price": 0,
        },
        "parameter_space": {
            "momentum_length": [5, 10, 12, 14, 20],
            "use_smoothing": [True, False],
            "smoothing_method": [0, 1, 2, 3],
            "smoothing_length": [5, 10, 20, 50, 100],
            "price": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "true-strength-index",
        "function": TrueStrengthIndex,
        "signal_function": signal_true_strength_index,
        "raw_function": TrueStrengthIndex,
        "description": "A momentum oscillator created by William Blau that uses double-smoothed EMAs of price changes. TSI oscillates around zero and ranges roughly between -100 and +100.",
        "parameters": {"first_r": 5, "second_s": 8},
        "parameter_space": {"first_r": [3, 5, 8, 13, 25], "second_s": [5, 8, 13, 21]},
    },
    {
        "name": "Zerolag_MACD.mq4",
        "function": ZerolagMACDMq4,
        "signal_function": signal_zerolag_macd_mq4,
        "raw_function": ZerolagMACDMq4,
        "description": "Zero-lag MACD using MT4-style EMA seeding; outputs ZL_MACD and ZL_Signal aligned to input index.",
        "parameters": {"fast": 12, "slow": 24, "signal": 9},
        "parameter_space": {
            "fast": [5, 8, 12, 15],
            "slow": [20, 24, 26, 30],
            "signal": [5, 9, 12],
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
    {
        "name": "BamsBung3",
        "function": BamsBung3,
        "signal_function": signal_bams_bung_3,
        "raw_function": BamsBung3,
        "description": "SMA-based Bollinger Band stop-and-signal indicator producing up/down trend stops, entry signals, and optional lines.",
        "parameters": {
            "length": 14,
            "deviation": 2.0,
            "money_risk": 0.02,
            "signal_mode": 1,
            "line_mode": 1,
        },
        "parameter_space": {
            "length": [10, 14, 20, 30],
            "deviation": [1.5, 2.0, 2.5, 3.0],
            "money_risk": [0.01, 0.02, 0.05, 0.1],
            "signal_mode": [0, 1, 2],
            "line_mode": [0, 1],
        },
    },
    {
        "name": "CVI_Multi",
        "function": CVIMulti,
        "signal_function": signal_cvi_multi,
        "raw_function": CVIMulti,
        "description": "Chartmill Value Indicator (CVI): normalized deviation of Close from a moving average of Median Price ((High+Low)/2). method selects the MA (0=SMA, 1=EMA, 2=SMMA/RMA, 3=LWMA). Denominator is ATR; if use_modified is True, ATR is scaled by sqrt(length).",
        "parameters": {"length": 14, "method": 0, "use_modified": False},
        "parameter_space": {
            "length": [7, 14, 21, 30],
            "method": [0, 1, 2, 3],
            "use_modified": [False, True],
        },
    },
    {
        "name": "Silence",
        "function": Silence,
        "signal_function": signal_silence,
        "raw_function": Silence,
        "description": "Silence indicator producing normalized (0..100, reversed) Aggressiveness and Volatility series over rolling windows; higher values indicate quieter conditions.",
        "parameters": {
            "my_period": 12,
            "buff_size": 96,
            "point": 0.0001,
            "redraw": True,
        },
        "parameter_space": {
            "my_period": [8, 12, 16, 24],
            "buff_size": [64, 96, 128, 192],
        },
    },
    {
        "name": "Stiffness Indicator",
        "function": StiffnessIndicator,
        "signal_function": signal_stiffness_indicator,
        "raw_function": StiffnessIndicator,
        "description": "Measures price stiffness by comparing Close to a thresholded MA and summing signals; outputs Stiffness and Signal.",
        "parameters": {
            "period1": 100,
            "method1": "sma",
            "period3": 60,
            "period2": 3,
            "method2": "sma",
        },
        "parameter_space": {
            "period1": [50, 100, 150, 200],
            "method1": ["sma", "ema", "smma", "lwma"],
            "period3": [30, 60, 90, 120],
            "period2": [2, 3, 5, 8],
            "method2": ["sma", "ema", "smma", "lwma"],
        },
    },
    {
        "name": "TTMS",
        "function": TTMS,
        "signal_function": signal_ttms,
        "raw_function": TTMS,
        "description": "A volatility indicator similar to the TTM Squeeze. It measures the relationship between Bollinger Bands and Keltner Channels to identify low-volatility squeeze conditions.",
        "parameters": {
            "bb_length": 20,
            "bb_deviation": 2.0,
            "keltner_length": 20,
            "keltner_smooth_length": 20,
            "keltner_smooth_method": 0,
            "keltner_deviation": 2.0,
        },
        "parameter_space": {
            "bb_length": [10, 14, 20, 30, 50],
            "bb_deviation": [1.0, 1.5, 2.0, 2.5, 3.0],
            "keltner_length": [10, 14, 20, 30, 50],
            "keltner_smooth_length": [10, 14, 20, 30, 50],
            "keltner_smooth_method": [0, 1, 2, 3],
            "keltner_deviation": [1.0, 1.5, 2.0, 2.5],
        },
    },
    {
        "name": "Volatility Ratio",
        "function": VolatilityRatio,
        "signal_function": signal_volatility_ratio,
        "raw_function": VolatilityRatio,
        "description": "Volatility Ratio (VR) = rolling std of price divided by its SMA over the same period. Outputs VR plus two alternating segments where VR < 1 (VR_below1_a and VR_below1_b).",
        "parameters": {"period": 25, "price": "Close"},
        "parameter_space": {
            "period": [10, 14, 20, 25, 30, 50],
            "price": ["Close", "HL2", "HLC3", "OHLC4", "Weighted"],
        },
    },
    {
        "name": "William Vix-Fix",
        "function": WilliamVixFix,
        "signal_function": signal_william_vix_fix,
        "raw_function": WilliamVixFix,
        "description": "* A volatility indicator created by Larry Williams that emulates the CBOE VIX using price data. It identifies spikes in fear or complacency.",
        "parameters": {"period": 22},
        "parameter_space": {"period": [7, 14, 20, 22, 30]},
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
    {
        "name": "ATRBasedEMAVariant1",
        "function": ATRBasedEMAVariant1,
        "signal_function": signal_atr_based_ema_variant_1,
        "raw_function": ATRBasedEMAVariant1,
        "description": "EMA on Close with a dynamic period driven by an EMA(14) of High/Low (ATR proxy). Higher volatility increases the equivalent EMA period, making the baseline slower. Outputs columns ['EMA_ATR_var1', 'EMA_Equivalent'].",
        "parameters": {"ema_fastest": 14.0, "multiplier": 300.0},
        "parameter_space": {
            "ema_fastest": [7.0, 10.0, 14.0, 21.0, 28.0],
            "multiplier": [100.0, 200.0, 300.0, 400.0, 500.0],
        },
    },
    {
        "name": "EhlersTwoPoleSuperSmootherFilter",
        "function": EhlersTwoPoleSuperSmootherFilter,
        "signal_function": signal_ehlers_two_pole_super_smoother_filter,
        "raw_function": EhlersTwoPoleSuperSmootherFilter,
        "description": "John Ehlers' Super Smoother is a two-pole low-pass filter designed to remove noise with minimal lag. Uses Open price, restarts on NaN segments, and preserves length.",
        "parameters": {"cutoff_period": 15},
        "parameter_space": {"cutoff_period": [10, 15, 20, 30, 40]},
    },
    {
        "name": "GChannel",
        "function": Gchannel,
        "signal_function": signal_gchannel,
        "raw_function": Gchannel,
        "description": "GChannel plots a recursive dynamic price channel (Upper/Middle/Lower). The midline can serve as a baseline; the channel adapts to volatility.",
        "parameters": {"length": 100, "price": "close"},
        "parameter_space": {
            "length": [20, 50, 100, 150, 200],
            "price": ["close", "open", "high", "low", "median", "typical", "weighted"],
        },
    },
    {
        "name": "Gd",
        "function": Gd,
        "signal_function": signal_gd,
        "raw_function": Gd,
        "description": "Generalized DEMA-style baseline: GD = (1+vf)*EMA - vf*EMA_of_EMA. Returns ['GD','EMA'] aligned to df.index; supports applied price selection.",
        "parameters": {"length": 20, "vf": 0.7, "price": 0},
        "parameter_space": {
            "length": [10, 20, 30, 50],
            "vf": [0.3, 0.5, 0.7, 1.0],
            "price": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "McGinley Dynamic 2.3",
        "function": McginleyDynamic23,
        "signal_function": signal_mcginley_dynamic_2_3,
        "raw_function": McginleyDynamic23,
        "description": "Created by J. R. McGinley, this indicator adjusts its smoothing factor dynamically based on the speed of price changes. Unlike a fixed-period moving average, the McGinley Dynamic speeds up in fast markets and slows down in slow markets.",
        "parameters": {
            "period": 12,
            "price": "close",
            "constant": 5.0,
            "method": "ema",
        },
        "parameter_space": {
            "period": [5, 8, 12, 14, 20, 24, 30],
            "price": ["close", "open", "high", "low", "median", "typical", "weighted"],
            "constant": [3.0, 5.0, 8.0, 10.0],
            "method": ["sma", "ema", "smma", "lwma", "gen"],
        },
    },
    {
        "name": "RangeFilterModified",
        "function": RangeFilterModified,
        "signal_function": signal_range_filter_modified,
        "raw_function": RangeFilterModified,
        "description": "Dynamic filter using Wilder ATR(atr_period) * multiplier to form adaptive bands around a recursive center line.",
        "parameters": {"atr_period": 14, "multiplier": 3.0},
        "parameter_space": {
            "atr_period": [7, 10, 14, 21, 28],
            "multiplier": [1.5, 2.0, 2.5, 3.0, 3.5],
        },
    },
    {
        "name": "SineWMA",
        "function": Sinewma,
        "signal_function": signal_sinewma,
        "raw_function": Sinewma,
        "description": "A moving average that applies half-sine weights across the lookback window. The highest weights are centered, providing a smooth curve that remains responsive to recent price while filtering noise.",
        "parameters": {"length": 20, "price": 0},
        "parameter_space": {
            "length": [10, 14, 20, 30, 50],
            "price": [0, 1, 2, 3, 4, 5, 6],
        },
    },
    {
        "name": "SmoothStep",
        "function": Smoothstep,
        "signal_function": signal_smoothstep,
        "raw_function": Smoothstep,
        "description": "A smoothing filter that applies the mathematical smoothstep function to price data, creating a very smooth baseline.",
        "parameters": {"period": 32, "price": "close"},
        "parameter_space": {
            "period": [8, 16, 32, 64],
            "price": [
                "close",
                "open",
                "high",
                "low",
                "median",
                "typical",
                "weighted",
                "lowhigh",
            ],
        },
    },
    {
        "name": "T3MA",
        "function": T3MA,
        "signal_function": signal_t3_ma,
        "raw_function": T3MA,
        "description": "The T3 moving average applies a triple smoothing process using exponential moving averages, offering a smoother curve with less lag than simple or double EMAs.",
        "parameters": {"length": 10, "b": 0.88, "price": 0},
        "parameter_space": {
            "length": [5, 8, 10, 14, 20, 30],
            "b": [0.7, 0.8, 0.88, 0.9, 0.95],
        },
    },
    {
        "name": "TetherLine",
        "function": TetherLine,
        "signal_function": signal_tether_line,
        "raw_function": TetherLine,
        "description": "Baseline from the midpoint of rolling Highest High and Lowest Low over length. Outputs AboveCenter when Close > midpoint and BelowCenter when Close < midpoint; ArrowUp/ArrowDown are NaN placeholders.",
        "parameters": {"length": 55},
        "parameter_space": {"length": [21, 34, 55, 89]},
    },
    {
        "name": "Trimagen",
        "function": Trimagen,
        "signal_function": signal_trimagen,
        "raw_function": Trimagen,
        "description": "Triangular Moving Average (TriMA): an SMA of an SMA, center-weighted smoothing that is smoother than a standard SMA.",
        "parameters": {"period": 20, "applied_price": "close"},
        "parameter_space": {
            "period": [10, 14, 20, 30, 50],
            "applied_price": [
                "close",
                "open",
                "high",
                "low",
                "median",
                "typical",
                "weighted",
            ],
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
