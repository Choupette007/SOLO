import pandas as pd

from solana_trading_bot_bundle.trading_bot.volume_utils import is_volume_spike
from solana_trading_bot_bundle.trading_bot.candlestick_patterns import (
    is_evening_star,
    is_morning_star,
    is_three_black_crows,
    is_three_white_soldiers,
    is_hammer,
    is_inverted_hammer,
    is_bullish_engulfing,
    is_bearish_engulfing,
    is_doji,
)


def _mk_candle(o, h, l, c, v=0):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def test_volume_spike_mad():
    # Create a stable baseline and one spike at the end
    vols = pd.Series([100, 120, 110, 90, 95, 105, 98, 102, 100, 115, 5000])
    res = is_volume_spike(vols, idx=-1, window=8, k=3, factor=2)
    assert res["spike"] is True
    assert res["reason"] in ("mad", "factor")


def test_evening_star_detected():
    df = pd.DataFrame(
        [
            _mk_candle(10, 15, 9, 14),  # bullish with decent body
            _mk_candle(15, 16, 14, 15.2),  # small indecision
            _mk_candle(15.3, 15.3, 9, 11),  # bearish closing deep into first body
        ]
    )
    assert is_evening_star(df) is True


def test_morning_star_detected():
    df = pd.DataFrame(
        [
            _mk_candle(14, 15, 13, 12.5),  # bearish
            _mk_candle(12.4, 12.8, 12, 12.6),  # small indecision
            _mk_candle(12.6, 16, 12.6, 15),  # bullish, closes above midpoint of first
        ]
    )
    assert is_morning_star(df) is True


def test_three_white_soldiers_and_black_crows():
    df_bull = pd.DataFrame(
        [
            _mk_candle(10, 11, 9.9, 10.8),
            _mk_candle(10.9, 12, 10.6, 11.7),
            _mk_candle(11.6, 13, 11.5, 12.9),
        ]
    )
    assert is_three_white_soldiers(df_bull) is True

    df_bear = pd.DataFrame(
        [
            _mk_candle(12, 12.5, 11.5, 11.7),
            _mk_candle(11.6, 11.8, 11, 10.9),
            _mk_candle(10.8, 11, 10.3, 10.5),
        ]
    )
    assert is_three_black_crows(df_bear) is True


def test_hammers_and_engulfing_and_doji():
    # hammer (long lower tail)
    df_hammer = pd.DataFrame([_mk_candle(10, 10.5, 9.0, 10.2)])
    assert is_hammer(df_hammer) is True

    # inverted hammer (long upper tail)
    df_inverted = pd.DataFrame([_mk_candle(10, 12.5, 9.8, 10.1)])
    assert is_inverted_hammer(df_inverted) is True

    # bullish engulfing
    df_engulf = pd.DataFrame([_mk_candle(10, 10.5, 9.5, 9.8), _mk_candle(9.7, 11, 9.6, 10.9)])
    assert is_bullish_engulfing(df_engulf) is True

    # bearish engulfing
    df_engulf_b = pd.DataFrame([_mk_candle(9.8, 10.3, 9.4, 10.2), _mk_candle(10.3, 10.4, 9.0, 9.2)])
    assert is_bearish_engulfing(df_engulf_b) is True

    # doji
    df_doji = pd.DataFrame([_mk_candle(10.0, 10.4, 9.6, 10.02)])
    assert is_doji(df_doji) is True