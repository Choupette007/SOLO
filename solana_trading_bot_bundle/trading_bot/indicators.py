# solana_trading_bot_bundle/trading_bot/indicators.py
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

def _to_np(a) -> np.ndarray:
    return np.asarray(a, dtype=float)

def rsi(close, period: int = 14) -> np.ndarray:
    c = _to_np(close)
    if c.size < period + 1: return np.full_like(c, np.nan)
    diff = np.diff(c, prepend=c[0])
    gains = np.clip(diff, 0, None)
    losses = -np.clip(diff, None, 0)
    # Wilder's smoothing
    rsis = np.full_like(c, np.nan)
    avg_gain = np.convolve(gains, np.ones(period), 'valid')[:1].mean()
    avg_loss = np.convolve(losses, np.ones(period), 'valid')[:1].mean()
    # initialize at index = period
    if avg_loss == 0:
        rsis[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsis[period] = 100.0 - (100.0 / (1.0 + rs))
    ag, al = avg_gain, avg_loss
    for i in range(period + 1, c.size):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            rsis[i] = 100.0
        else:
            rs = ag / al
            rsis[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsis

def ema(arr, period: int) -> np.ndarray:
    x = _to_np(arr)
    if x.size < period: 
        return np.full_like(x, np.nan)
    k = 2.0 / (period + 1)
    out = np.empty_like(x); out[:] = np.nan
    # seed with SMA
    sma = np.nanmean(x[:period])
    out[period-1] = sma
    for i in range(period, x.size):
        out[i] = x[i] * k + out[i-1] * (1 - k)
    return out

def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    c = _to_np(close)
    ema_fast = ema(c, fast)
    ema_slow = ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}

def bollinger(close, period: int = 20, stddev: float = 2.0) -> Dict[str, np.ndarray]:
    c = _to_np(close)
    n = c.size
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    width = np.full(n, np.nan)
    percent_b = np.full(n, np.nan)
    if n < period: 
        return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}
    # rolling mean/std (simple, O(n*period); fine for 100–500 bars)
    for i in range(period-1, n):
        w = c[i-period+1:i+1]
        m = w.mean(); s = w.std(ddof=0)
        mid[i] = m
        upper[i] = m + stddev * s
        lower[i] = m - stddev * s
        width[i] = (upper[i] - lower[i]) / m if m != 0 and not np.isnan(m) else np.nan
        if not np.isnan(upper[i]) and not np.isnan(lower[i]) and upper[i] != lower[i]:
            percent_b[i] = (c[i] - lower[i]) / (upper[i] - lower[i])
    return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}

def td9(close, lookback: int = 30) -> Dict[str, np.ndarray]:
    """
    Basic TD Sequential setup counts:
      - Bullish setup: close[n] < close[n-4] → count down (1..9)
      - Bearish setup: close[n] > close[n-4] → count up   (1..9)
    We expose the running counts and booleans at 9.
    """
    c = _to_np(close)
    n = c.size
    up = np.zeros(n, dtype=int)     # bearish setup count
    down = np.zeros(n, dtype=int)   # bullish setup count
    setup = np.full(n, "", dtype=object)
    td9_up = np.zeros(n, dtype=bool)
    td9_down = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < 4:
            continue
        if c[i] > c[i-4]:
            up[i] = (up[i-1] + 1) if up[i-1] > 0 else 1
            down[i] = 0
        elif c[i] < c[i-4]:
            down[i] = (down[i-1] + 1) if down[i-1] > 0 else 1
            up[i] = 0
        else:
            up[i] = 0; down[i] = 0
        if up[i] == 9:
            setup[i] = "td9_up"; td9_up[i] = True
        elif down[i] == 9:
            setup[i] = "td9_down"; td9_down[i] = True
    # optional: only keep last `lookback` active
    if lookback and n > lookback:
        start = n - lookback
        td9_up[:start] = False
        td9_down[:start] = False
    return {"up": up, "down": down, "setup": setup, "td9_up": td9_up, "td9_down": td9_down}
