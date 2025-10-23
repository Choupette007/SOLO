# solana_trading_bot_bundle/trading_bot/patterns.py
from __future__ import annotations
import numpy as np
from typing import Dict, List

def _to_np(a): return np.asarray(a, dtype=float)

def bullish_engulfing(open_, close) -> np.ndarray:
    o, c = _to_np(open_), _to_np(close)
    prev_bear = c[:-1] < o[:-1]
    curr_bull = c[1:]  > o[1:]
    engulf    = (c[1:] >= o[:-1]) & (o[1:] <= c[:-1])
    out = np.zeros_like(c, dtype=bool)
    out[1:] = prev_bear & curr_bull & engulf
    return out

def bearish_engulfing(open_, close) -> np.ndarray:
    o, c = _to_np(open_), _to_np(close)
    prev_bull = c[:-1] > o[:-1]
    curr_bear = c[1:]  < o[1:]
    engulf    = (o[1:] >= c[:-1]) & (c[1:] <= o[:-1])
    out = np.zeros_like(c, dtype=bool)
    out[1:] = prev_bull & curr_bear & engulf
    return out

def hammer(open_, high, low, close, tol: float = 0.3) -> np.ndarray:
    o, h, l, c = map(_to_np, (open_, high, low, close))
    body = np.abs(c - o)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    # lower shadow >= 2 * body, upper <= body, close near high
    cond = (lower >= 2*body) & (upper <= body) & ((h - c) <= tol * (h - l))
    return cond

def shooting_star(open_, high, low, close, tol: float = 0.3) -> np.ndarray:
    o, h, l, c = map(_to_np, (open_, high, low, close))
    body = np.abs(c - o)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    cond = (upper >= 2*body) & (lower <= body) & ((c - l) <= tol * (h - l))
    return cond

def doji(open_, close, eps: float = 1e-6, rel: float = 0.1) -> np.ndarray:
    o, c = _to_np(open_), _to_np(close)
    rng = np.maximum(np.abs(c - o), eps)
    return (np.abs(c - o) / rng) < rel

def classify_patterns(ohlcv: Dict[str, List[float]], names: List[str]) -> Dict[str, np.ndarray]:
    o, h, l, c = map(_to_np, (ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]))
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name == "bullish_engulfing": out[name] = bullish_engulfing(o, c)
        elif name == "bearish_engulfing": out[name] = bearish_engulfing(o, c)
        elif name == "hammer": out[name] = hammer(o, h, l, c)
        elif name == "shooting_star": out[name] = shooting_star(o, h, l, c)
        elif name == "doji": out[name] = doji(o, c)
    return out
