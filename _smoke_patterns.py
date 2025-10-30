import importlib, sys
sys.path.insert(0, ".")
mod = importlib.import_module("solana_trading_bot_bundle.trading_bot.tradingNew")

tok = {
    "_ohlc_open":  [1.00, 1.01, 1.02, 1.03, 1.02],
    "_ohlc_high":  [1.01, 1.02, 1.03, 1.04, 1.03],
    "_ohlc_low":   [0.99, 1.00, 1.01, 1.02, 1.01],
    "_ohlc_close": [1.00, 1.01, 1.015, 1.025, 1.02],
}
mod.attach_patterns_if_available(tok)
print({k: v for k, v in tok.items() if k.startswith("pat_") or k.startswith("bb_")})
