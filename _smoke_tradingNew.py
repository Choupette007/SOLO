import importlib, sys
sys.path.insert(0, ".")
mod = importlib.import_module("solana_trading_bot_bundle.trading_bot.tradingNew")
print("Imported tradingNew OK. Functions:", [n for n in dir(mod) if n.startswith("_metrics_") or n.startswith("attach_")][:8])
