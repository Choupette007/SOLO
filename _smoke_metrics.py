import importlib, sys, time
sys.path.insert(0, ".")
trn = importlib.import_module("solana_trading_bot_bundle.trading_bot.tradingNew")
me  = importlib.import_module("metrics_engine")  # must be importable on PYTHONPATH

store = me.MetricsStore(prorate_fees=False)
trn._metrics_store = store  # activate wrapper

trn._metrics_on_fill(
    token_addr="So1111...", symbol="SOL", side="BUY",
    qty=0.5, price_usd=150.0, fee_usd=0.12, txid="DEMO", simulated=True, source="smoke"
)
trn._metrics_snapshot_equity_point()

print("Trades:", len(getattr(store, "trades", lambda: [])()))
print("Equity points:", len(getattr(store, "equity_curve", lambda: [])()))
print("Summary keys:", list(getattr(store, "summary", lambda: {})().keys())[:6])
