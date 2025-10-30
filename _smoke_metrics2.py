import importlib, sys
sys.path.insert(0, ".")
trn = importlib.import_module("solana_trading_bot_bundle.trading_bot.tradingNew")
me  = importlib.import_module("solana_trading_bot_bundle.trading_bot.metrics_engine")

store = me.MetricsStore(prorate_fees=False)
trn._metrics_store = store

trn._metrics_on_fill(
  token_addr="So1111...", symbol="SOL", side="BUY",
  qty=0.5, price_usd=150.0, fee_usd=0.12, txid="DEMO", simulated=True, source="smoke"
)
trn._metrics_snapshot_equity_point()

print("Trades:", len(store.trades()))
print("Equity points:", len(store.equity_curve()))
print("Summary keys:", list(store.summary().keys())[:6])
