from solana_trading_bot_bundle.trading_bot.utils import load_config
import json
cfg = load_config()
print("=== rugcheck ===")
print(json.dumps(cfg.get("rugcheck", {}), indent=2))
print("=== discovery ===")
print(json.dumps(cfg.get("discovery", {}), indent=2))
print("=== holders ===")
print(json.dumps(cfg.get("holders", {}), indent=2))
