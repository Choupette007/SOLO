import importlib, sys

print("sys.path[0] =", sys.path[0])

def chk(name):
    try:
        m = importlib.import_module(name)
        print(f"\n{name} ->", getattr(m, "__file__", "?"))
        for fn in ("ensure_valid_rugcheck_headers", "get_rugcheck_headers"):
            print(f"  has {fn}:", hasattr(m, fn))
    except Exception as e:
        print(f"\n{name} IMPORT FAILED:", e)

chk("solana_trading_bot_bundle.trading_bot.rugcheck_auth")
chk("utils.rugcheck_auth")
