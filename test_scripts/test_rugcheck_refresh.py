import os, json, time, asyncio, inspect, base64, sys
from pathlib import Path

import aiohttp

# Try both possible locations
headers_provider = None
provider_name = None
try:
    from solana_trading_bot_bundle.trading_bot.rugcheck_auth import ensure_valid_rugcheck_headers as headers_provider  # async preferred
    provider_name = "ensure_valid_rugcheck_headers (trading_bot)"
except Exception:
    try:
        from utils.rugcheck_auth import ensure_valid_rugcheck_headers as headers_provider  # async preferred
        provider_name = "ensure_valid_rugcheck_headers (utils)"
    except Exception:
        try:
            from utils.rugcheck_auth import get_rugcheck_headers as headers_provider  # sync fallback
            provider_name = "get_rugcheck_headers (utils)"
        except Exception:
            print("❌ Could not import any Rugcheck header function.")
            sys.exit(1)

# STATUS_FILE used by your GUI
try:
    from solana_trading_bot_bundle.trading_bot.fetching import STATUS_FILE
except Exception:
    STATUS_FILE = Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".local" / "share"))) / "SOLOTradingBot" / "rugcheck_status.json"

USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

def mask(tok: str) -> str:
    if not tok:
        return "<empty>"
    tok = tok.replace("Bearer ","")
    return tok[:8] + "..." + tok[-8:] if len(tok) > 20 else "<short token>"

def jwt_exp(ts: str) -> str:
    """Decode JWT exp without verifying signature, return human-ish age info."""
    try:
        parts = ts.split(".")
        if len(parts) < 2:
            return "n/a"
        # pad base64
        def fix(b):
            return b + "=" * (-len(b) % 4)
        payload = json.loads(base64.urlsafe_b64decode(fix(parts[1])).decode("utf-8"))
        exp = int(payload.get("exp", 0) or 0)
        if not exp:
            return "n/a"
        now = int(time.time())
        delta = exp - now
        return f"exp={exp} (in {delta}s)" if delta >= 0 else f"exp={exp} (expired {-delta}s ago)"
    except Exception:
        return "n/a"

async def get_headers(session, force=False):
    """Call whichever header func we found, with or without force refresh if supported."""
    sig = None
    try:
        sig = inspect.signature(headers_provider)
    except Exception:
        pass

    # async ensure_valid_rugcheck_headers(session, force_refresh=...)
    try:
        if inspect.iscoroutinefunction(headers_provider):
            if sig and "force_refresh" in sig.parameters:
                return await headers_provider(session, force_refresh=force)
            else:
                return await headers_provider(session)
    except TypeError:
        pass
    except Exception as e:
        print("header provider raised:", e)

    # sync get_rugcheck_headers()
    try:
        return headers_provider()
    except Exception as e:
        print("sync header provider raised:", e)
        return {}

async def test_once(force_refresh=False):
    async with aiohttp.ClientSession() as session:
        headers = await get_headers(session, force=force_refresh)
        auth = headers.get("Authorization", "")
        print(f"[{provider_name}] force_refresh={force_refresh} -> has Authorization? {bool(auth)}")
        if auth:
            print("  token(masked):", mask(auth))
            print("  token(exp):   ", jwt_exp(auth.replace('Bearer ','').strip()))

        # Probe Rugcheck with USDC
        url = f"https://api.rugcheck.xyz/v1/tokens/{USDC}/report"
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                status = resp.status
                print(f"  Rugcheck probe HTTP status: {status}")
                if status == 200:
                    # don't dump the full JSON; just prove it works
                    print("  ✅ Rugcheck query OK (token usable)")
                elif status in (401,403):
                    print("  ❌ Unauthorized/Forbidden (token likely invalid/expired)")
                else:
                    print("  ℹ️  Unexpected status; body excerpt follows:")
                    try:
                        txt = await resp.text()
                        print("    body:", txt[:160], "..." if len(txt) > 160 else "")
                    except Exception:
                        pass
        except Exception as e:
            print("  request error:", e)

        # Show STATUS_FILE (GUI banner source)
        try:
            print("STATUS_FILE ->", STATUS_FILE)
            if Path(STATUS_FILE).exists():
                print("STATUS_JSON ->")
                print(Path(STATUS_FILE).read_text(encoding="utf-8"))
            else:
                print("STATUS_JSON -> <missing>")
        except Exception as e:
            print("Could not read STATUS_FILE:", e)

async def main():
    print("== Pass 1: current headers ==")
    await test_once(force_refresh=False)
    print("\n== Pass 2: force refresh ==")
    await test_once(force_refresh=True)

if __name__ == "__main__":
    asyncio.run(main())
