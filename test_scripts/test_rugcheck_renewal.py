import os, time, json, inspect, asyncio
from importlib import import_module

# Optional: pyjwt is typically "jwt"
try:
    import jwt
except Exception as e:
    print("pyjwt not installed? pip install pyjwt")
    raise

import aiohttp

def mod_info(name):
    try:
        m = import_module(name)
        path = getattr(m, "__file__", "<no __file__>")
        print(f"{name} -> {path}")
        return m
    except Exception as e:
        print(f"IMPORT FAILED for {name}: {e}")
        return None

def summarize_token(token: str) -> dict:
    out = {"present": bool(token), "prefix": None, "exp": None, "exp_human": None}
    if not token:
        return out
    out["prefix"] = token[:16] + "..." if len(token) > 16 else token
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        exp = decoded.get("exp")
        if exp:
            out["exp"] = int(exp)
            out["exp_human"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp))
    except Exception as e:
        out["exp_human"] = f"<decode failed: {e}>"
    return out

async def call_ensure(func):
    """Call ensure_valid_rugcheck_headers with best-effort signature handling."""
    try:
        sig = inspect.signature(func)
    except Exception:
        sig = None

    async with aiohttp.ClientSession() as sess:
        try:
            # Try common signatures:
            # ensure(session, force_refresh=True)
            if sig and "session" in sig.parameters and "force_refresh" in sig.parameters:
                if inspect.iscoroutinefunction(func):
                    return await func(sess, force_refresh=True)
                else:
                    return func(sess, force_refresh=True)

            # ensure(session)
            if sig and "session" in sig.parameters:
                if inspect.iscoroutinefunction(func):
                    return await func(sess)
                else:
                    return func(sess)

            # ensure(force_refresh=True)
            if sig and "force_refresh" in sig.parameters:
                if inspect.iscoroutinefunction(func):
                    return await func(force_refresh=True)
                else:
                    return func(force_refresh=True)

            # ensure()
            if inspect.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except Exception as e:
            print(f"  ensure_valid_rugcheck_headers call failed: {e}")
            return {}

async def run():
    print(f"PYTHONPATH head = {os.getcwd()}")
    print()

    util = mod_info("utils.rugcheck_auth")
    pkg  = mod_info("solana_trading_bot_bundle.trading_bot.rugcheck_auth")
    print()

    async def probe(mod, label):
        if not mod:
            return
        get_hdrs = getattr(mod, "get_rugcheck_headers", None)
        ensure_fn = getattr(mod, "ensure_valid_rugcheck_headers", None)

        print(f"== {label} ==")
        if get_hdrs:
            h0 = get_hdrs(force_refresh=False) if "force_refresh" in inspect.signature(get_hdrs).parameters else get_hdrs()
            tok0 = (h0.get("Authorization","").split(" ",1)[1] if "Authorization" in h0 else "")
            s0 = summarize_token(tok0)
            print(f"  headers(before): present={s0['present']} token={s0['prefix']} exp={s0['exp_human']}")

        if ensure_fn:
            h1 = await call_ensure(ensure_fn) or {}
            tok1 = (h1.get("Authorization","").split(" ",1)[1] if "Authorization" in h1 else "")
            s1 = summarize_token(tok1)
            print(f"  headers(after ensure): present={s1['present']} token={s1['prefix']} exp={s1['exp_human']}")
        else:
            print("  ensure_valid_rugcheck_headers not found")

        print()

    await probe(util, "utils.rugcheck_auth")
    await probe(pkg,  "pkg.rugcheck_auth")

if __name__ == "__main__":
    asyncio.run(run())
