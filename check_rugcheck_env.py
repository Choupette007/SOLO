import os, base64, json, time
from pathlib import Path
try:
    from dotenv import dotenv_values
except Exception:
    print("python-dotenv not installed. In your venv: pip install python-dotenv"); raise

canon = Path(r"C:\Users\Admin\AppData\Local\SOLOTradingBot\.env")
cfg = dotenv_values(str(canon))

tok = cfg.get("RUGCHECK_API_KEY") or cfg.get("RUGCHECK_JWT") or cfg.get("RUGCHECK_JWT_TOKEN")

def decode_jwt_exp(jwt):
    try:
        parts = jwt.split(".")
        if len(parts) != 3:
            return None
        pad = "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + pad).decode())
        return payload.get("exp")
    except Exception:
        return None

print("Loaded from:", canon)
print("Has RUGCHECK_API_KEY?  ", bool(cfg.get("RUGCHECK_API_KEY")))
print("Has RUGCHECK_JWT?      ", bool(cfg.get("RUGCHECK_JWT")))
print("Has RUGCHECK_JWT_TOKEN?", bool(cfg.get("RUGCHECK_JWT_TOKEN")))
if tok and "." in tok:
    exp = decode_jwt_exp(tok)
    now = int(time.time())
    print("Looks like a JWT. exp:", exp, " now:", now, " delta_s:", (exp-now) if exp else None)
elif tok:
    print("Found a non-JWT token (likely API key).")
else:
    print("No Rugcheck token found in this .env file.")
