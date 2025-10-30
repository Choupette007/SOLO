# solana_trading_bot_bundle/trading_bot/rugcheck_auth.py
# Auto-verify Rugcheck JWT and auto-refresh via an external login script.
# On success, persist JWT to .env and to appdata fallback.

# rugcheck_auth.py — shared helper used by fetching.validate_rugcheck()
from __future__ import annotations

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import aiohttp

# For signing the login message
from solders.keypair import Keypair
from solders.pubkey import Pubkey

logger = logging.getLogger("TradingBot")

# --------------------------
# .env resolution & I/O
# --------------------------
def _appdata_base() -> Path:
    try:
        from solana_trading_bot_bundle.common.constants import appdata_dir as _appdata_dir  # type: ignore
        base = _appdata_dir() if callable(_appdata_dir) else Path(_appdata_dir)
    except Exception:
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "SOLOTradingBot"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _env_file_path() -> Path:
    try:
        from solana_trading_bot_bundle.common.constants import env_path as _env_path  # type: ignore
        envp = _env_path() if callable(_env_path) else Path(_env_path)
    except Exception:
        envp = _appdata_base() / ".env"
    return Path(envp)

def _load_env_map(envp: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if envp.exists():
        for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.startswith('"') and v.endswith('"') and len(v) >= 2:
                v = v[1:-1]
            out[k] = v
    return out

def _write_env_file(envp: Path, kv: Dict[str, str]) -> None:
    lines = ["# SOLOTradingBot .env file"]
    for k, v in kv.items():
        if v is None:
            continue
    # keep original order stable-ish
    for k, v in kv.items():
        s = str(v).replace("\r", "").replace("\n", "").replace('"', r'\"')
        lines.append(f'{k}="{s}"')
    tmp = envp.with_suffix(envp.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(envp)
    if os.name != "nt":
        try:
            os.chmod(envp, 0o600)
        except Exception:
            pass

def _persist_token(token: str) -> None:
    envp = _env_file_path()
    envp.parent.mkdir(parents=True, exist_ok=True)
    kv = _load_env_map(envp)
    kv["RUGCHECK_JWT"] = token
    kv["RUGCHECK_JWT_TOKEN"] = token
    _write_env_file(envp, kv)
    os.environ["RUGCHECK_JWT"] = token
    os.environ["RUGCHECK_JWT_TOKEN"] = token

def _load_dotenv_into_env() -> None:
    envp = _env_file_path()
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=envp)
        return
    except Exception:
        pass
    for k, v in _load_env_map(envp).items():
        os.environ.setdefault(k, v)

# --------------------------
# JWT helpers
# --------------------------
def _decode_jwt_noverify(token: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 3:
            return None, None
        # Add padding safely
        payload_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8", errors="ignore")
        data = json.loads(payload_json)
        exp = data.get("exp")
        iat = data.get("iat")
        return (int(exp) if exp is not None else None, int(iat) if iat is not None else None)
    except Exception:
        return None, None

def _is_token_expired(token: str, skew_seconds: int = 60) -> bool:
    exp, _ = _decode_jwt_noverify(token)
    if exp is None:
        return True
    return int(time.time()) >= (exp - max(0, skew_seconds))

def get_rugcheck_headers(force_refresh: bool = False) -> Dict[str, str]:
    """
    Sync, no-network version (for quick callers). Returns whatever is in env.
    Network refresh is done by ensure_valid_rugcheck_headers(...).
    """
    _load_dotenv_into_env()
    tok = (os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or "").strip()
    if not tok or (force_refresh and _is_token_expired(tok)):
        return {}
    return {"Authorization": f"Bearer {tok}"}

# --------------------------
# Real login: /auth/login/solana
# --------------------------
def _build_login_payload(kp: Keypair) -> Dict:
    """
    Builds the canonical message + signature required by RugCheck.
    """
    msg = {
        "message": "Sign-in to Rugcheck.xyz",
        "timestamp": int(time.time() * 1000),  # ms
        "publicKey": str(kp.pubkey()),
    }
    # Deterministic JSON for signing
    msg_json = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
    sig = kp.sign_message(msg_json.encode("utf-8"))

    # Signature bytes -> list[int]
    try:
        sig_bytes = bytes(sig)  # works on recent solders
    except Exception:
        import base58  # pip install base58
        sig_bytes = base58.b58decode(str(sig))

    return {
        "message": msg,  # object (same fields we signed)
        "signature": {"data": list(sig_bytes), "type": "ed25519"},
        "wallet": str(kp.pubkey()),
    }

async def _login_via_solana(session: aiohttp.ClientSession, login_url: str) -> str:
    """
    POST /auth/login/solana with signed message; return the raw JWT or ''.
    """
    priv = (os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or "").strip()
    if not priv:
        logger.error("SOLANA_PRIVATE_KEY (or WALLET_PRIVATE_KEY) not set; cannot login.")
        return ""
    try:
        kp = Keypair.from_base58_string(priv)
    except Exception:
        logger.error("SOLANA_PRIVATE_KEY is not a valid base58-encoded secret key.")
        return ""

    payload = _build_login_payload(kp)
    try:
        async with session.post(
            login_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=20, sock_connect=6, sock_read=12),
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                logger.error("RugCheck login failed: HTTP %s, body: %s", resp.status, body)
                return ""
            try:
                data = json.loads(body)
            except Exception:
                logger.error("Unexpected RugCheck login response: %s", body[:200])
                return ""
            # Handle common field names robustly
            token = (
                data.get("token")
                or data.get("jwt")
                or data.get("jwt_token")
                or data.get("accessToken")
                or data.get("access_token")
                or ""
            ).strip()
            if not token:
                logger.error("RugCheck login response missing token field.")
            return token
    except aiohttp.ClientError as e:
        logger.error("RugCheck login network error: %s", e)
        return ""

# --------------------------
# Main entry the fetcher uses
# --------------------------
async def ensure_valid_rugcheck_headers(
    session: Optional[aiohttp.ClientSession] = None,
    force_refresh: bool = False,
) -> Dict[str, str]:
    """
    Returns headers; refreshes token via /auth/login/solana if needed.
    """
    _load_dotenv_into_env()
    tok = (os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or "").strip()

    if tok and not force_refresh and not _is_token_expired(tok):
        return {"Authorization": f"Bearer {tok}"}

    # Need a fresh token
    login_url = os.getenv("RUGCHECK_LOGIN_URL", "https://api.rugcheck.xyz/auth/login/solana")
    close_me = False
    if session is None:
        session = aiohttp.ClientSession()
        close_me = True
    try:
        fresh = await _login_via_solana(session, login_url)
        if fresh:
            _persist_token(fresh)
            return {"Authorization": f"Bearer {fresh}"}
        # If refresh failed, return empty to let the caller soft-fail (or block if HARD_FAIL).
        return {}
    finally:
        if close_me:
            try:
                await session.close()
            except Exception:
                pass
