# sign_send_swap.py
import os
import sys
import json
import base64
import asyncio
from typing import Optional
from dotenv import load_dotenv  # pip install python-dotenv
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts

# Load .env from current directory (or parent if you prefer)
load_dotenv()

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


def _load_keypair_from_env() -> Keypair:
    """
    Load a signing keypair from environment variables or sensible defaults.

    Supported sources (checked in order):
      1) SOLANA_KEYPAIR_JSON -> path to a JSON file containing 32 or 64 integers
      2) SOLANA_PRIVATE_KEY (or PRIVATE_KEY) -> base58 private key (32-byte seed or 64-byte secret)
      3) Fallback to typical Solana CLI id.json paths
    """
    # 1) JSON file path
    key_json_path = os.getenv("SOLANA_KEYPAIR_JSON", "").strip()
    if key_json_path and os.path.exists(key_json_path):
        with open(key_json_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        b = bytes(arr)
        if len(b) == 32:
            return Keypair.from_seed(b)
        if len(b) == 64:
            return Keypair.from_bytes(b)
        raise ValueError(f"SOLANA_KEYPAIR_JSON must contain 32 or 64 bytes; got {len(b)}")

    # 2) Base58 private key string (64-byte secret or 32-byte seed)
    b58 = (os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("PRIVATE_KEY") or "").strip()
    if b58:
        try:
            import base58  # pip install base58
        except Exception:
            raise RuntimeError("Please install base58: pip install base58")
        raw = base58.b58decode(b58)
        if len(raw) == 32:
            return Keypair.from_seed(raw)
        if len(raw) == 64:
            return Keypair.from_bytes(raw)
        raise ValueError(f"SOLANA_PRIVATE_KEY decoded to {len(raw)} bytes (need 32 or 64)")

    # 3) Fallbacks (optional): Solana CLI default id.json
    defaults = [
        os.path.expanduser("~/.config/solana/id.json"),
        os.path.join(os.environ.get("USERPROFILE", ""), ".config", "solana", "id.json"),
    ]
    for p in defaults:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f)
            b = bytes(arr)
            if len(b) == 32:
                return Keypair.from_seed(b)
            return Keypair.from_bytes(b)

    raise RuntimeError("No key found. Set SOLANA_KEYPAIR_JSON or SOLANA_PRIVATE_KEY in .env")


def _deserialize_vtx(raw: bytes) -> VersionedTransaction:
    """
    Cross-version loader:
    - Prefer VersionedTransaction.from_bytes (newer solders)
    - Fallback to VersionedTransaction.deserialize (older solders)
    """
    if hasattr(VersionedTransaction, "from_bytes"):
        return VersionedTransaction.from_bytes(raw)
    if hasattr(VersionedTransaction, "deserialize"):
        return VersionedTransaction.deserialize(raw)
    raise RuntimeError(
        "Your solders package lacks both from_bytes and deserialize on VersionedTransaction. "
        "Upgrade solders (e.g., pip install -U solders solana)."
    )


def _load_unsigned_vtx(b64_path: str) -> VersionedTransaction:
    with open(b64_path, "r", encoding="ascii") as f:
        b64 = f.read().strip()
    raw = base64.b64decode(b64)
    return _deserialize_vtx(raw)


async def _try_update_blockhash(vx: VersionedTransaction, client: AsyncClient) -> bool:
    """
    Best-effort: fetch a fresh blockhash and try to patch it into the message.
    Many solders builds don't allow mutating the message from Python; if so, this will return False.
    """
    try:
        latest = await client.get_latest_blockhash()
        bh = latest.value.blockhash
    except Exception as e:
        print("Failed to fetch latest blockhash:", e)
        return False

    print("Fetched latest blockhash:", bh)
    updated = False

    # Try a couple of common mutation hooks (often unavailable)
    try:
        if hasattr(vx.message, "recent_blockhash"):
            setattr(vx.message, "recent_blockhash", bh)
            updated = True
    except Exception:
        pass

    try:
        msg = getattr(vx, "message", None)
        if msg is not None and hasattr(msg, "set_recent_blockhash"):
            try:
                msg.set_recent_blockhash(bh)
                updated = True
            except Exception:
                pass
    except Exception:
        pass

    if not updated:
        print(
            "Couldn't update the blockhash in-place. If you hit BlockhashNotFound, "
            "rebuild swap_tx.b64 immediately before signing."
        )

    return updated


def _extract_signature_from_response(resp) -> Optional[str]:
    """Handle different client return shapes and pull out the transaction signature."""
    # solana-py usually returns an RpcResponse with .value
    try:
        sig = getattr(resp, "value", None)
        if sig:
            return sig
    except Exception:
        pass
    # some variants return dict-like {"result": "..."}
    try:
        return resp["result"]
    except Exception:
        pass
    return None


async def _simulate_and_maybe_send(vx: VersionedTransaction, kp: Keypair, do_send: bool):
    # Sign after any attempted blockhash refresh
    async with AsyncClient(RPC_URL) as client:
        # Try to refresh the blockhash (won't hurt if it fails)
        await _try_update_blockhash(vx, client)
        signed = VersionedTransaction(vx.message, [kp])

        # Simulate
        sim = await client.simulate_transaction(signed)
        print("=== SIMULATION ===")
        print("err:", sim.value.err)
        if sim.value.logs:
            print("logs:")
            for line in sim.value.logs:
                print("  ", line)

        # Friendly hint for the common expiry case
        if sim.value.err and "BlockhashNotFound" in str(sim.value.err):
            print("Hint: The blockhash expired. Rebuild swap_tx.b64 and run this script immediately.")
        if sim.value.err is not None:
            print("Simulation failed; not sending.")
            return

        if do_send:
            # Try modern send API
            try:
                sent = await client.send_transaction(
                    signed,
                    opts=TxOpts(skip_preflight=False, preflight_commitment="processed"),
                )
                print("=== SEND (send_transaction) ===")
                print("raw response:", sent)
                sig = _extract_signature_from_response(sent)
                if sig:
                    print("signature:", sig)
                    return
                else:
                    print("No signature in response; will fall back to send_raw_transaction.")
            except Exception as e:
                print("send_transaction failed; will fall back to send_raw_transaction:", repr(e))

            # Fallback: send_raw_transaction (base64)
            raw_b64 = base64.b64encode(bytes(signed)).decode("ascii")
            sent2 = await client.send_raw_transaction(
                raw_b64,
                opts=TxOpts(skip_preflight=False, preflight_commitment="processed"),
            )
            print("=== SEND (send_raw_transaction) ===")
            print("raw response:", sent2)
            sig2 = _extract_signature_from_response(sent2)
            if sig2:
                print("signature:", sig2)
            else:
                print("No signature in send_raw_transaction response either.")


async def main():
    # Usage: python sign_send_swap.py <swap_tx.b64> [send]
    if len(sys.argv) < 2:
        print("Usage: python sign_send_swap.py <swap_tx.b64> [send]")
        return
    b64_path = sys.argv[1]
    do_send = (
        len(sys.argv) >= 3
        and sys.argv[2].lower() in ("1", "true", "yes", "y", "send")
    )

    print("RPC_URL:", RPC_URL)

    kp = _load_keypair_from_env()
    print("Using pubkey:", kp.pubkey())

    # Optional sanity check vs .env
    env_pub = (os.getenv("SOLANA_PUBLIC_KEY") or "").strip()
    if env_pub and str(kp.pubkey()) != env_pub:
        print("WARNING: Derived keypair pubkey != SOLANA_PUBLIC_KEY in .env")
        print("  derived:", kp.pubkey())
        print("  .env   :", env_pub)

    vx = _load_unsigned_vtx(b64_path)
    await _simulate_and_maybe_send(vx, kp, do_send)


if __name__ == "__main__":
    asyncio.run(main())



