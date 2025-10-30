import asyncio, aiohttp
from solana_trading_bot_bundle.trading_bot.fetching import validate_rugcheck, STATUS_FILE, WHITELISTED_TOKENS

# Pick something unlikely to be whitelisted or even a token (it will still write status on non-200)
TEST_MINT = "11111111111111111111111111111111"

async def main():
    print("Is USDC whitelisted?", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" in WHITELISTED_TOKENS)
    async with aiohttp.ClientSession() as s:
        ok = await validate_rugcheck(TEST_MINT, s)
        print("validate_rugcheck(TEST_MINT) ->", ok)
        print("STATUS_FILE ->", STATUS_FILE)
        try:
            print("STATUS_JSON ->", STATUS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print("Could not read status:", e)

asyncio.run(main())
