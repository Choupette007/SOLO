import asyncio, aiohttp, json
from solana_trading_bot_bundle.trading_bot.fetching import validate_rugcheck, STATUS_FILE

TOKEN='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'  # USDC (just for a clean 200)

async def main():
    async with aiohttp.ClientSession() as s:
        ok = await validate_rugcheck(TOKEN, s)
        print('validate_rugcheck ->', ok)
        print('STATUS_FILE ->', STATUS_FILE)
        try:
            print('STATUS_JSON ->', STATUS_FILE.read_text(encoding='utf-8'))
        except Exception as e:
            print('Could not read status:', e)

asyncio.run(main())
