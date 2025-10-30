import asyncio
from utils.rugcheck_auth import get_rugcheck_headers
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestRugcheck')

async def test_token():
    async with aiohttp.ClientSession() as session:
        headers = get_rugcheck_headers()  # Synchronous call
        async with session.get(
            'https://api.rugcheck.xyz/v1/tokens/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v/report',
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15, sock_connect=5, sock_read=10)
        ) as resp:
            if resp.status == 200:
                logger.info('Token valid: %s', await resp.json(content_type=None))
            else:
                body = await resp.text()
                logger.error('Token invalid: HTTP %s — %s', resp.status, body[:200])

if __name__ == '__main__':
    asyncio.run(test_token())
