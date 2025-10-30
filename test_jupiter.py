# test_jupiter.py
import asyncio, aiohttp, socket

async def main():
    url = "https://quote-api.jup.ag/v6/quote"
    params = {"inputMint":"So11111111111111111111111111111111111111112","outputMint":"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v","amount":1000}
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        try:
            async with s.get(url, params=params) as r:
                print("normal session status", r.status)
        except Exception as e:
            print("normal session failed:", repr(e))
    # IPv4-only test
    try:
        conn = aiohttp.TCPConnector(family=socket.AF_INET)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s2:
            try:
                async with s2.get(url, params=params) as r2:
                    print("ipv4 session status", r2.status)
            except Exception as e2:
                print("ipv4 session failed:", repr(e2))
    except Exception as e3:
        print("ipv4 connector creation failed:", repr(e3))

asyncio.run(main())