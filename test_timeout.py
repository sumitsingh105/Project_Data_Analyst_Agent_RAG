import asyncio
import aiohttp
import time

async def test_concurrent_requests():
    """Test 3 simultaneous requests"""
    
    test_question = "Test question for timeout handling"
    
    async def make_request():
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('questions.txt', test_question, content_type='text/plain')
            
            try:
                async with session.post('http://localhost:10000/api/', data=data, timeout=300) as resp:
                    result = await resp.json()
                    elapsed = time.time() - start_time
                    print(f"✅ Request completed in {elapsed:.2f}s: {type(result)}")
                    return result
            except asyncio.TimeoutError:
                print("❌ Request timed out")
                return None
    
    # Run 3 requests simultaneously
    tasks = [make_request() for _ in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"📊 Results: {len([r for r in results if r is not None])} successful")

if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())

