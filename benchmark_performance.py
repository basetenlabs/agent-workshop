import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="BASETEN-API-KEY-HERE",
    base_url="https://inference.baseten.co/v1"
)

# Single request function with timing logic
async def run_single_request(index: int, prompt: str):
    start_time = time.monotonic()
    first_token_time = None
    token_count = 0

    response = await client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        stop=[],
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True
        },
        top_p=1,
        max_tokens=1000,
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0
    )

    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            if first_token_time is None:
                first_token_time = time.monotonic()
                ttft = first_token_time - start_time
                print(f"[Request {index}] Time to First Token: {ttft:.3f} sec")
            token_count += 1

    total_time = time.monotonic() - start_time
    tps = token_count / total_time if total_time > 0 else 0
    print(f"[Request {index}] Tokens: {token_count}, Time: {total_time:.3f} sec, TPS: {tps:.2f}")
    return {
        "index": index,
        "ttft": first_token_time - start_time if first_token_time else None,
        "tps": tps,
        "total_time": total_time,
        "token_count": token_count
    }

# Run N concurrent requests
async def main():
    prompt = "Write a poem about open-source AI models"
    tasks = [run_single_request(i, prompt) for i in range(5)]
    results = await asyncio.gather(*tasks)

    # Summary
    avg_ttft = sum(r["ttft"] for r in results if r["ttft"] is not None) / len(results)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    print("\n=== Summary ===")
    print(f"Avg Time to First Token: {avg_ttft:.3f} sec")
    print(f"Avg Tokens per Second: {avg_tps:.2f}")

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
