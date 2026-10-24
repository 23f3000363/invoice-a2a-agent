"""Optional live AI Pipe smoke test. Makes one small model request; no grader call."""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)

async def main():
    from app import storage
    await storage.init_db()
    from app.ai_decision import _call_model
    answer = await _call_model([
        {"role": "system", "content": "Return exactly the requested token, no other text."},
        {"role": "user", "content": "Return exactly AI_CONNECTION_OK"},
    ])
    print("AI provider response:", repr(answer))
    if "AI_CONNECTION_OK" not in answer:
        raise SystemExit("AI provider responded, but exact-output check did not pass.")
    print("PASS: AI provider connection")

if __name__ == "__main__":
    asyncio.run(main())
