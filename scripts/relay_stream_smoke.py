"""Smoke: does the relay accept stream=True + stream_options include_usage?

This is THE load-bearing assumption of the audit-hardening change: if the
relay 400s on stream_options (some OpenAI-compatible gateways do), every
production call would fail over to Anthropic. One tiny real call settles it.
Cost: a few hundred tokens on gpt-5.5.
"""
import os
import sys

# Load .env the same way the app does (no external deps needed).
for line in open("/home/yebo/quant-agent/.env"):
    line = line.strip()
    if line.startswith("export "):
        line = line[len("export "):]
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI

base_url = os.environ.get("OPENAI_BASE_URL") or None
print(f"base_url: {base_url or 'api.openai.com (default)'}")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url,
                timeout=60.0, max_retries=0)

stream = client.chat.completions.create(
    model="gpt-5.5",
    max_completion_tokens=512,
    messages=[
        {"role": "system", "content": "You are a smoke test."},
        {"role": "user", "content": "Reply with exactly: OK"},
    ],
    stream=True,
    stream_options={"include_usage": True},
)

parts, finish_reason, usage, n_chunks = [], None, None, 0
for chunk in stream:
    n_chunks += 1
    if getattr(chunk, "usage", None) is not None:
        usage = chunk.usage
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        continue
    delta = getattr(choices[0], "delta", None)
    piece = getattr(delta, "content", None) if delta is not None else None
    if piece:
        parts.append(piece)
    fr = getattr(choices[0], "finish_reason", None)
    if isinstance(fr, str):
        finish_reason = fr

content = "".join(parts)
print(f"chunks: {n_chunks}")
print(f"content: {content!r}")
print(f"finish_reason: {finish_reason!r}")
print(f"usage: {usage!r}")

ok = bool(content) and finish_reason == "stop" and usage is not None
print("RESULT:", "PASS — relay supports streaming + include_usage" if ok else
      "PARTIAL — see fields above (usage None means the estimate fallback will engage)")
sys.exit(0 if content and finish_reason else 1)
