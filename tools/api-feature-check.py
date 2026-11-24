#!/usr/bin/env python3
"""The API-parity check: every served model supports the whole API but vision.

Runs against a TinyTitanServer the operator already started on 127.0.0.1 (it
never starts, stops or installs a model) and prints PASS/FAIL with evidence for
each surface a client relies on:

  plain chat, streamed chat with usage, a tool call, a streamed tool call, the
  follow-up turn that reads a tool result, a stop string, a strict JSON schema,
  thinking, Anthropic /v1/messages with tools (streamed), and Responses
  /v1/responses with tools.

Exit status is 0 only when every check passes. AGENTS.md makes this the bar a
model route must clear before it is called supported.

Usage:
  tools/api-feature-check.py --port 8080 [--model <id>]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--model", help="model id to send; default: the first /v1/models row")
options = parser.parse_args()
ROOT = f"http://127.0.0.1:{options.port}"
with urllib.request.urlopen(ROOT + "/v1/models", timeout=30) as listing:
    MODEL = options.model or json.load(listing)["data"][0]["id"]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
ASK = "What's the weather in Berlin right now? Use the tool."
results = []


def request(path, body, stream=False):
    req = urllib.request.Request(
        ROOT + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        if not stream:
            return json.loads(resp.read())
        events = []
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    event = json.loads(payload)
                    if isinstance(event, dict):
                        event["_arrived"] = time.monotonic()
                    events.append(event)
        return events


def streamed_incrementally(events, has_text):
    """Whether text deltas arrived while the answer was still decoding.

    A route that decodes the whole answer and then sends every delta in one
    burst still yields several deltas, so the count alone cannot catch it
    (TT-128). Decoded tokens are milliseconds apart (2 ms is 500 tok/s) while
    a burst arrives microseconds apart, so the text deltas must span at least
    2 ms per gap. Prefill time is excluded on purpose: it says nothing about
    whether decode streamed."""
    times = [e["_arrived"] for e in events if has_text(e)]
    if len(times) < 2:
        return False
    return times[-1] - times[0] >= 0.002 * (len(times) - 1)


def check(name, fn):
    started = time.time()
    try:
        ok, evidence = fn()
    except urllib.error.HTTPError as err:
        ok, evidence = False, f"HTTP {err.code}: {err.read().decode()[:300]}"
    except Exception as err:  # noqa: BLE001 - report every failure as a row
        ok, evidence = False, f"{type(err).__name__}: {err}"
    results.append((name, ok))
    print(
        f"{'PASS' if ok else 'FAIL'} {name} ({time.time() - started:.1f}s): {evidence}", flush=True
    )


base = {"model": MODEL, "temperature": 0, "max_tokens": 400}
chat = "/v1/chat/completions"


def plain():
    data = request(
        chat,
        {
            **base,
            "reasoning_effort": "off",
            "messages": [{"role": "user", "content": "Capital of France? One word."}],
        },
    )
    text = data["choices"][0]["message"]["content"] or ""
    return "paris" in text.lower(), repr(text[:80])


def streaming():
    events = request(
        chat,
        {
            **base,
            "stream": True,
            "reasoning_effort": "off",
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Count 1 to 5, comma separated."}],
        },
        stream=True,
    )
    pieces = [
        c["delta"].get("content")
        for e in events
        for c in e.get("choices", [])
        if c.get("delta", {}).get("content")
    ]
    usage = [e["usage"] for e in events if e.get("usage")]
    text = "".join(pieces)
    incremental = streamed_incrementally(
        events,
        lambda e: any(c.get("delta", {}).get("content") for c in e.get("choices", [])),
    )
    return len(pieces) > 1 and incremental and "5" in text and bool(
        usage
    ), f"{len(pieces)} deltas, incremental={incremental} {text[:60]!r} usage={bool(usage)}"


def tool_call():
    data = request(chat, {**base, "tools": TOOLS, "messages": [{"role": "user", "content": ASK}]})
    choice = data["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    ok = (
        choice["finish_reason"] == "tool_calls"
        and calls
        and calls[0]["function"]["name"] == "get_weather"
        and "berlin" in calls[0]["function"]["arguments"].lower()
    )
    return bool(ok), json.dumps(calls)[:200]


def streaming_tool_call():
    events = request(
        chat,
        {**base, "stream": True, "tools": TOOLS, "messages": [{"role": "user", "content": ASK}]},
        stream=True,
    )
    deltas = [
        tc
        for e in events
        for c in e.get("choices", [])
        for tc in (c.get("delta", {}).get("tool_calls") or [])
    ]
    finish = [
        c.get("finish_reason")
        for e in events
        for c in e.get("choices", [])
        if c.get("finish_reason")
    ]
    args = "".join((d.get("function") or {}).get("arguments") or "" for d in deltas)
    return bool(
        deltas
    ) and "tool_calls" in finish and "berlin" in args.lower(), f"finish={finish} args={args[:80]!r}"


def follow_up():
    first = request(chat, {**base, "tools": TOOLS, "messages": [{"role": "user", "content": ASK}]})
    calls = first["choices"][0]["message"].get("tool_calls")
    if not calls:
        return False, "no tool call to follow up"
    data = request(
        chat,
        {
            **base,
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": ASK},
                {"role": "assistant", "content": None, "tool_calls": calls},
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": '{"temp_c": 18, "sky": "cloudy"}',
                },
            ],
        },
    )
    text = data["choices"][0]["message"]["content"] or ""
    return "18" in text, repr(text[:100])


def stop_string():
    # Stopping must cut the answer where the stop string begins: the text
    # before it is kept and nothing from it on is returned.
    data = request(
        chat,
        {
            **base,
            "stop": ["delta"],
            "reasoning_effort": "off",
            "messages": [
                {
                    "role": "user",
                    "content": "Repeat exactly, nothing else: alpha beta gamma delta epsilon",
                }
            ],
        },
    )
    text = data["choices"][0]["message"]["content"] or ""
    ok = "gamma" in text and "delta" not in text and "epsilon" not in text
    return ok, repr(text)


def json_schema():
    data = request(
        chat,
        {
            **base,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "city",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                        "required": ["city", "country"],
                    },
                },
            },
            "messages": [{"role": "user", "content": "Name the capital of Japan and its country."}],
        },
    )
    text = data["choices"][0]["message"]["content"] or ""
    doc = json.loads(text)
    return set(doc) == {"city", "country"} and "tokyo" in doc["city"].lower(), text


def thinking():
    data = request(
        chat,
        {
            **base,
            "reasoning_effort": "high",
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": "What is 17 * 23?"}],
        },
    )
    msg = data["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return bool(reasoning) and "391" in (
        msg.get("content") or ""
    ), f"reasoning {len(reasoning)} chars, answer {(msg.get('content') or '')[:40]!r}"


def anthropic_tools():
    events = request(
        "/v1/messages",
        {
            "model": MODEL,
            "max_tokens": 400,
            "stream": True,
            "temperature": 0,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "input_schema": TOOLS[0]["function"]["parameters"],
                }
            ],
            "messages": [{"role": "user", "content": ASK}],
        },
        stream=True,
    )
    starts = [
        e
        for e in events
        if e.get("type") == "content_block_start" and e["content_block"].get("type") == "tool_use"
    ]
    stop = [e["delta"].get("stop_reason") for e in events if e.get("type") == "message_delta"]
    return bool(starts) and "tool_use" in stop, f"tool_use blocks={len(starts)} stop={stop}"


def responses_tools():
    data = request(
        "/v1/responses",
        {
            "model": MODEL,
            "temperature": 0,
            "max_output_tokens": 400,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": TOOLS[0]["function"]["parameters"],
                }
            ],
            "input": ASK,
        },
    )
    calls = [o for o in data.get("output", []) if o.get("type") == "function_call"]
    return bool(calls) and calls[0]["name"] == "get_weather", json.dumps(calls)[:160]


for name, fn in [
    ("plain", plain),
    ("streaming", streaming),
    ("tool call", tool_call),
    ("streaming tool call", streaming_tool_call),
    ("tool result follow-up", follow_up),
    ("stop string", stop_string),
    ("strict JSON schema", json_schema),
    ("thinking", thinking),
    ("Anthropic messages + tools (stream)", anthropic_tools),
    ("Responses + tools", responses_tools),
]:
    check(name, fn)
passed = sum(ok for _, ok in results)
print(f"SUMMARY {MODEL}: {passed}/{len(results)} passed")
sys.exit(0 if passed == len(results) else 1)
