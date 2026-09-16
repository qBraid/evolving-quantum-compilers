"""Verify an LLM endpoint before spending a run on it.

    # qBraid AI Gateway
    python setup/check_endpoint.py --gateway

    # a server you started yourself
    python setup/check_endpoint.py --base-url http://localhost:8000/v1 \
                                   --model Qwen/Qwen2.5-Coder-14B-Instruct

Checks, in order: the endpoint is reachable, the model name is actually served,
a completion round-trips, and -- for the gateway -- that you have quota left.

Run this first. Every failure mode it catches is one that would otherwise show
up ten minutes into an evolution run as a wall of retries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

GATEWAY_URL = "https://api-v2.qbraid.com/api/v1/ai"
GATEWAY_DEFAULT_MODEL = "gpt-5.4-nano"

def gateway_key() -> str | None:
    """The bearer token for the qBraid gateway.

    On a qBraid Lab instance ``QBRAID_ACCESS_TOKEN`` is already exported into
    every login shell, and the gateway accepts it, so there is normally nothing
    for the user to set up. ``QBRAID_API_KEY`` still wins if it is present, for
    anyone running this off-platform with a key from account.qbraid.com.
    """
    return os.environ.get("QBRAID_API_KEY") or os.environ.get("QBRAID_ACCESS_TOKEN")


OK = "  ok   "
BAD = " FAIL  "


def _request(url: str, api_key: str | None, payload: dict | None = None, timeout: int = 120):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        # Bearer works for both the qBraid gateway and for vLLM/SGLang started
        # with --api-key. The gateway also accepts X-API-Key; Bearer is the
        # common denominator, and it is what the OpenAI SDK sends natively.
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _fail(message: str, detail: str = "") -> int:
    print(f"[{BAD}] {message}")
    if detail:
        print(f"         {detail}")
    return 1


def check(base_url: str, model: str, api_key: str | None, is_gateway: bool) -> int:
    base_url = base_url.rstrip("/")
    print(f"endpoint : {base_url}")
    print(f"model    : {model}")
    print(f"auth     : {'Bearer <key>' if api_key else 'none'}")
    print()

    # 1. reachable, and does it serve the model we asked for?
    try:
        models = _request(f"{base_url}/models", api_key, timeout=30)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:300]
        if exc.code in (401, 403):
            return _fail(
                f"Authentication rejected (HTTP {exc.code}).",
                "Check your API key. For the gateway, create one at "
                "https://account.qbraid.com/account/api-keys",
            )
        return _fail(f"GET /models returned HTTP {exc.code}.", body)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"Could not reach {base_url}.",
            f"{type(exc).__name__}: {exc}. Is the server running?",
        )

    served = [entry.get("id") for entry in models.get("data", [])]
    print(f"[{OK}] endpoint reachable, serves {len(served)} model(s)")
    if model not in served:
        return _fail(
            f"{model!r} is not served by this endpoint.",
            f"Available: {', '.join(map(str, served[:12]))}",
        )
    print(f"[{OK}] {model!r} is served")

    # 2. does a completion actually round-trip?
    try:
        completion = _request(
            f"{base_url}/chat/completions",
            api_key,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
                "max_tokens": 4000,
            },
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:400]
        if exc.code == 402:
            return _fail(
                "Out of credits (HTTP 402).",
                f"{body}  Top up at https://account.qbraid.com/",
            )
        return _fail(f"Chat completion returned HTTP {exc.code}.", body)
    except Exception as exc:  # noqa: BLE001
        return _fail("Chat completion failed.", f"{type(exc).__name__}: {exc}")

    content = (completion["choices"][0]["message"].get("content") or "").strip()
    usage = completion.get("usage", {})
    print(f"[{OK}] completion round-trips (reply: {content[:40]!r})")
    print(f"         tokens in/out: {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}")

    # 3. gateway only: how much quota is left?
    if is_gateway:
        try:
            quota = _request(f"{base_url}/quota", api_key, timeout=30)
        except Exception as exc:  # noqa: BLE001
            print(f"[{BAD}] could not read quota ({exc}) -- continuing anyway")
        else:
            print(
                f"[{OK}] plan {quota.get('plan')!r}, "
                f"quota remaining {quota.get('quotaRemaining')}/{quota.get('quotaAllowed')}"
            )
            if not quota.get("quotaRemaining") and not quota.get("autoSwitchToCredits"):
                return _fail(
                    "No quota left and credit fallback is off.",
                    "The run would fail immediately.",
                )

    print()
    print("Endpoint is good. Shinka model string:")
    print(f"    local/{model}@{base_url}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Check the qBraid AI Gateway (uses $QBRAID_API_KEY).",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, ending in /v1.")
    parser.add_argument("--model", help="Model name as the server reports it.")
    parser.add_argument(
        "--api-key",
        help="Bearer token. Defaults to $QBRAID_API_KEY for --gateway, "
        "else $LOCAL_OPENAI_API_KEY if set.",
    )
    args = parser.parse_args()

    if args.gateway:
        base_url = args.base_url or GATEWAY_URL
        model = args.model or GATEWAY_DEFAULT_MODEL
        api_key = args.api_key or gateway_key()
        if not api_key:
            return _fail(
                "No qBraid credentials found.",
                "Create a key at https://account.qbraid.com/account/api-keys "
                "then: export QBRAID_API_KEY=...",
            )
        return check(base_url, model, api_key, is_gateway=True)

    if not args.base_url or not args.model:
        parser.error("Pass --gateway, or both --base-url and --model.")
    api_key = args.api_key or os.environ.get("LOCAL_OPENAI_API_KEY")
    return check(args.base_url, args.model, api_key, is_gateway=False)


if __name__ == "__main__":
    sys.exit(main())
