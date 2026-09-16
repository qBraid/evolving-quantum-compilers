"""Make ShinkaEvolve's OpenAI calls acceptable to the qBraid AI Gateway.

The gateway rejects a handful of chat-completion parameters outright rather than
ignoring them, on the grounds that they change the shape of the response:

    {"code": "qbraid_unsupported_param",
     "message": "... does not support n. These parameters change the response
                 and are rejected rather than silently ignored."}

ShinkaEvolve's local-OpenAI provider hardcodes ``n=1`` in both its sync and async
paths (``shinka/llm/providers/local_openai.py``). The gateway rejects the
*presence* of ``n``, not its value, so **every** LLM call fails with HTTP 400 and
the run makes no progress at all -- it just backs off and retries forever.

This module strips the rejected parameters from outgoing requests. Import and
call :func:`install` before building the Shinka runner::

    import qbraid_gateway_compat
    qbraid_gateway_compat.install()

Only requests to the qBraid gateway are touched; a self-hosted vLLM/SGLang
endpoint keeps whatever parameters it was given, since it supports them.

Verified against the gateway on 2026-09-16:

    rejected : n, seed, presence_penalty, frequency_penalty, logprobs, stop
    accepted : temperature, top_p, max_tokens, max_completion_tokens,
               reasoning_effort, stream

If a run starts failing with ``qbraid_unsupported_param`` for something not in
:data:`UNSUPPORTED`, add it here.
"""

from __future__ import annotations

UNSUPPORTED = frozenset(
    {"n", "seed", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs", "stop"}
)

GATEWAY_HOSTS = ("api-v2.qbraid.com", "api.qbraid.com")

_installed = False


def _targets_gateway(bound_self) -> bool:
    """True if this completions resource is bound to a qBraid gateway client."""
    try:
        base_url = str(bound_self._client.base_url)
    except Exception:
        return False  # Unknown client: leave the request alone.
    return any(host in base_url for host in GATEWAY_HOSTS)


def _strip(kwargs: dict) -> list[str]:
    removed = [key for key in UNSUPPORTED if key in kwargs]
    for key in removed:
        kwargs.pop(key)
    return removed


def install() -> None:
    """Patch the OpenAI SDK to drop gateway-rejected parameters. Idempotent."""
    global _installed
    if _installed:
        return

    from openai.resources.chat.completions import AsyncCompletions, Completions

    sync_create = Completions.create
    async_create = AsyncCompletions.create

    def patched_create(self, *args, **kwargs):
        if _targets_gateway(self):
            _strip(kwargs)
        return sync_create(self, *args, **kwargs)

    async def patched_acreate(self, *args, **kwargs):
        if _targets_gateway(self):
            _strip(kwargs)
        return await async_create(self, *args, **kwargs)

    Completions.create = patched_create
    AsyncCompletions.create = patched_acreate
    _installed = True


def selftest(api_key: str, base_url: str, model: str) -> bool:
    """Send the exact call Shinka makes (``n=1``) and report whether it lands."""
    import openai

    install()
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with READY."}],
            n=1,
        )
    except Exception as exc:  # noqa: BLE001 - surfacing the reason is the point
        print(f"gateway compatibility shim FAILED: {exc}")
        return False
    return True


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup"))
    from check_endpoint import gateway_key

    ok = selftest(
        gateway_key(),
        "https://api-v2.qbraid.com/api/v1/ai",
        sys.argv[1] if len(sys.argv) > 1 else "gpt-5.4-nano",
    )
    print("gateway compatibility shim OK" if ok else "shim did not work")
    raise SystemExit(0 if ok else 1)
