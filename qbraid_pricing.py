"""Teach ShinkaEvolve what the qBraid AI Gateway charges.

Why this exists
---------------
ShinkaEvolve stops a run once cumulative spend reaches ``max_api_costs``. It
computes that spend by looking the model up in its bundled ``pricing.csv``.

Models reached through a custom endpoint -- anything addressed as
``local/<model>@<url>``, which is how both paths in this repo address their
LLM -- are usually absent from that table. When a model is missing, Shinka
silently prices every call at **zero**, and ``max_api_costs`` never trips. The
run does not stop. On a self-hosted GPU that is harmless; against a metered
gateway it means your budget cap is decorative.

So before launching, we fetch the live price list from the gateway's
``/models`` endpoint and inject it into Shinka's in-memory pricing table.
Prices come from the endpoint rather than a hardcoded table here, so they stay
correct as qBraid's lineup changes.

Usage:
    from qbraid_pricing import register_gateway_pricing
    register_gateway_pricing(api_key)     # call before building the runner
"""

from __future__ import annotations

import json
import urllib.request
from typing import Dict, Optional

GATEWAY_URL = "https://api-v2.qbraid.com/api/v1/ai"

# The gateway quotes prices in credits. The published conversion is
# 100 credits = 1 USD. Shinka's table is in USD per million tokens.
CREDITS_PER_USD = 100.0


def fetch_gateway_pricing(
    api_key: str, base_url: str = GATEWAY_URL, timeout: int = 30
) -> Dict[str, Dict[str, float]]:
    """Return ``{model_name: {"input": usd_per_1M, "output": usd_per_1M}}``."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())

    prices: Dict[str, Dict[str, float]] = {}
    for entry in payload.get("data", []):
        model = entry.get("id")
        pricing = (entry.get("_qbraid") or {}).get("pricing") or {}
        input_credits = pricing.get("inputCreditsPerMillionTokens")
        output_credits = pricing.get("outputCreditsPerMillionTokens")
        if model is None or input_credits is None or output_credits is None:
            continue
        prices[model] = {
            "input": float(input_credits) / CREDITS_PER_USD,
            "output": float(output_credits) / CREDITS_PER_USD,
        }
    return prices


def register_gateway_pricing(
    api_key: str,
    base_url: str = GATEWAY_URL,
    overwrite: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Inject live gateway prices into Shinka's pricing table.

    Args:
        api_key: qBraid API key.
        base_url: Gateway base URL.
        overwrite: Replace entries already present in Shinka's table. The
            bundled table prices the upstream vendor APIs; the gateway may
            charge differently, so the gateway's own numbers win by default.
        verbose: Print a short summary.

    Returns:
        The price map that was applied.

    Raises:
        RuntimeError: if the prices could not be fetched. Fail loudly -- a
            silent fallback here is what produces an unenforced budget.
    """
    try:
        import pandas as pd
        from shinka.llm.providers import pricing as shinka_pricing
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ShinkaEvolve is not installed (pip install shinka-evolve)."
        ) from exc

    try:
        prices = fetch_gateway_pricing(api_key, base_url)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not fetch gateway pricing from {base_url}/models: {exc}. "
            "Refusing to continue: without it, max_api_costs would not be "
            "enforced and the run could spend without limit."
        ) from exc

    if not prices:
        raise RuntimeError(
            f"{base_url}/models returned no priced models. Refusing to continue; "
            "max_api_costs would not be enforced."
        )

    table = shinka_pricing._PRICING_DF
    million = 1_000_000.0
    added, updated = [], []

    for model, price in sorted(prices.items()):
        present = model in table.index
        if present and not overwrite:
            continue
        # Shinka stores per-token prices; the CSV loader divides by 1e6 on load,
        # so injected rows must already be per-token to match.
        row = {
            "provider": "qbraid_gateway",
            "input_price": price["input"] / million,
            "output_price": price["output"] / million,
            "input_price_tier2": float("nan"),
            "output_price_tier2": float("nan"),
            "tier_threshold": float("nan"),
            "is_reasoning": False,
            "think_temp_fixed": False,
            "requires_reasoning": False,
        }
        # Only assign columns the installed version actually has, so this keeps
        # working if upstream adds or removes fields.
        table.loc[model] = pd.Series(
            {key: value for key, value in row.items() if key in table.columns}
        )
        (updated if present else added).append(model)

    if verbose:
        print(
            f"qBraid gateway pricing registered: "
            f"{len(added)} added, {len(updated)} updated "
            f"({len(prices)} models priced)."
        )
        sample = sorted(prices)[:3]
        for model in sample:
            print(
                f"    {model:<20} ${prices[model]['input']:.2f} in / "
                f"${prices[model]['output']:.2f} out  per 1M tokens"
            )

    return prices


def verify(model: str) -> bool:
    """True if Shinka can now price ``model`` (so ``max_api_costs`` will bind)."""
    try:
        from shinka.llm.providers.pricing import calculate_cost, model_exists
    except ImportError:
        return False
    if not model_exists(model):
        return False
    cost_in, cost_out = calculate_cost(model, 1_000_000, 1_000_000)
    return (cost_in + cost_out) > 0


if __name__ == "__main__":
    import os
    import sys

    key = os.environ.get("QBRAID_API_KEY")
    if not key:
        print("Set QBRAID_API_KEY first.", file=sys.stderr)
        sys.exit(1)
    applied = register_gateway_pricing(key)
    print()
    for name in sorted(applied):
        status = "priced" if verify(name) else "NOT PRICED"
        print(f"  {name:<20} {status}")
