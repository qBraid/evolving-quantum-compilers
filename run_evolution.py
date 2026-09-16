#!/usr/bin/env python3
"""Launch a ShinkaEvolve run against either qBraid inference path.

    # qBraid AI Gateway (metered)
    export QBRAID_API_KEY=...
    python run_evolution.py --endpoint gateway --budget 2.00

    # a model you are serving yourself on this instance's GPU
    python run_evolution.py --endpoint local \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-Coder-14B-Instruct

Use this rather than `shinka_run` for the gateway path. It registers live
gateway pricing with Shinka first, and without that step `max_api_costs` is
silently ignored -- see qbraid_pricing.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
GATEWAY_URL = "https://api-v2.qbraid.com/api/v1/ai"

sys.path.insert(0, str(HERE / "setup"))
from check_endpoint import check, gateway_key  # noqa: E402

# Shown to the model as system context alongside the seed program. It states the
# problem and the rules; it deliberately does not suggest strategies, because
# the point is for the search to find them.
TASK_SYS_MSG = """\
You are improving a qubit-layout heuristic for a quantum compiler.

A quantum circuit assumes any qubit can interact with any other. Real hardware
has a fixed coupling graph, so the compiler inserts SWAP gates to move states
into place. Each SWAP costs three CNOTs, and CNOTs dominate both runtime and
error rate. How many are needed depends strongly on where each logical qubit
starts -- the initial layout -- and choosing that layout is the whole task.

You are editing one function:

    choose_layout(circuit, coupling_map) -> list[int]

`layout[i]` is the physical qubit that logical qubit `i` starts on. Entries must
be distinct and lie within the device.

Scoring: for each of five benchmark instances the evaluator routes the circuit
with a fixed, seeded SabreSwap pass and counts two-qubit gates. Your score is
the mean of (gates with the identity layout) / (gates with your layout), so
higher is better and 1.0 means you matched the identity layout. Routing, basis
gates, optimisation level and seed are all pinned -- the layout is the only
thing that varies.

Hard requirements:
  - Determinism. The evaluator calls your function twice on identical inputs
    and rejects the candidate if the results differ. Seed any randomness with a
    fixed constant.
  - Valid layouts: right length, distinct entries, inside the device.
  - No filesystem, no network, no importing the evaluator or its benchmarks.
  - A few seconds per instance at most.

One instance (`qft-8q-line`) has an all-to-all interaction graph and almost no
headroom; it is a control. Do not distort the heuristic to chase it, but do not
regress on it either. Read the per-instance feedback -- it tells you exactly
which instances you are winning and losing.
"""


def build_model_string(model: str, base_url: str) -> str:
    """Shinka addresses any OpenAI-compatible endpoint as local/<model>@<url>."""
    return f"local/{model}@{base_url.rstrip('/')}"


def preflight(base_url: str, model: str, api_key: str | None, is_gateway: bool) -> None:
    """Refuse to launch against an endpoint that is not actually working."""
    print("=" * 68)
    print("Pre-flight")
    print("=" * 68)
    if check(base_url, model, api_key, is_gateway) != 0:
        print("\nPre-flight failed. Fix the endpoint before launching.", file=sys.stderr)
        raise SystemExit(1)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        choices=["gateway", "local"],
        required=True,
        help="gateway = qBraid AI Gateway (metered); local = your own server.",
    )
    parser.add_argument("--config", help="YAML config. Defaults by endpoint.")
    parser.add_argument("--model", help="Model name as the endpoint reports it.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL ending in /v1.")
    parser.add_argument("--api-key", help="Defaults to $QBRAID_API_KEY / $LOCAL_OPENAI_API_KEY.")
    parser.add_argument("--budget", type=float, help="Override max_api_costs, in USD.")
    parser.add_argument("--generations", type=int, help="Override num_generations.")
    parser.add_argument("--results-dir", help="Override results_dir.")
    parser.add_argument(
        "--eval-timeout",
        default="00:05:00",
        help="Per-candidate wall clock limit (HH:MM:SS). Default 00:05:00.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt."
    )
    args = parser.parse_args()

    is_gateway = args.endpoint == "gateway"
    config_path = Path(
        args.config or (HERE / "configs" / ("gateway.yaml" if is_gateway else "local_gpu.yaml"))
    )
    if not config_path.is_absolute():
        config_path = HERE / config_path
    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    evo = config["evo_config"]

    # ---- resolve endpoint, model, key ------------------------------------
    if is_gateway:
        base_url = args.base_url or GATEWAY_URL
        model = args.model or "gpt-5.4-mini"
        api_key = args.api_key or gateway_key()
        if not api_key:
            print(
                "No qBraid credentials found.\n"
                "On a qBraid Lab instance QBRAID_ACCESS_TOKEN is exported for you;\n"
                "if you are running off-platform, create a key at\n"
                "  https://account.qbraid.com/account/api-keys\n"
                "then: export QBRAID_API_KEY=...",
                file=sys.stderr,
            )
            return 1
    else:
        if not args.base_url or not args.model:
            parser.error("--endpoint local requires --base-url and --model.")
        base_url = args.base_url
        model = args.model
        api_key = args.api_key or os.environ.get("LOCAL_OPENAI_API_KEY")

    # Shinka's local provider reads its bearer token from this variable. The
    # qBraid gateway accepts the key as a bearer token, so the stock provider
    # works against it with no patching.
    if api_key:
        os.environ["LOCAL_OPENAI_API_KEY"] = api_key

    preflight(base_url, model, api_key, is_gateway)

    # ---- apply overrides --------------------------------------------------
    evo["llm_models"] = [build_model_string(model, base_url)]
    if args.generations is not None:
        evo["num_generations"] = args.generations
    if args.results_dir is not None:
        evo["results_dir"] = args.results_dir
    if args.budget is not None:
        evo["max_api_costs"] = args.budget
    evo["task_sys_msg"] = TASK_SYS_MSG
    evo["init_program_path"] = str(HERE / evo["init_program_path"])

    # ---- make the gateway accept Shinka's calls at all --------------------
    # Shinka's local-OpenAI provider hardcodes n=1, which the gateway rejects
    # outright; without this every call 400s and the run never progresses.
    if is_gateway:
        import qbraid_gateway_compat  # noqa: PLC0415

        qbraid_gateway_compat.install()

    # ---- make the budget real --------------------------------------------
    if is_gateway:
        from qbraid_pricing import register_gateway_pricing, verify  # noqa: PLC0415

        register_gateway_pricing(api_key, base_url)
        if not verify(model):
            print(
                f"\nRefusing to launch: Shinka still cannot price {model!r}, so "
                "max_api_costs would not be enforced.",
                file=sys.stderr,
            )
            return 1
        print(f"Budget enforcement active for {model!r}.\n")
    elif evo.get("max_api_costs") is not None:
        print(
            "Note: max_api_costs is set but this is a self-hosted endpoint, so "
            "Shinka prices every call at $0 and the cap will never trip. Stop "
            "the run on generations or by hand.\n"
        )

    # ---- confirm ----------------------------------------------------------
    budget = evo.get("max_api_costs")
    print("=" * 68)
    print("Launch plan")
    print("=" * 68)
    print(f"  endpoint     : {base_url}")
    print(f"  model        : {model}")
    print(f"  generations  : {evo['num_generations']}")
    print(f"  budget       : {f'${budget:.2f} (enforced)' if is_gateway and budget else 'none (self-hosted)'}")
    print(f"  results      : {evo['results_dir']}")
    print(f"  eval timeout : {args.eval_timeout} per candidate")
    print()
    if is_gateway and not args.yes:
        print("This spends real qBraid quota/credits.")
        if input("Type 'yes' to launch: ").strip().lower() != "yes":
            print("Aborted.")
            return 1
        print()

    # ---- launch -----------------------------------------------------------
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner  # noqa: PLC0415
    from shinka.database import DatabaseConfig  # noqa: PLC0415
    from shinka.launch import LocalJobConfig  # noqa: PLC0415

    runner = ShinkaEvolveRunner(
        evo_config=EvolutionConfig(**evo),
        # Relative to the task directory: Shinka runs the evaluator with this
        # as its working directory, which is also how `import benchmarks`
        # inside evaluate.py resolves.
        job_config=LocalJobConfig(
            eval_program_path=str(HERE / "task" / "evaluate.py"),
            time=args.eval_timeout,
        ),
        db_config=DatabaseConfig(**config["db_config"]),
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()

    print()
    print("Done. Inspect the run with:")
    print(f"    shinka_visualize --db {evo['results_dir']}/evolution.sqlite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
