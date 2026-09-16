#!/usr/bin/env python3
"""Run the decoder evolution on one qBraid GPU, with exploration agents alongside.

Everything happens on a single on-demand GPU: vLLM serves the model, the
evolutionary search runs against it over localhost, and a couple of Claude
agents sit on the same box reading the same code and exploring decoding ideas in
parallel. The orchestration is meant to be invisible -- one command, and the GPU
exists only for as long as the work does.

    python run_decoder_on_gpu.py --profile gpu-h100-2x --generations 40 --agents 2

What it does, in order:

    provision -> guardrails -> ssh -> clone the repo -> install the decoder and
    serving stack -> serve the model -> launch the agents -> evolve -> copy the
    results back -> terminate

The instance is terminated on every exit path and carries the same server-side
caps as ``qbraid_remote_gpu.py``: ``max_session_minutes`` is applied before the
machine is even ready, so it stops itself even if this process is killed.
"""

from __future__ import annotations

import argparse
import contextlib
import shlex
import subprocess
import time
from pathlib import Path

from qbraid_remote_gpu import RemoteGPUEndpoint, _log

REPO_URL = "https://github.com/qBraid/evolving-quantum-compilers"
REMOTE_DIR = "$HOME/evolving-quantum-compilers"
REMOTE_DIR_TILDE = "~/evolving-quantum-compilers"

# Briefs for the agents that explore alongside the search. They read the same
# task and the same trusted evaluator, but they are explicitly fenced off from
# the evaluator and the results so they cannot contaminate the run they sit next
# to -- an agent that "improves" the scorer would invalidate every number.
AGENT_BRIEFS = [
    (
        "decoder-correlations",
        "You are on a qBraid GPU instance alongside a running ShinkaEvolve search in "
        "~/evolving-quantum-compilers. Read task_decoder/ (problem_description.md first). "
        "The search is trying to beat plain MWPM on rotated surface codes by exploiting "
        "error correlations that MWPM ignores. Explore, independently, how per-shot "
        "reweighting of the matching graph can exploit those correlations: belief "
        "propagation, two-stage matching, correlated/hook-error handling, or anything "
        "else you judge promising. Prototype in a scratch file and score it with "
        "'python task_decoder/evaluate.py --program_path <your file> --results_dir /tmp/you', "
        "which scores exactly the way the search does. Record what you tried, the score it "
        "got, and why you think it behaved that way, in notes/agent-correlations.md. "
        "STRICT: do not modify task_decoder/evaluate.py or task_decoder/benchmarks.py -- "
        "they are the trusted evaluator and changing them invalidates the run. Do not "
        "touch results/. Do not stop or restart the vLLM server on port 8000.",
    ),
    (
        "decoder-speed",
        "You are on a qBraid GPU instance alongside a running ShinkaEvolve search in "
        "~/evolving-quantum-compilers. Read task_decoder/ (problem_description.md first). "
        "The evaluator gives a candidate 300 seconds for the whole panel, and the strongest "
        "known decoders here (belief-matching) are slow in Python, so good ideas fail on "
        "wall clock rather than on accuracy. Explore how to make a correlated decoder fast "
        "enough: vectorising across shots, reserving the expensive path for the minority of "
        "syndromes that actually need it, caching reweighted graphs, batching. Measure "
        "honestly with 'python task_decoder/evaluate.py --program_path <your file> "
        "--results_dir /tmp/you' and report real timings. Write findings to "
        "notes/agent-speed.md. STRICT: do not modify task_decoder/evaluate.py or "
        "task_decoder/benchmarks.py -- they are the trusted evaluator. Do not touch "
        "results/. Do not stop or restart the vLLM server on port 8000.",
    ),
]


def ssh_stream(alias: str, command: str, timeout: int = 7200) -> int:
    """Run a remote command, streaming its output here as it goes."""
    process = subprocess.Popen(
        [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30", alias, command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    start = time.time()
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        if time.time() - start > timeout:
            process.kill()
            return 124
    return process.wait()


def launch_agents(alias: str, count: int) -> list:
    """Start exploration agents on the GPU box itself."""
    launched = []
    for name, brief in AGENT_BRIEFS[:count]:
        _log(f"launching agent {name} on {alias}")
        result = subprocess.run(
            [
                "qbraid", "agents", "launch",
                "--tool", "claude-auto",
                "--compute", alias,
                "--name", name,
                "--cwd", "/home/jovyan/evolving-quantum-compilers",
                "--instructions", brief,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            launched.append(name)
            _log(f"  {name} up")
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            _log(f"  {name} FAILED: {detail[-1][:180] if detail else 'no output'}")
    return launched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="gpu-h100-2x")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--max-session-minutes", type=int, default=150)
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--skip-diagnose", action="store_true",
        help="Skip the headroom check. Only sensible if you have run it already.",
    )
    args = parser.parse_args()

    endpoint = RemoteGPUEndpoint(
        profile=args.profile,
        model=args.model,
        max_session_minutes=args.max_session_minutes,
        auto_stop_idle_minutes=30,
    )

    with endpoint as gpu:
        alias = gpu.alias
        if not alias:
            _log("no ssh alias; aborting")
            return 1

        _log("cloning the repo and installing the decoder stack ...")
        setup = (
            "set -e; "
            f"rm -rf {REMOTE_DIR}; "
            f"git clone -q --branch {shlex.quote(args.branch)} {REPO_URL} {REMOTE_DIR}; "
            f"cd {REMOTE_DIR}; "
            "$HOME/vllm-env/bin/python -m pip install -q "
            "stim pymatching beliefmatching 'shinka-evolve>=0.0.7' 'qiskit>=2.4' pyyaml; "
            "mkdir -p notes; "
            "echo SETUP_OK"
        )
        if ssh_stream(alias, setup, timeout=2400) != 0:
            _log("setup failed")
            return 1

        if not args.skip_diagnose:
            _log("checking the panel still has headroom before spending the run ...")
            ssh_stream(
                alias,
                f"cd {REMOTE_DIR} && $HOME/vllm-env/bin/python task_decoder/diagnose_panel.py",
                timeout=2400,
            )

        agents = launch_agents(alias, args.agents) if args.agents else []
        if agents:
            _log(f"exploring alongside the search: {', '.join(agents)}")
            _log("follow them with: qbraid agents list / qbraid agents read <id>")

        _log(f"evolving the decoder for {args.generations} generations ...")
        evolve = (
            f"cd {REMOTE_DIR} && $HOME/vllm-env/bin/python -u run_evolution.py "
            "--endpoint local --base-url http://localhost:8000/v1 "
            f"--model {shlex.quote(args.model)} "
            "--config configs/decoder_gpu.yaml "
            "--sys-msg-file task_decoder/problem_description.md "
            f"--generations {args.generations} "
            "--results-dir results/decoder"
        )
        code = ssh_stream(alias, evolve, timeout=7200)

        _log("copying results and agent notes back ...")
        local = Path("results/decoder_h100")
        local.mkdir(parents=True, exist_ok=True)
        for remote, destination in (
            (f"{REMOTE_DIR_TILDE}/results/decoder/.", local),
            (f"{REMOTE_DIR_TILDE}/notes", local),
        ):
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["scp", "-q", "-o", "StrictHostKeyChecking=no", "-r",
                     f"{alias}:{remote}", str(destination)],
                    timeout=900, check=False,
                )
        gpu.save_server_log(str(local / "vllm-server.log"))
        _log(f"results in {local}")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
