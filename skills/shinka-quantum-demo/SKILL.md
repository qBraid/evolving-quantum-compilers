---
name: shinka-quantum-demo
description: Run the qBraid compiler-evolution project: LLM-driven evolutionary search that invents a better qubit-layout heuristic, scored by the two-qubit gate count Qiskit emits after routing. Use for "try ShinkaEvolve", "evolve a qubit layout", "beat SabreLayout".
---

# Evolving a quantum compiler pass on qBraid

An LLM proposes candidate `choose_layout` heuristics; a trusted evaluator routes
five benchmark circuits with a pinned Qiskit pass and counts two-qubit gates. The
score is the mean of `(gates with identity layout) / (gates with your layout)`,
so **1.0 ties the identity layout** and higher is better.

Two inference paths, same task and same code — only the endpoint changes:

| path | cost | when |
|---|---|---|
| **qBraid AI Gateway** | per token | first run, any instance, no GPU |
| **self-hosted on a qBraid GPU** | GPU time only | long runs, no per-token cost |

## Get set up

```bash
qbraid projects install evolving-quantum-compilers    # repo + environment in one step
```

If the project is not in the Hub yet, do it by hand:

```bash
git clone https://github.com/qBraid/evolving-quantum-compilers
qbraid envs install shinka        # or: qbraid envs create -n shinka -r requirements.txt
qbraid envs activate shinka
```

In JupyterLab the kernel is **Python 3 [ShinkaEvolve]**.

## Credentials — there is nothing to set up

The gateway accepts `QBRAID_ACCESS_TOKEN`, which is already exported into every
login shell on a qBraid Lab instance. **Do not ask the user for an API key** and
do not tell them to visit the API-keys page; that step only applies off-platform.
`QBRAID_API_KEY` still takes precedence if they have set one.

Never print the token. To confirm one exists:
`[ -n "$QBRAID_ACCESS_TOKEN" ] && echo set || echo unset`

## Run it

Always in this order. Each step catches a failure the next one would otherwise
hit ten minutes in.

```bash
# 1. Endpoint reachable, model served, completion round-trips, quota left
python setup/check_endpoint.py --gateway

# 2. What the seed scores (~3 s, no LLM, no spend). Establishes the starting point.
python task/evaluate.py --program_path task/initial.py --results_dir /tmp/seed_eval

# 3. Evolve, with an enforced ceiling
python run_evolution.py --endpoint gateway --model gpt-5.4-mini \
    --budget 2.00 --generations 30 --results-dir results/gateway
```

**Launch through `run_evolution.py`, never `shinka_run`.** See the budget rule
below — this is not a style preference, the cap genuinely does not bind otherwise.

Confirm the spend with the user before step 3. Report the model and the cap.

### GPU path

Only worth it for long runs. Launch an on-demand instance, then:

```bash
pip install "vllm==0.19.1"        # match the driver, see below
bash setup/serve_vllm.sh                      # or setup/serve_sglang.sh
python setup/check_endpoint.py --base-url http://localhost:8000/v1 --model <model>
python run_evolution.py --endpoint local --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3.5-9B --generations 100
```

Model size matters more than expected: below ~14B the run stalls on unparseable
SEARCH/REPLACE diffs rather than failing cleanly. See `docs/CHOOSING_A_MODEL.md`.

**Match the vLLM version to the driver.** Current vLLM links `libcudart.so.13` (CUDA 13) and needs driver >= 580; several qBraid GPU images are older (an A10 measured 570.148.08), where `vllm==0.19.1` (torch 2.10, CUDA 12.8) is the newest that works. Check with `nvidia-smi --query-gpu=driver_version --format=csv,noheader`. `qbraid_remote_gpu.py` resolves this automatically.

**Terminate the instance when done** — `qbraid compute instances terminate <label>
--yes`. Stopped still bills storage; terminated is the only zero-cost state.

## Read the results

```python
import sqlite3, pandas as pd
rows = sqlite3.connect("results/gateway/programs.sqlite").execute(
    "SELECT generation, combined_score, correct FROM programs ORDER BY generation"
).fetchall()
df = pd.DataFrame(rows, columns=["generation", "score", "correct"])
print(f"{int(df.correct.sum())}/{len(df)} scored   seed {df.score.iloc[0]:.4f} -> best {df.score.max():.4f}")
```

The winner is written to `results/<name>/best/`. `shinka_visualize --db
results/<name>/programs.sqlite` opens the interactive view.

**Check the scored fraction first.** A low fraction means the model is failing
the diff protocol, not that the task is hard — a bigger model fixes it.

**Verify the winner independently** before reporting a number. The evaluator is
deterministic, so re-scoring must reproduce it exactly:

```bash
python task/evaluate.py --program_path results/gateway/best/main.py --results_dir /tmp/verify
```

## Rules that are easy to get wrong

**Budgets do not bind by default.** Shinka computes spend from a bundled price
table. Both paths address the LLM as `local/<model>@<url>`, which is not in that
table, and **Shinka prices unknown models at zero** — so `max_api_costs` silently
never trips. `qbraid_pricing.py` fetches live prices from the gateway's `/models`
and injects them; `run_evolution.py` refuses to launch if the model still cannot
be priced. This is the whole reason to use the launcher. The cap is checked
*between* generations, so parallel proposals can overshoot by about one
generation.

On a self-hosted endpoint zero is the correct price — bound those runs with
`--generations` or wall-clock instead.

**Scores move between Qiskit releases.** The task counts gates the transpiler
emits, and Qiskit's routing changes between minor versions. The evaluator
fingerprints its cached baselines with the Qiskit version and recomputes on
change, so numbers are always internally consistent — but numbers from different
Qiskit versions are not comparable. State the Qiskit version whenever quoting a
score. The published environment pins it for exactly this reason.

**Candidates cannot fake a score.** They return a list of integers; the evaluator
owns the circuits, the routing and the counting. A candidate is rejected if the
layout is malformed, or if calling it twice on identical inputs disagrees.

**Never hand-edit `task/baselines.json`** — it is regenerated per Qiskit version.

## Changing the problem

Edit `task/benchmarks.py`, then **always** run `python task/diagnose_panel.py`.
It checks each instance actually has headroom; an instance where random search
cannot beat the identity layout has no gradient to climb and will silently waste
the whole run. `qft-8q-line` is deliberately a near-zero-headroom control.

To evolve something other than qubit layout, use the `shinka-evolve` skill.
