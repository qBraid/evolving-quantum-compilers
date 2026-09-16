# Evolving a quantum compiler pass with ShinkaEvolve on qBraid

An LLM-driven evolutionary search that writes a better qubit-layout heuristic
than the one it started with, running entirely on qBraid.

You pick where the model runs:

- **qBraid AI Gateway** — one API key, no GPU, no setup. Good for a first run.
- **Your own GPU** — serve an open-weight model on a qBraid on-demand GPU
  instance with vLLM or SGLang. No per-token cost, so you can run much longer.

Both paths run the same task and the same code. Only the endpoint changes.

## Get it

On a qBraid Lab instance, in one step:

```bash
qbraid projects install evolving-quantum-compilers
```

That clones this repo and installs the `shinka` environment (Qiskit + ShinkaEvolve
+ a Jupyter kernel). Or do it by hand:

```bash
git clone https://github.com/qBraid/evolving-quantum-compilers
cd evolving-quantum-compilers
qbraid envs install shinka && qbraid envs activate shinka
```

In JupyterLab, pick the **Python 3 [ShinkaEvolve]** kernel.

If you drive qBraid through an AI agent, two skills ship with this repo — see
[skills/](skills/).

---

## The problem

A quantum circuit is written as if any qubit can talk to any other. Real
hardware has a fixed coupling graph, so the compiler inserts SWAP gates to shuttle
states into place. Each SWAP costs three CNOTs, and CNOTs dominate both runtime
and error rate — reducing them is one of the highest-leverage things a quantum
compiler does.

How many you need depends heavily on where each logical qubit *starts*. That
assignment is the **initial layout**, and choosing it well is NP-hard. Qiskit
ships a good heuristic for it (`SabreLayout`). Here, an LLM tries to invent a
better one.

The function being evolved is exactly this:

```python
def choose_layout(circuit, coupling_map) -> list[int]:
    """layout[i] = the physical qubit that logical qubit i starts on."""
```

Everything else — routing pass, basis gates, optimisation level, random seed —
is pinned, so any change in the score is caused by the layout and nothing else.

## How it is scored

Five benchmark instances, each a (circuit, device-connectivity) pair:

| instance | interaction graph | device |
|---|---|---|
| `qaoa3reg-12q-grid3x4` | random 3-regular | 3×4 grid |
| `qaoa-sparse-10q-line` | random sparse | 10-qubit line |
| `ring-12q-grid3x4` | 12-cycle | 3×4 grid |
| `twolocal-10q-line` | stride-3 entangler | 10-qubit line |
| `qft-8q-line` | all-to-all (QFT) | 8-qubit line |

For each, the evaluator routes the circuit and counts two-qubit gates. The score
is the mean of `(gates with identity layout) / (gates with your layout)` — so
**1.0 means you tied the identity layout** and higher is better.

The seed program scores **0.973**: a plausible-looking degree-matching heuristic
that is actually slightly *worse* than doing nothing. That is the starting point.

A real 30-generation run over the gateway, `gpt-5.4-mini`, costing **$0.56**:

| | mean score |
|---|---|
| seed program | 0.973 |
| **best of a 30-generation run** | **1.388** |

27 of 29 candidates scored; the winner re-scores to `1.387549774348349` exactly,
which is the point of a deterministic evaluator.

Qiskit's own `SabreLayout` is the bar worth beating, and the run gets close
without clearing it:

| instance | winner | SabreLayout | ratio |
|---|---|---|---|
| `ring-12q-grid3x4` | 48 | 48 | **1.000** |
| `twolocal-10q-line` | 112 | 111 | 0.991 |
| `qaoa3reg-12q-grid3x4` | 126 | 123 | 0.976 |
| `qaoa-sparse-10q-line` | 186 | 163 | 0.876 |
| `qft-8q-line` | 215 | 170 | 0.791 |

It matched `SabreLayout` exactly on `ring-12q-grid3x4` by finding the Hamiltonian
cycle in the 3×4 grid, which needs zero SWAPs. It also *regressed* on
`qft-8q-line` — the near-zero-headroom control — trading the control away to win
elsewhere, which is exactly what the problem statement tells it not to do. Good
material for a longer run.

> Absolute numbers depend on your Qiskit version, because its transpiler changes
> between releases. The evaluator recomputes and re-fingerprints its baselines
> whenever the version changes, so scores are always internally consistent, but
> numbers from different Qiskit versions are not comparable. **The figures above
> are Qiskit 2.5.2**, which is what the `shinka` environment pins.

---

## Quickstart A — qBraid AI Gateway

No GPU needed. Runs on any qBraid Lab instance, including free CPU ones.

**There is no API key to set up.** The gateway accepts `QBRAID_ACCESS_TOKEN`,
which qBraid already exports into every shell on a Lab instance.

```bash
# 1. Confirm the endpoint works, the model exists, and you have quota
python setup/check_endpoint.py --gateway

# 2. Evolve, with a hard $2 ceiling
python run_evolution.py --endpoint gateway --budget 2.00
```

(Running off-platform? Create a key at
<https://account.qbraid.com/account/api-keys> and `export QBRAID_API_KEY=...`;
it takes precedence when set.)

`--model` picks the model; `gpt-5.4-mini` is the default. `python
setup/check_endpoint.py --gateway` lists what your key can reach.

> **On the budget.** `--budget` is enforced, but only because
> `run_evolution.py` registers live gateway pricing with Shinka first. Launch
> with `shinka_run` directly and the cap is silently ignored — see
> [the budget note](#budgets-are-not-enforced-by-default) below. The cap is also
> checked *between* generations, so with parallel proposals you can overshoot by
> roughly one generation. Budget what you can afford to lose.

## Quickstart B — your own GPU

Launch a GPU instance from the **On-Demand** tab of the
[qBraid dashboard](https://account.qbraid.com/dashboard), then:

```bash
# terminal 1 — serve a model (first run also downloads weights)
pip install "vllm==0.19.1"                    # see the driver note below
bash setup/serve_vllm.sh                      # or: bash setup/serve_sglang.sh

# terminal 2 — confirm it is up, then evolve
python setup/check_endpoint.py \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-Coder-14B-Instruct

python run_evolution.py --endpoint local \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-Coder-14B-Instruct
```

Inference is free once the GPU is running, so this config runs 100 generations
rather than 30. See [docs/CHOOSING_A_MODEL.md](docs/CHOOSING_A_MODEL.md) —
model size matters more than you would expect, and below ~14B the run tends to
stall on unparseable diffs rather than fail cleanly.

Any OpenAI-compatible endpoint works here, not just one you host. Shinka
addresses all of them as `local/<model>@<base-url>`.

### Pick the vLLM version to match the driver

**A plain `pip install vllm` fails on several qBraid GPU images.** vLLM ships a
compiled extension linked against a specific CUDA runtime, and the images do not
all carry the same NVIDIA driver — an A10 measured `570.148.08` (CUDA 12.8),
while an L4 was older still. Current vLLM links `libcudart.so.13` (CUDA 13),
which needs driver **≥ 580**, so on those images it dies at import with either
*"The NVIDIA driver on your system is too old"* or
*"ImportError: libcudart.so.13: cannot open shared object file"*.

| vLLM | torch | CUDA runtime | needs driver |
|---|---|---|---|
| ≥ 0.20 | 2.11+ | 13 | ≥ 580 |
| 0.17 – 0.19 | 2.10.0 | 12.8 | ≥ 570 |
| 0.14 – 0.16 | 2.9.1 | 12.8 | ≥ 570 |

Check with `nvidia-smi --query-gpu=driver_version --format=csv,noheader` and
install accordingly. Pointing pip at a `cu128` torch index does **not** help: it
changes which torch resolves while leaving vLLM's own binary on CUDA 13, which
fails more confusingly.

`qbraid_remote_gpu.py` (Quickstart C) does this automatically.

---

## Quickstart C — search here, model on a GPU over there

Quickstart B assumes you are already sitting on a GPU instance. This one does not:
the orchestrator stays on whatever instance you are using — a free CPU one is
fine — and the model runs on an on-demand GPU that is launched, served, tunnelled
back to `localhost`, and **terminated for you**.

```bash
python qbraid_remote_gpu.py --profile gpu-l40s \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --generations 40
```

Or from Python, where the same guarantee is a context manager:

```python
from qbraid_remote_gpu import RemoteGPUEndpoint

with RemoteGPUEndpoint(profile="gpu-l40s", model="Qwen/Qwen2.5-Coder-14B-Instruct") as gpu:
    ...  # gpu.base_url is a local URL
# the instance is gone here, including if the block raised
```

See [`notebooks/03_remote_gpu.ipynb`](notebooks/03_remote_gpu.ipynb).

The vLLM version is chosen from the instance's driver automatically, so this
path works across images with different drivers. Verified on `gpu-a10`
(driver 570.148.08): the module installed `vllm==0.19.1`, served
`Qwen2.5-Coder-7B-Instruct`, tunnelled it back, ran the search and terminated
the instance. With a 7B model only 3 of 10 candidates scored — which is the
under-14B diff-protocol failure `docs/CHOOSING_A_MODEL.md` describes, not a
problem with the endpoint. Use a 14B or larger for real runs.

### How the GPU is stopped from outliving you

A forgotten GPU is the most expensive mistake available here, so there are four
independent stops rather than one:

| stop | runs on | covers |
|---|---|---|
| `max_session_minutes` | qBraid server | this process being killed, the kernel dying, you closing the tab |
| `auto_stop_idle_minutes` | qBraid server | the run finishing but nothing using the GPU |
| `__exit__` | your machine | normal completion, and any exception |
| `atexit` + orphan sweep | your machine | failure *between* provisioning and receiving an instance id |

Only the server-side pair survive your machine going away, which is exactly the
case where you would otherwise keep paying. The orphan sweep exists because that
gap is real: the first version of this module leaked a running instance by
raising after provisioning but before it had an id to terminate.

Terminate, not stop — a stopped instance still bills for its disk.

## Watching a run

```bash
shinka_visualize --db results/gateway/programs.sqlite
```

Or read it directly — every candidate, its score, and the feedback it was given:

```python
import sqlite3
rows = sqlite3.connect("results/gateway/programs.sqlite").execute(
    "SELECT generation, combined_score, correct FROM programs ORDER BY generation"
)
print(list(rows))
```

The winning program is written to `results/<name>/best/main.py`, alongside
`original.py` and `edit.diff` showing exactly what changed.

## Repository layout

```
task/
  initial.py            seed program; the EVOLVE-BLOCK is the only mutable part
  evaluate.py           trusted evaluator — routes, counts, scores
  benchmarks.py         the five instances (candidates never see this)
  problem_description.md  the problem as stated to the model
  diagnose_panel.py     checks each instance actually has headroom
configs/
  gateway.yaml          search settings, metered path
  local_gpu.yaml        search settings, self-hosted path
setup/
  check_endpoint.py     pre-flight for any OpenAI-compatible endpoint
  serve_vllm.sh         single-instance vLLM server
  serve_sglang.sh       single-instance SGLang server
notebooks/
  01_gateway_quickstart.ipynb
  02_local_gpu_quickstart.ipynb
  03_remote_gpu.ipynb
skills/
  shinka-quantum-demo/  agent skill: run this project end to end
  shinka-evolve/        agent skill: point the search at your own code
run_evolution.py        launcher for both paths
qbraid_pricing.py       makes max_api_costs actually bind on the gateway
qbraid_gateway_compat.py  strips request params the gateway rejects
qbraid_remote_gpu.py      launches/serves/tears down a GPU worker
```

## Trying your own ideas

Edit `choose_layout` in `task/initial.py` and score it directly:

```bash
python task/evaluate.py --program_path task/initial.py --results_dir /tmp/out
```

That is exactly what the search does to every candidate, so a change that scores
well by hand will score identically in a run.

To change the problem rather than the solution, edit `task/benchmarks.py` — then
run `python task/diagnose_panel.py` to confirm your new instances actually have
room to improve. An instance where random search cannot beat the identity layout
has no gradient to climb, and the script flags it.

---

## Things worth knowing

### The gateway rejects some parameters outright

ShinkaEvolve's local-OpenAI provider hardcodes `n=1` on every call. The qBraid
gateway rejects `n` — the parameter's presence, not its value — with
`qbraid_unsupported_param`, on the grounds that it changes the shape of the
response. Unpatched, **every** call returns HTTP 400 and the run makes no
progress at all; it just backs off and retries indefinitely, which reads as a
slow run rather than a broken one.

`qbraid_gateway_compat.py` strips the rejected parameters, and
`run_evolution.py` installs it automatically on the gateway path. Verified
rejected: `n`, `seed`, `presence_penalty`, `frequency_penalty`, `logprobs`,
`stop`. Accepted: `temperature`, `top_p`, `max_tokens`, `reasoning_effort`.
Self-hosted endpoints are left alone, since they support these.

`python qbraid_gateway_compat.py` sends the exact call Shinka makes and reports
whether it lands.

### Budgets are not enforced by default

Shinka stops a run when cumulative spend hits `max_api_costs`, computed by
looking the model up in its bundled price table. Models addressed as
`local/<model>@<url>` — which is how *both* paths here reach their LLM — are not
in that table, and **Shinka prices unknown models at zero**. The cap never
trips.

`qbraid_pricing.py` fixes this by fetching live prices from the gateway's
`/models` endpoint and injecting them into Shinka's table before launch.
`run_evolution.py` calls it and refuses to start if the model still cannot be
priced. This is the main reason to launch through `run_evolution.py` rather than
`shinka_run` on the gateway path.

On a self-hosted endpoint the zero price is accurate — there is nothing to meter
— so bound those runs with generations or wall-clock instead.

### The candidate cannot fake its score

Candidates return a layout and nothing else. `evaluate.py` owns the circuits,
performs the routing, and counts the gates itself. A candidate cannot report its
own score, choose its own benchmark, or alter the routing pass. It is rejected
outright if the layout is malformed, or if calling it twice on identical inputs
gives different answers.

### Failed candidates get told why

Shinka's crash path builds its metrics from a fixed key list that has no
`text_feedback` field, so a candidate that raises is reported to the model as
"score 0" with no explanation, and the model tends to repeat the mistake.
`evaluate.py` re-injects the error and a short checklist. In testing this was
about a third of early generations, so it is worth having.

## Further reading

- [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) — the search engine
- [qBraid AI Gateway](https://docs.qbraid.com/v2/ai/integrations/ai-gateway)
- [qBraid GPU instances](https://docs.qbraid.com/v2/lab/user-guide/gpus)
- Li, Ding & Xie, *Tackling the Qubit Mapping Problem for NISQ-Era Quantum
  Devices* (ASPLOS 2019) — the SABRE algorithm this is measured against
