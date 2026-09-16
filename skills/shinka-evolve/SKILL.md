---
name: shinka-evolve
description: Point LLM-driven evolutionary search (ShinkaEvolve) at the user's own code on qBraid - designing the evolvable task and a trustworthy evaluator, launching against the AI Gateway or a self-hosted GPU, capping spend, and harvesting the winner. Use for "evolve this function", "find a better heuristic for X", "AlphaEvolve-style search".
---

# ShinkaEvolve on qBraid — bring your own problem

ShinkaEvolve mutates one marked block of Python, scores each candidate with an
evaluator **you** write, and keeps what wins. Your job is almost entirely the
evaluator: the search is only ever as trustworthy as the number it climbs.

The `evolving-quantum-compilers` repo is a complete worked reference. Read its
`task/` directory before writing a new one — the scaffolding is task-agnostic and
only `task/` changes.

## First: is this the right tool?

Say no when it is. A run that cannot work wastes real money.

Good fit — **all** of these hold:

- The objective is a **number a program computes**, not a judgement call.
- Evaluation is **cheap** (seconds). The LLM should be the bottleneck, not you.
- There is **headroom**: something clearly better than the seed exists.
- The answer is **short code** — a heuristic, scoring rule, schedule, ansatz.
- Success is **verifiable independently** of the candidate.

Bad fit: objectives needing human taste; evaluations taking minutes each;
problems where the seed is already near-optimal; anything needing a large new
codebase rather than a better function body.

If evaluation is expensive, fix that first — subsample, cache, shrink the
instance — or the run dies on wall-clock, not on ideas.

## Setup

```bash
qbraid envs install shinka && qbraid envs activate shinka
```

Or build one: `qbraid envs create -n shinka -r requirements.txt -k "Python 3 [ShinkaEvolve]"`.
Note `qbraid envs create` rejects compound PEP 440 specifiers — write `qiskit==2.5.2`,
not `qiskit>=2.4,<2.6`, and use full `x.y.z` versions.

**Credentials: nothing to set up.** The gateway accepts `QBRAID_ACCESS_TOKEN`,
already exported into every Lab shell. Do not ask the user for an API key. Never
print the token.

## The three files

### 1. Seed program — `initial.py`

Only the code between the markers is mutable:

```python
# EVOLVE-BLOCK-START
def choose_layout(circuit, coupling_map):
    ...
# EVOLVE-BLOCK-END
```

- Put the **contract in the module docstring**: signature, argument types,
  return shape, and the hard rules. The model reads this.
- Seed something **plausible but clearly improvable**. A seed that already wins
  leaves nothing to find; a seed that crashes wastes early generations. A seed
  that scores slightly *worse* than the trivial baseline is ideal — it proves
  the metric discriminates.
- Helpers outside the block are **immutable**. If the model should be able to
  change something, move it inside.
- Keep the entry point the evaluator calls **outside** the block so candidates
  cannot rewrite their own interface.

### 2. Evaluator — `evaluate.py`

Writes two files into `results_dir`:

```python
# metrics.json
{
  "combined_score": 1.31,          # what the search maximises
  "public":  {...},                # shown to the model
  "private": {...},                # kept for your analysis, never shown
  "text_feedback": "per-instance breakdown..."
}
# correct.json
{"correct": true}
```

Non-negotiable properties:

- **The evaluator owns the data.** Candidates return a small answer — a list, a
  number, a schedule. The evaluator holds the problem instances and computes the
  score itself. Never let a candidate report its own score or choose its own
  benchmark; that is the failure mode where a run "succeeds" and means nothing.
- **Check determinism.** Call the candidate twice on identical inputs and reject
  on disagreement. Otherwise the archive fills with lucky noise.
- **Validate hard.** Wrong shape, out-of-range, duplicated entries — reject with
  a precise message.
- **Always populate `text_feedback`, including on the failure path.** Shinka's
  crash handler builds metrics from a fixed key list with **no `text_feedback`
  field**, so a candidate that raises is reported to the model as "score 0" with
  no reason, and the model repeats the mistake. Re-inject the exception and a
  short checklist yourself. This was about a third of early generations in
  testing.
- **Fingerprint cached baselines** with the version of anything whose behaviour
  drifts (Qiskit, torch, a solver). Reusing a baseline across a library upgrade
  silently shifts every number.
- **Per-instance feedback beats a scalar.** Tell the model which instances it
  won and lost; otherwise it is guessing which change did what.

### 3. Problem description

States the problem and the rules. Deliberately **do not suggest strategies** —
finding those is the search's job. Do say: the scoring rule, the hard
requirements, the time limit, and which instances are controls.

### Before spending anything: prove there is headroom

Write the equivalent of `diagnose_panel.py` — random or perturbed candidates
should beat the trivial baseline on every instance. An instance with no gradient
cannot be climbed and silently absorbs the whole run. Keeping one near-zero-
headroom instance as a deliberate **control** is good practice; know which it is.

## Launch

```bash
python run_evolution.py --endpoint gateway --model gpt-5.4-mini \
    --budget 2.00 --generations 30 --results-dir results/myrun
```

Confirm the spend with the user first. Start small — 5–10 generations — to prove
the loop closes before committing a real budget.

### Config footguns

These fail in non-obvious ways; all are documented in `configs/gateway.yaml`:

| setting | rule |
|---|---|
| `max_novelty_attempts` | must be **≥ 1**. `0` makes every proposal fail instantly. |
| `llm_dynamic_selection` | must not be null — selection runs before any query. Use `ucb1`. |
| `reasoning_efforts` | must be a real level (`"medium"`). The default `""` is rejected by reasoning models. |
| `use_text_feedback` | `true`. Highest-value setting in the file. |
| `embedding_model`, `novelty_llm_models`, `meta_llm_models` | null unless wanted — each issues extra billable calls, and the gateway serves no embedding model. |
| `num_islands` | 1 for short runs; islands need generations to pay off. |
| `archive_criteria: {loc: -0.3}` | mild pressure toward shorter programs, so the model stops bolting on special cases. |

### Budgets do not bind by default

Shinka prices models from a bundled table. Anything addressed as
`local/<model>@<url>` — which is how both qBraid paths reach their LLM — is
**unknown to that table, and unknown models are priced at zero**, so
`max_api_costs` never trips. `qbraid_pricing.py` injects live gateway prices and
`run_evolution.py` refuses to launch unpriced. **Never launch the gateway path
with bare `shinka_run`.**

The cap is checked *between* generations, so parallel proposals can overshoot by
roughly one generation. On a self-hosted endpoint zero is the true price — bound
those runs by generations or wall clock.

## Monitor

```bash
sqlite3 results/myrun/programs.sqlite \
  "SELECT generation, round(combined_score,4), correct FROM programs ORDER BY generation"
```

`shinka_visualize --db results/myrun/programs.sqlite` for the interactive view.

Watch the **scored fraction** before the score. Lots of `correct=0` means the
model is failing the SEARCH/REPLACE diff protocol, not that the problem is hard.
Fix by using a stronger model — below ~14B on a self-hosted endpoint this is the
dominant failure.

Flat score with high correctness means the opposite: the metric may not
discriminate, or there was no headroom. Re-run the headroom check.

## Harvest

The winner lands in `results/<name>/best/`. **Always re-score it independently**
before reporting — a deterministic evaluator must reproduce the number exactly.
If it does not, the evaluator is not deterministic and every number in the run is
suspect.

Then read the winning code and say **why** it works. A score with no mechanism is
not a result. Check specifically that it solved the problem rather than the
metric: an unexpected jump is a reward hack until proven otherwise.

## Self-hosted GPU path

Free inference, so run far longer (100+ generations).

```bash
qbraid compute up <profile>          # then, on the instance:
pip install "vllm==0.19.1" && bash setup/serve_vllm.sh
python run_evolution.py --endpoint local --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --generations 100
```

Any OpenAI-compatible endpoint works. Use ≥14B — smaller models stall on
unparseable diffs. **Terminate when done** (`qbraid compute instances terminate
<label> --yes`); stopped instances still bill storage. Copy `results/` off the
instance first — on-demand filesystems are deleted on terminate.

**Match the vLLM version to the driver.** Current vLLM links `libcudart.so.13` (CUDA 13) and needs driver >= 580; several qBraid GPU images are older (an A10 measured 570.148.08), where `vllm==0.19.1` (torch 2.10, CUDA 12.8) is the newest that works. Check with `nvidia-smi --query-gpu=driver_version --format=csv,noheader`. `qbraid_remote_gpu.py` resolves this automatically.
