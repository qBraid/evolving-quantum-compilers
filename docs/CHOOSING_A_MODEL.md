# Choosing a model

The single most common way a self-hosted run disappoints is picking a model
that is too small. This note explains why, and what to run on the GPU you have.

## What the model is actually asked to do

ShinkaEvolve does not ask for free-form code. It asks for an edit to an existing
program, usually as a diff:

```
<<<<<<< SEARCH
    logical_order = sorted(range(num_logical), key=lambda q: (-logical_load[q], q))
=======
    logical_order = _spectral_order(weights, num_logical)
>>>>>>> REPLACE
```

The SEARCH block must match the seed program **character for character**. If it
does not, the patch is discarded and the generation is wasted.

That is a harder instruction-following problem than writing the same code from
scratch, and it is where small models fall down. They produce plausible code
inside a block that does not quite match — wrong indentation, a paraphrased
comment, a line silently omitted. Shinka retries
(`max_patch_resamples`, `max_patch_attempts`) and then moves on.

The failure is quiet. Nothing errors out; the run simply makes no progress while
the GPU bills by the minute. If your score is flat after twenty generations,
check how many candidates actually applied before you tune anything else.

## Rough guidance

| Parameters | Behaviour on this task |
|---|---|
| under ~4B | Mostly unusable. Diffs rarely apply; most generations are wasted. |
| ~9B | Workable. Expect a meaningful fraction of failed patches; raise the share of `full` rewrites. |
| ~27-30B | Comfortable. Diffs apply reliably and proposals are substantive. |
| gateway models | Best quality, but metered per token. |

Prefer a **code- or instruction-tuned** model. Base/completion models are poor
at the SEARCH/REPLACE protocol regardless of size.

Generation matters as much as size: these thresholds track roughly one size
class *down* per model generation, so a current 9B behaves about like the 14B
this guidance was first written against.

## Mixture-of-experts changes the trade

The advice above is about *quality*, and it used to cost you speed: a bigger
model follows the diff protocol better and decodes proportionally slower, which
matters because Shinka is latency-bound — it waits on one proposal at a time per
parallel job.

MoE models break that coupling. `Qwen/Qwen3-Coder-30B-A3B-Instruct` has 30B
total parameters but activates only ~3B per token, so it follows diffs like a
30B and decodes closer to a 3B. You pay for it in VRAM, not in time: the whole
30B must be resident even though a fraction is used per token.

So the rule is no longer "take the biggest model that fits". It is:

- **take the largest MoE that fits in VRAM**, because active parameters, not
  total, set your throughput; and
- fall back to a dense model only when no MoE fits — on 24 GB, that means
  `Qwen/Qwen3.5-9B`.

This is also why quantisation is usually worth it here (see below): it buys
total parameters, which is what the diff protocol cares about.

## Fitting the model to the GPU

bf16 weights need roughly `2 bytes x parameters`, plus KV cache and activation
headroom. A useful rule is *parameters in billions × 2.5 GB*.

| qBraid GPU | VRAM | Comfortable at bf16 | Suggested model |
|---|---|---|---|
| RTX 4090, L4 | 24 GB | ~9B | `Qwen/Qwen3.5-9B` |
| L40S, RTX 6000 Ada | 48 GB | ~18B | `Qwen/Qwen3.8-27B-FP8` (FP8 needs Hopper+) |
| A100 80GB, H100, H200 | 80 GB | ~30B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| GH200 | 96 GB+ | ~35B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| multi-GPU (2×, 4×, 8×) | — | scales with `--tensor-parallel-size` | `Qwen/Qwen3.8-27B` |

All of the above are Apache-2.0. Note that `Qwen3.8-Flash-Next` is **not** a
small model despite the name — it is 180B and licensed `other`; the
self-hostable member of the 3.8 line is `Qwen3.8-27B`.

The serve scripts set tensor parallelism automatically from the number of
visible GPUs, rounding down to a power of two (tensor parallel size must divide
the attention head count; powers of two always do).

Quantised weights let you run a larger model on the same card. For this
workload that is usually the right trade — a 4-bit 32B follows the diff protocol
better than a bf16 14B.

## If the model is small anyway

Shift the search toward whole-function rewrites, which are much easier to get
right than exact-match diffs:

```yaml
patch_types: [diff, full, cross]
patch_type_probs: [0.25, 0.65, 0.10]     # default local config is 0.45/0.45/0.10
```

Also make sure `max_tokens` in the config is comfortably below the server's
context length (`--max-model-len` / `--context-length`). The prompt carries the
seed program, inspirations and evaluator feedback; an overflow is a hard
failure, not a truncation.

## Context length

The prompt for this task is small — a ~150-line program plus feedback — so
32k is ample and the defaults in `setup/serve_*.sh` reflect that. A longer
context costs KV cache memory that is better spent on batching, so do not raise
it without a reason.

## Checking your choice

Before a long run, do a short one:

```bash
python run_evolution.py --endpoint local \
    --base-url http://localhost:8000/v1 --model <your-model> \
    --generations 5 --results-dir results/probe
```

Then count how many candidates actually scored:

```python
import sqlite3
rows = sqlite3.connect("results/probe/programs.sqlite").execute(
    "SELECT generation, combined_score, correct FROM programs ORDER BY generation"
).fetchall()
scored = sum(1 for _, _, ok in rows if ok)
print(f"{scored}/{len(rows)} candidates scored")
```

Most should score. If most do not, read `results/probe/gen_*/` — the rejected
patch and the error are both there — and either move up a size class or raise
the `full`-rewrite probability.

For reference, a 30-generation gateway run against `gpt-5.4-mini` scored 27 of 29
candidates (93%) and improved the seed from 0.973 to 1.388 for $0.56. A scored
fraction in the 80-95% range is healthy; below ~60% the model is fighting the
diff protocol rather than the problem, and a size class up will help more than
any config change.
