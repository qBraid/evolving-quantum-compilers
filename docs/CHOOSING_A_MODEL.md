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
| under ~7B | Mostly unusable. Diffs rarely apply; most generations are wasted. |
| ~14B | Workable. Expect a meaningful fraction of failed patches; raise the share of `full` rewrites. |
| ~32B | Comfortable. Diffs apply reliably and proposals are substantive. |
| gateway models | Best quality, but metered per token. |

Prefer a **code- or instruction-tuned** model. Base/completion models are poor
at the SEARCH/REPLACE protocol regardless of size.

## Fitting the model to the GPU

bf16 weights need roughly `2 bytes x parameters`, plus KV cache and activation
headroom. A useful rule is *parameters in billions × 2.5 GB*.

| qBraid GPU | VRAM | Comfortable at bf16 | With 4-bit quantisation |
|---|---|---|---|
| RTX 4090, L4 | 24 GB | 7B | 14B |
| L40S, RTX 6000 Ada | 48 GB | 14B | 32B |
| A100 80GB, H100, H200 | 80 GB | 32B | 70B |
| GH200 | 96 GB+ | 32B comfortably | 70B+ |
| multi-GPU (2×, 4×, 8×) | — | scales with `--tensor-parallel-size` | — |

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
