# Agent skills

Two skills for driving this work through an AI coding agent on qBraid.

| skill | what it does |
|---|---|
| [`shinka-quantum-demo`](shinka-quantum-demo/) | Runs **this** project end to end — environment, pre-flight, seed score, evolution, results, independent verification. |
| [`shinka-evolve`](shinka-evolve/) | Points ShinkaEvolve at **your own** code: judging fit, designing the evolvable task and evaluator, launching, monitoring spend, harvesting the winner. |

## Install

```bash
qbraid skills install shinka-quantum-demo --target all
qbraid skills install shinka-evolve --target all
```

`--target all` installs under `~/.claude/skills` and `~/.agents/skills`, so Claude
Code, Codex and CodeQ all pick them up. `qbraid skills installed` lists what is
present.

To use them straight from a clone instead:

```bash
cp -r skills/* ~/.claude/skills/
```

Both encode the failure modes that are expensive to rediscover — budgets that
silently never bind, gateway-rejected request parameters, candidates that are
told "score 0" with no reason, benchmark instances with no headroom. Read
`shinka-evolve/SKILL.md` before designing a new task.
