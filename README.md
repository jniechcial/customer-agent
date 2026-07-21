# customer-agent

Customer support agent engine — a Python experimentation platform for RAG agents,
benchmarked on [Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA). See `CLAUDE.md`
for architecture and design decisions.

## Setup

```bash
cp .env.example .env        # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY (judge), HF_TOKEN (optional)
uv sync
docker compose up -d        # Weaviate (:8080) + Phoenix tracing UI (http://localhost:6007)
```

## Usage

```bash
# 1. Index the knowledge base (6,221 Wix Help Center articles) into Weaviate.
#    Collection name is derived from chunk/embedding config, so index variants coexist.
uv run python scripts/index_kb.py [--limit N]

# 2. Talk to the agent in the terminal (multi-turn).
#    --show-steps renders interim steps (tool calls, reasoning) in gray as the agent works.
uv run python scripts/chat.py [--show-steps]

# 3. Run the eval against a split of wixqa_expertwritten (100 train / 100 validation).
uv run python scripts/run_eval.py --split validation [--limit N]
#    Re-score an existing generation run without re-running the agent:
uv run python scripts/run_eval.py --rescore runs/<run_id>.jsonl
```

Everything is traced to Phoenix: http://localhost:6007 (projects: `chat`, `indexing`, `eval-*`).

## Eval

Two phases: generation (agent answers -> `runs/<id>.jsonl`) and scoring (deepeval).

- **AnswerCorrectness** — GEval, judged by Claude Sonnet 5 against the dataset answer
  (cross-vendor to the GPT agent to avoid self-preference bias).
- **precision/recall/MRR/MAP@{5,10}** — deterministic, at article level: retrieved chunks
  are deduped to articles (first occurrence, merged across tool calls) and compared to the
  gold `article_ids`.

## Tests

```bash
uv run pytest
```

No network, no LLM calls, no Weaviate — external boundaries are faked. Covered: split
determinism/disjointness, chunking (overlap, token-exact reassembly, metadata), all
retrieval metrics against hand-computed values, pipeline ranking/slicing/formatting,
the retrieval recorder (including concurrent-question isolation), the generation
runner and its JSONL artifacts, scoring aggregation, and the tool contract
(name/schema/invocation). Runs without `.env` present, so it's CI-ready.

## Sandboxed agent runs

PRs on `agent/*` branches are opened by Claude Code running headless inside an isolated
[OpenShell](https://github.com/NVIDIA/openshell) sandbox. `scripts/run-agent` launches one and
hands your terminal straight back — the run continues in the background and cleans up after
itself. Assumptions made during a run are documented in the PR body.

```bash
# Fire and forget. Prints a run id, a Phoenix URL, and returns in ~1s.
scripts/run-agent --prompt "Make chat.py label agent answers as 'agent>' in a distinct colour"

scripts/run-agent status              # all runs: state, supervisor alive?, PR link
scripts/run-agent logs <run-id> -f    # follow the agent's transcript
scripts/run-agent cancel <run-id>     # kill a run and tear everything down
scripts/run-agent gc                  # sweep anything a crashed run left behind
```

Useful flags: `--model` (default `sonnet`), `--agent-budget` (dollar cap on the agent's own
session), `--max-calls` (hard cap on the project's LLM calls, default 60) or
`--unlimited-calls` to turn that cap off, `--no-services` for doc/UI changes that don't need
the stack, `--keep-phoenix` to keep traces after the run, and `--dry-run` to render the policy
and prompt without launching.

One-time setup — the OpenShell CLI with a running local gateway, Docker, `gh auth login`, then:

```bash
scripts/build-weaviate-seed                  # bake the current KB index into a seed image
openshell provider create --name run-agent-claude --type claude --from-existing
openshell provider create --name run-agent-github --type github --credential GITHUB_TOKEN
```

Providers inject credentials into the sandbox as env vars only, never as files. Rebuild the
seed image after re-indexing the KB.

### Architecture

Each run gets a **disposable dev environment**: a private Weaviate started from a seed image
with the KB already indexed (so no re-indexing), a fresh Phoenix, and a metered LLM proxy —
three sidecar containers on the gateway's Docker network, reachable from the sandbox by
container name.

Two boundaries do the real work:

- **A sandbox policy** grants exactly what the run needs: push access to this one repository
  (a push to any other repo, or any other host, is denied by the proxy — regardless of how
  broad the GitHub token's scopes are), plus the sidecar ports.
- **The metered proxy** holds the real OpenAI/Anthropic keys, so the sandbox only ever sees a
  per-run token in their place. It enforces a hard call cap from outside the sandbox, which the
  agent cannot raise; past the cap, calls fail with `429`. Budgets are private to a run: the
  token is what the sandbox presents as its API key, so a concurrent run sharing the same
  Docker network cannot spend a neighbour's budget.

Lifecycle is owned by a **detached supervisor**, not the shell you launched from: OpenShell's
`--no-keep` is enforced client-side, so a killed launcher would otherwise orphan the sandbox.
The supervisor waits on the run, then deletes the sandbox and its sidecars; `gc` (which also
runs automatically on every launch) sweeps any run whose supervisor died. Run state lives in
`runs/agent/<run-id>/`.

The exact grants live in `scripts/run-agent-payload/policy.template.yaml`, rendered per run.
Design notes, measurements and known gaps: [plans/OPENSHELL.md](plans/OPENSHELL.md).

## Experimentation surfaces

- `retrieval/reranker.py` — reranking stage (stub; identity passthrough)
- `indexing/chunking.py` — chunking strategies (default: 512-token windows, 64 overlap)
- `agent/prompts.py` — versioned system prompts
- `config.py` — every knob (models, k's, granularity, chunk sizes)
- `evaluation/user_simulator.py` — synthetic multi-turn user (stub; single-turn for now)
