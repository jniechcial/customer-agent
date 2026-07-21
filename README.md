# customer-agent

Customer support agent engine — a Python experimentation platform for RAG agents,
benchmarked on [Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA). See `CLAUDE.md`
for architecture and design decisions. This is an experimentation platform and is not production-ready.

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

PRs labelled `agent/*` are opened by Claude Code running inside an isolated OpenShell sandbox with no persistent state and network access restricted to this repository only. They represent autonomous, headless changes — assumptions made during the run are documented in the PR body.

## Experimentation surfaces

- `retrieval/reranker.py` — reranking stage (stub; identity passthrough)
- `indexing/chunking.py` — chunking strategies (default: 512-token windows, 64 overlap)
- `agent/prompts.py` — versioned system prompts
- `config.py` — every knob (models, k's, granularity, chunk sizes)
- `evaluation/user_simulator.py` — synthetic multi-turn user (stub; single-turn for now)
