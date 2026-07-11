# customer-agent

Experimentation platform for a customer-support RAG agent, benchmarked on
[Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA). Python-only, no frontend, no
productionization. Everything is traced; the retrieval pipeline is built to be swapped
out piece by piece.

## Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Benchmark data | `wixqa_expertwritten` (200 rows) | Human-grounded, multi-article synthesis. Other subsets available behind a CLI flag. |
| Knowledge base | `wix_kb_corpus` (6,221 articles) | The retrieval target; indexed in full regardless of QA split. |
| Split | Deterministic 50/50 train/validation, seed=42, computed at load time | Train for prompt/pipeline optimization, validation for eval. No committed artifact; HF dataset revision pinned in config. |
| Agent framework | OpenAI Agents SDK | Structured agent/tool abstractions, first-class tracing hooks. |
| Models | Agent: GPT-5.5. Judge: `claude-sonnet-5` | Cross-vendor judge avoids self-preference bias. |
| Vector store | Weaviate in Docker | Scripts run on host; only infra in Docker. |
| Embeddings | Client-side (`text-embedding-3-small`) | Embedding model is a config knob, not baked into the collection. |
| Tracing | Arize Phoenix (Docker) + OpenInference | Agent runs, tool calls, LLM calls, eval runs — all traced. Agents SDK's default export to the OpenAI platform is disabled. |
| Eval framework | deepeval | LLM-judged correctness + custom deterministic retrieval metrics. |
| Tooling | uv, Python 3.12, `src/` layout | |

## Architecture

```
                       ┌──────────────────────────────┐
 scripts/chat.py ────► │  Agent (OpenAI Agents SDK)   │
 scripts/run_eval.py ─►│  system prompt + agentic loop │──► OpenAI API
                       └──────────────┬───────────────┘
                                      │ tool: search_knowledge_base(query)
                                      ▼
                       ┌──────────────────────────────┐
                       │  RetrievalPipeline           │
                       │  retrieve ──► rerank (stub)  │──► Weaviate (Docker)
                       └──────────────────────────────┘
 scripts/index_kb.py ──► Chunker ──► Embedder ──► Weaviate

 Everything ──────────► Phoenix (Docker, OTLP)
```

`src/customer_agent/` packages: `config.py` (pydantic-settings, every knob),
`tracing.py` (Phoenix bootstrap, one call per entry point), `agent/` (Agent, prompts,
tool), `retrieval/` (pipeline, retriever, reranker stub, embeddings), `indexing/`
(chunker, indexer), `data/` (HF loaders, splits), `evaluation/` (runner, metrics,
user-simulator stub). Entry points live in `scripts/`; eval artifacts in `runs/`
(gitignored, JSONL per run).

## Key design points

- **Indexing**: token-based chunking, 512 tokens / 64 overlap (config knobs); `Chunker`
  is a protocol so other strategies drop in. Chunks carry `article_id`, `url`,
  `article_type`, `chunk_index`. The Weaviate collection name is derived from the
  chunk/embedding config (e.g. `KB_chunk512o64_te3small`) so index experiments coexist;
  the active collection is a config knob. Indexing is batched and idempotent.
- **Retrieval**: embed query → vector search top `k_retrieve` (20) → rerank (identity
  stub) → top `k_final` (5) → format with article id/url so the agent can cite. Output
  granularity (chunks vs full articles) is a knob. `RetrievalResult` keeps ranked chunks
  AND deduped ranked `article_id`s (first-occurrence order) — the latter is what eval
  consumes. During eval, retrievals are recorded per-question so metrics see everything
  across multiple tool calls (concat in call order, then dedup — v1 aggregation rule).
- **Agent**: one Agent, one tool; the model decides when/how often to search.
- **Eval**: two phases, deliberately separated. Generation (async, bounded concurrency)
  persists per-question JSONL to `runs/<run_id>.jsonl` so runs can be re-scored via
  `--rescore` without re-paying generation. Scoring: GEval answer correctness vs the
  dataset answer (judge = Claude via deepeval custom model; criteria must tolerate style
  differences — dataset answers are procedural markdown), plus deterministic
  precision/recall/MRR/MAP@{5,10} at **article level**. Per-question results (all
  scores + the judge's reasoning) go to `runs/<run_id>.scores.jsonl` AND to Phoenix as
  span annotations: each question is wrapped in a `question-<i>` root span whose ids are
  persisted in the artifact, so scoring (incl. `--rescore`) attaches judge score/label/
  reasoning to the right trace — browse/sort per conversation in the Phoenix UI. Only
  per-question-meaningful metrics are annotated (correctness, precision/recall); mrr/map
  are run-level means and stay in the scores file/summary only.
- **Synthetic user (stubbed)**: the eval runner drives conversations through a
  `UserSimulator` interface; v1 is single-turn (asks the dataset question, stops). An
  LLM-backed multi-turn simulator later is a config change plus one class.
- **Phoenix**: UI on host port **6007** (6006 was taken locally). Project names per
  entry point: `chat`, `indexing`, `eval-<run_id>`.

## Status (2026-07-11)

M0–M5 done and smoke-tested end to end with real APIs: full KB indexed (10,081 chunks),
chat grounded with 3–4 refined searches per question, eval plumbing validated on 3
questions (correctness 0.63, recall@10 0.67 — NOT a baseline). TODO: full 100-question
validation run for a real baseline.

## Non-goals (for now)

Reranker implementations, alternative chunkers, hybrid/BM25 search, query rewriting,
multi-turn eval, any frontend or deployment. The architecture allows all of these;
they're intentionally not built.

## Open questions

- Per-call retrieval metrics if the agent starts issuing many refined queries.
- Reranker direction (local cross-encoder vs API) — decides future deps/keys.
- GEval criteria wording — style mismatch between conversational answers and procedural
  dataset answers is the main risk to score validity.
- Statistical rigor at n=100 (bootstrap CIs, paired tests) — add when experiments
  disagree by small margins.
