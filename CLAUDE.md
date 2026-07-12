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
| Reranker | Voyage `rerank-2.5` (API) | Effectively free at project scale (200M-token free tier), top-tier quality, no torch dependency. `reranker=identity` env knob restores the passthrough for A/B. |
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
                       │  retrieve ──► rerank (Voyage)│──► Weaviate (Docker)
                       └──────────────────────────────┘
 scripts/index_kb.py ──► Chunker ──► Embedder ──► Weaviate

 Everything ──────────► Phoenix (Docker, OTLP)
```

`src/customer_agent/` packages: `config.py` (pydantic-settings, every knob),
`tracing.py` (Phoenix bootstrap, one call per entry point), `agent/` (Agent, prompts,
tool), `retrieval/` (pipeline, retriever, rerankers, embeddings), `indexing/`
(chunker, indexer), `data/` (HF loaders, splits), `evaluation/` (runner, metrics,
user-simulator stub). Entry points live in `scripts/`; eval artifacts in `runs/`
(gitignored, JSONL per run).

## Key design points

- **Indexing**: token-based chunking, 512 tokens / 64 overlap (config knobs); `Chunker`
  is a protocol so other strategies drop in. Chunks carry `article_id`, `url`,
  `article_type`, `chunk_index`. The Weaviate collection name is derived from the
  chunk/embedding config (e.g. `KB_chunk512o64_te3small`) so index experiments coexist;
  the active collection is a config knob. Indexing is batched and idempotent.
- **Retrieval**: embed query → vector or hybrid BM25+vector search (`search_mode`/
  `hybrid_alpha` knobs; hybrid uses Weaviate's native fusion on the same collection)
  top `k_retrieve` (20) → rerank (Voyage
  `rerank-2.5`; `reranker`/`rerank_model` knobs, identity fallback) → top `k_final` (5)
  → format with article id/url so the agent can cite. Reranking reorders all 20
  candidates without truncating, replaces chunk scores with Voyage relevance, and emits
  an OpenInference RERANKER span (input/output docs inspectable in Phoenix); the
  reranker and search-mode ids are recorded per question in run artifacts. Output
  granularity (chunks vs full articles) is a knob. `RetrievalResult` keeps ranked chunks
  AND deduped ranked `article_id`s (first-occurrence order) — the latter is what eval
  consumes. During eval, retrievals are recorded per-question so metrics see everything
  across multiple tool calls. Aggregation rule (v2): articles ordered by the **best rank
  achieved in any call**, ties by call order, deduped. (v1 concatenated calls in order,
  burying a later search's rank-1 hit behind the first call's whole ranking — it
  penalized agents for searching more; switching to v2 alone moved v2b recall@5
  0.793→0.823.) The merge is recomputed from per-call rankings at scoring time, so
  `--rescore` applies the current rule to old artifacts.
- **Agent**: one Agent, one tool; the model decides when/how often to search.
- **Eval**: two phases, deliberately separated. Generation (async, bounded concurrency)
  persists per-question JSONL to `runs/<run_id>.jsonl` so runs can be re-scored via
  `--rescore` without re-paying generation. Scoring: GEval answer correctness vs the
  dataset answer (judge = Claude via deepeval custom model; criteria must tolerate style
  differences — dataset answers are procedural markdown), plus deterministic
  precision/recall/MRR/MAP@{5,10} at **article level**. The judge is **grounded in
  gold∩retrieved article texts** (passed as GEval context): extras beyond the terse
  reference answer are fine when supported by a gold article the agent actually
  retrieved; specifics found in neither are hallucinations. Gold articles the agent
  never retrieved are withheld on purpose — matching them is parametric memory, not
  grounding. Correctness stays binary; the partial flag means "on the right track", and
  the pair reads as three buckets: correct / on-track (0,1) / wrong (0,0). The two flags
  are independent LLM calls and overlap on correct answers, so read buckets, not the
  partial mean. Per-question results (all
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

## Status (2026-07-12)

M0–M5 done; full KB indexed (10,081 chunks). First real experiment ladder complete on
the 50-question train split, scored with the grounded judge (artifacts in `runs/`,
old-judge scores in `runs/judge-v1-backup/`). Rows below use the v2 best-rank merge
rule; V2A/V2C were not rescored and keep v1-merge retrieval numbers (marked †).
Judge buckets shift a few points on any rescore (LLM variance) — the grounding input
is identical under both merge rules, so bucket deltas across these rescores are noise.

| config | correct | on-track | wrong | recall@5 | mrr@5 |
|---|---|---|---|---|---|
| V1 prompt + identity (`v1-identity-train`) | 0.28 | 0.60 | 0.12 | 0.743 | 0.563 |
| V1 + voyage (`voyage-reranker-train`) | 0.22 | 0.72 | 0.06 | 0.793 | 0.684 |
| V2A + voyage † | 0.39 | 0.49 | 0.12 | 0.769† | 0.688† |
| **V2B + voyage** (active) | **0.46** | 0.46 | 0.08 | 0.823 | 0.709 |
| V2C + voyage † | 0.45 | 0.41 | 0.14 | 0.786† | 0.657† |

Findings so far: the reranker improves retrieval but not answers (the agent compensates
for weak rankings with more searches — under the v2 merge rule identity-vs-voyage
recall@5 is 0.743 vs 0.793, much closer than the v1 rule suggested); the prompt converts
retrieval into correctness — V2B ("faithful messenger for one primary article") is ~2×
the V1 baseline. The wrong bucket barely moves (~0.06–0.14); the game is converting
on-track into correct. Recall loss decomposition on v2b (66 gold pairs): 12% never
retrieved by any call (2 easy under raw-question vector search — agent query
formulation; 2 BM25-only; 2 need k>20; 2 unfindable at k=100), 12% retrieved but
ranked 6+; ceiling recall over the full merged ranking is 0.913.

Gotchas: `runs/baseline.jsonl` is the **validation** split (script default at the time)
— zero question overlap with the train runs; rescored for methodology consistency
(0.40 correct) but don't cross-compare. deepeval aborts a whole scoring pass if the
judge once emits invalid JSON — rerun `--rescore` on flake. TODO: V2B validation run
for a held-out baseline.

## Non-goals (for now)

Additional rerankers (local cross-encoder, LLM-based), alternative chunkers, query
rewriting, multi-turn eval, any frontend or deployment. The architecture allows all of
these; they're intentionally not built.

## Open questions

- Per-call retrieval metrics if the agent starts issuing many refined queries (the v2
  best-rank merge removed the worst cross-call distortion, but per-call quality is
  still invisible).
- Judge grounding is gold-articles-only by design: true facts pulled from *non-gold*
  retrieved articles still count as ungrounded. Deliberate (don't reward wrong-article
  content), but it penalizes some genuinely helpful cross-article answers.
- In sentinel cases (agent retrieved no gold article) an answer with extras can never
  score correct, even if it covers the full expected resolution.
- The two judge flags could collapse into one 3-class call (correct/on-track/wrong):
  removes the overlap inconsistency and halves judge cost.
- Statistical rigor at n=50 (bootstrap CIs, paired tests) — add when experiments
  disagree by small margins; reranker-vs-identity correctness (0.28 vs 0.22) is already
  inside the noise band.
