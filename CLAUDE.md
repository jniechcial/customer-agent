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
| First-stage search | Hybrid BM25+vector, α=0.5 (Weaviate native) | +5pts recall@5 over pure vector on identical train questions, answers unchanged. `search_mode=vector` env knob restores pure vector for A/B. |
| Tool output | Full articles (top-5 hits deduped to whole articles) | 512-token chunks were the dominant cause of missing-steps failures; expected answers are written from whole articles. `tool_output_granularity=chunks` env knob restores fragments for A/B. |
| Judge grounding | v3: everything the agent read (`seen_article_ids` texts); extras must be relevant to the question AND grounded | The previous gold∩retrieved grounding scored faithful relay of non-gold retrieved articles as hallucination — half the on-track bucket. Correctness numbers across judge conventions are NOT comparable. |
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
  reranker and search-mode ids are recorded per question in run artifacts. Tool output
  defaults to **full articles** (the top-`k_final` hits deduped to whole articles;
  `tool_output_granularity=chunks` restores fragments). `RetrievalResult` keeps ranked
  chunks AND deduped ranked `article_id`s (first-occurrence order) — the latter is what
  eval consumes — plus `seen_article_ids` (only what the tool output contained; judge
  grounding consumes this, per-call in artifacts). During eval, retrievals are recorded per-question so metrics see everything
  across multiple tool calls. Aggregation rule (v2): articles ordered by the **best rank
  achieved in any call**, ties by call order, deduped. (v1 concatenated calls in order,
  burying a later search's rank-1 hit behind the first call's whole ranking — it
  penalized agents for searching more; switching to v2 alone moved v2b recall@5
  0.793→0.823.) The merge is recomputed from per-call rankings at scoring time, so
  `--rescore` applies the current rule to old artifacts.
- **Agent**: one Agent, one tool; the model decides when/how often to search, capped
  at `max_searches` (2, config knob) calls per question (eval) / user turn (chat) — a
  UX decision (too many tool calls). Enforced twice: the prompt (V4 = V3 + hard
  grounding cap, direct answers, all-methods completeness, verbatim click-paths,
  two-article synthesis — from the on-track failure analysis of `tool-call-constraint`)
  tells the model, and the tool deterministically returns a "search budget
  exhausted" error on calls beyond the cap (no retrieval runs, nothing recorded, the
  agent loop survives and must answer). The budget is a contextvar set per question
  by the eval runner and per turn by chat.py; bare tool use outside those contexts is
  uncapped. `max_searches`, `prompt_version`, and `tool_output_granularity` are
  recorded in run artifacts (self-describing runs). The measured cost of the cap
  is in Status (`tool-call-constraint` vs `hybrid-train`).
- **Eval**: two phases, deliberately separated. Generation (async, bounded concurrency)
  persists per-question JSONL to `runs/<run_id>.jsonl` so runs can be re-scored via
  `--rescore` without re-paying generation. Scoring: GEval answer correctness vs the
  dataset answer (judge = Claude via deepeval custom model; criteria must tolerate style
  differences — dataset answers are procedural markdown), plus deterministic
  precision/recall/MRR/MAP@{5,10} at **article level**. The judge (v3) is **grounded
  in the full texts of every article the agent saw in tool output** — gold or not
  (union of per-call `seen_article_ids`; old artifacts fall back to the top-`k_final`
  prefix of each call's ranking, so their rescores are indicative only). Extras beyond
  the terse reference answer are fine only when BOTH relevant to the user's question
  AND supported by those texts; either failure (off-topic padding, ungrounded claim)
  makes the answer incorrect. Articles the agent never saw are withheld on purpose —
  matching them is parametric memory, not grounding. The partial flag is defined by
  missingness: right problem, accurate content, but expected material absent (never
  retrieved, or over-compressed). Judge v1 (no grounding) and v2 (gold∩retrieved
  grounding) scores live in `runs/judge-v1-backup/` / `runs/judge-v2-backup/`;
  correctness is NOT comparable across judge versions — v2 scored faithful relay of
  non-gold retrieved articles as hallucination.
  Correctness stays binary; the partial flag means "on the right track", and
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

M0–M5 done; full KB indexed (10,081 chunks).

**Active baseline: `full-articles-train`** (V4 prompt + full-article tool output +
judge v3), the first run clean end-to-end under the current setup, on the **full
100-question train split** (earlier runs used only the first 50):

| config | correct | on-track | wrong | recall@5 |
|---|---|---|---|---|
| V3 + chunks (`tool-call-constraint`, judge-v3 rescore◊, n=49) | 0.63 | 0.33 | 0.04 | 0.816 |
| V4 + chunks (`system-v4`, judge-v3 rescore◊, n=50) | 0.70 | 0.24 | 0.06 | 0.800 |
| **V4 + full articles (`full-articles-train`, n=100)** | **0.74** | 0.23 | **0.03** | 0.855 |

◊ Rescored chunk-run grounding uses the fallback (full texts of the top-`k_final`
prefix — generous: the agent only saw chunks); indicative only. On the shared first
50 questions full-articles ties system-v4 at 0.70 under honest grounding, with the 3
gains being exactly the predicted chunk-granularity fixes (content in gold articles
that never made it into a retrieved chunk) against 3 churn losses. Side effect of
full articles: the agent searches less (1.38→1.20 avg calls) and merged recall@5
*improves* (0.800→0.870 on the shared 50) — under the v2 best-rank merge a weak
second call dilutes the top-5, so fewer searches means a cleaner ranking. Cost ~flat
($0.054/answer).

The judge-v3 change (see decisions table) roughly halved the on-track bucket by no
longer scoring faithful relay of non-gold retrieved articles as hallucination:
`tool-call-constraint` 0.43→0.63, `system-v4` 0.40→0.70 on identical answers. It also
resolved the V4-prompt verdict: under judge v2 V4 looked like a regression (0.40 vs
0.43); under v3 it is +7pts over V3 — its completeness/synthesis behaviors were being
punished, not failing. Wrong stayed flat (0.04–0.06) — the new judge is not absolving
bad answers.

--- Historical ladder below: scored with **judge v2** (gold∩retrieved grounding,
backed up in `runs/judge-v2-backup/`); correctness columns NOT comparable to the
judge-v3 numbers above. First 50 train questions only; v2 best-rank merge rule;
V2A/V2C keep v1-merge retrieval numbers (marked †). Judge buckets shift a few points
on any rescore (LLM variance).

| config | correct | on-track | wrong | recall@5 | mrr@5 |
|---|---|---|---|---|---|
| V1 prompt + identity (`v1-identity-train`) | 0.28 | 0.60 | 0.12 | 0.743 | 0.563 |
| V1 + voyage (`voyage-reranker-train`) | 0.22 | 0.72 | 0.06 | 0.793 | 0.684 |
| V2A + voyage † | 0.39 | 0.49 | 0.12 | 0.769† | 0.688† |
| V2B + voyage | 0.46 | 0.46 | 0.08 | 0.823 | 0.709 |
| V2C + voyage † | 0.45 | 0.41 | 0.14 | 0.786† | 0.657† |
| V2B + voyage + hybrid (`hybrid-train`, n=47‡) | **0.51** | 0.38 | 0.11 | **0.862** | 0.695 |
| V3 (2-search cap) + voyage + hybrid (`tool-call-constraint`, n=49‡) | 0.43 | 0.51 | 0.06 | 0.816 | 0.698 |
| V3 + voyage + vector (`tool-call-constraint-just-vector`) | 0.40 | 0.50 | 0.10 | 0.833 | 0.703 |

‡ Transient OpenAI 520s during generation: `hybrid-train` is missing questions 27, 47,
49; `tool-call-constraint` is missing question 24. On the same 47 questions v2b-vector
scores 0.49 correct / 0.812 recall@5, so hybrid's retrieval gain there (+5pts recall@5,
+4.6pts recall@3) is real while the answer buckets are within judge noise.

**The 2-search cap is a deliberate UX-over-quality trade** (kept in all later
configs). Cost vs the uncapped `hybrid-train`: recall@5
0.862→0.816, correct 0.51→0.43 (buckets shifted toward on-track; wrong stayed low at
0.06). Avg tool calls 1.49, max 2 — the deterministic cap held. Notably, under the cap
hybrid's recall edge over pure vector disappears (0.816 vs 0.833 — if anything
reversed, within noise at n≈50): part of hybrid's earlier gain came through the agent's
extra searches, which the cap removes.

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
— zero question overlap with the train runs; don't cross-compare. deepeval aborts a
whole scoring pass if the judge once emits invalid JSON — rerun `--rescore` on flake.
The partial flag still overlaps correct answers (53 of 74 corrects on
`full-articles-train` also scored partial=1 — independent LLM calls, the partial judge
is stricter about missingness): read buckets, never the partial mean. TODO: validation
run of the active config (V4 + full articles + voyage + hybrid) for a held-out
baseline.

## Non-goals (for now)

Additional rerankers (local cross-encoder, LLM-based), alternative chunkers, query
rewriting, multi-turn eval, any frontend or deployment. The architecture allows all of
these; they're intentionally not built.

## Open questions

- Per-call retrieval metrics if the agent starts issuing many refined queries (the v2
  best-rank merge removed the worst cross-call distortion, but per-call quality is
  still invisible).
- Sentinel cases shrank but remain: when the agent retrieves *nothing*, an answer
  with extras can never score correct, even if it covers the full expected resolution.
  (The former gold-only-grounding question is resolved by judge v3.)
- The two judge flags could collapse into one 3-class call (correct/on-track/wrong):
  removes the overlap inconsistency and halves judge cost.
- Statistical rigor (bootstrap CIs, paired tests) — the baseline is n=100 now, but
  cross-run deltas of a few points are still inside judge noise (~10/49 questions
  flipped buckets between two runs of near-identical configs); add when experiments
  disagree by small margins.
