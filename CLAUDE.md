# customer-agent

Experimentation platform for a customer-support RAG agent, benchmarked on
[Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA). Python-only, no frontend, no
productionization. Everything is traced; the retrieval pipeline is built to be swapped
out piece by piece.

## Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Benchmark data | `wixqa_expertwritten` (200 rows) | Human-grounded, multi-article synthesis. Other subsets available behind a CLI flag. |
| Out-of-scope extension | [`jniechcial/WixQA_Extended`](https://huggingface.co/datasets/jniechcial/WixQA_Extended) (40 rows, revision-pinned) | Hand-curated off-topic/competitor/abuse/injection/harmful/out-of-KB questions + gray over-refusal traps; scored by ScopeHandling, not the correctness ladder. `--extended` on run_eval appends them (`--limit 0 --extended` = extension rows only); `extended_dataset_name=""` falls back to `extended/wixqa_extended.jsonl`. |
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
  is in LOGBOOK.md (`tool-call-constraint` vs `hybrid-train`).
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

## Status (2026-07-13)

M0–M5 done; full KB indexed (10,081 chunks).

**Active baseline: `prompt-v6.2-train-extended`** — V6.2 prompt (V5 + scope
triage + unconditional disjoint closings) + full articles + Voyage + hybrid +
2-search cap, judge v3 + ScopeHandling v1, train `--extended` (100 standard +
20 out-of-scope): standard **0.83 correct / 0.15 on-track / 0.02 wrong**
(guarded: matches `prompt-v5-train` with symmetric flips), recall@5 0.825,
**scope 0.81**, $0.054/answer, 17 s avg latency.

**Per-run results live in [LOGBOOK.md](LOGBOOK.md)** — chronological, judge-v3
scores throughout, with 🟢/🔴 deltas vs each run's comparison run. **After every
experiment run, add a logbook entry** (1–2 bullets on what changed + a ladder row);
if the judge changed, back up old scores to `runs/judge-vN-backup/` and `--rescore`
first. Do not accumulate run tables or per-run analysis here.

Durable findings:

- The wrong bucket barely moves across all configs (0.01–0.08); the game is
  converting on-track into correct, and it is **answer-side**: in
  `full-articles-train`, 16 of 26 non-correct answers had every gold article in the
  tool output. Both prompt overhauls that attacked the answer layer (V4, V5) paid
  off; the retrieval-side win (k_final=8) did not.
- The reranker and hybrid search improve retrieval but not answer buckets (the agent
  compensates for weak rankings with extra searches). Caution on prompt history: the
  judge-v2 claim that V2B doubled V1's correctness did not survive the judge-v3
  rescore — most of that gap was v2 punishing V1's relay of non-gold articles (see
  the logbook's ◊ grounding caveat before comparing across eras).
- **The 2-search cap is a deliberate UX-over-quality trade** (kept everywhere since
  `tool-call-constraint`). It cost ~5 pts recall@5 vs uncapped `hybrid-train` and
  erased hybrid's recall edge over pure vector (part of that edge came through extra
  searches); it halved tokens/cost. The deterministic cap held (max 2 calls).
- Recall-loss decomposition on v2b (66 gold pairs): 12% never retrieved by any call
  (2 easy under raw-question vector search — agent query formulation; 2 BM25-only;
  2 need k>20; 2 unfindable at k=100), 12% retrieved but ranked 6+; ceiling recall
  over the full merged ranking is 0.913.
- **Scope handling is prompt-layer work** (WixQA_Extended, runs #12–#16): V5 →
  V6.2 moved scope 0.375 → 0.81 with retrieval untouched, and *cut* cost (out-of-
  scope questions skip searching). The coupled risk is over-refusal: a topic-shaped
  escalation trigger ("refunds, billing…") cost 5 standard-correct points by
  diverting doc-answerable policy questions (V6); wording the trigger as what the
  docs can *decide* recovered them (V6.1), and making the closing offers
  unconditional recovered the scope losses that caused (V6.2). Escalation/
  deflection closings must stay disjoint: Customer Care only for Wix
  account/money/responsibility matters, never for off-Wix deflections.

Gotchas: deepeval
aborts a whole scoring pass if the judge once emits invalid JSON — rerun `--rescore`
on flake. The partial flag overlaps correct answers (independent LLM calls, the
partial judge is stricter about missingness): read buckets, never the partial mean.

TODO: validation run of the active config (`validation --extended`) for a held-out
baseline (explicitly deferred on 2026-07-17). Open from Bet 3: (b) a nudge to
spend the second search on the uncovered part of multi-part/enumerative questions
(7 of 10 unseen-gold failures in `full-articles-train` used only 1 of 2 searches).

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
- ScopeHandling criteria v1.1 candidates (2 of V6.2's 3 scope misses are judge-,
  not agent-side): the scope judge sees no retrieval context, so grounded
  specifics in scope answers score as invented; and silently ignoring an
  injection while answering the legitimate part scores as a miss though it is
  arguably ideal. Also: the pelican `off_topic` trap reads as a legitimate
  Custom Element question (0/4 runs) — reword or reclassify the dataset item.
  And OpenAI's moderation API-blocks the phishing `harmful_use` question in
  most runs (`cyber_policy`) — that category's n is unstable by design.
- Statistical rigor (bootstrap CIs, paired tests) — the baseline is n=100 now, but
  cross-run deltas of a few points are still inside judge noise (~10/49 questions
  flipped buckets between two runs of near-identical configs); add when experiments
  disagree by small margins.
