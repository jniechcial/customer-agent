# Experiment logbook

Chronological log of every eval run. **Update this file after each experiment run**
(and rescore older runs first whenever the judge changes — back up the previous
scores to `runs/judge-vN-backup/`). CLAUDE.md keeps only the active baseline and
durable findings; per-run numbers live here.

## Conventions

- **Judge**: every correctness number below is **judge v3** (grounded in the full
  texts of everything the agent saw in tool output). Runs 2–8 predate judge v3 and
  were rescored on 2026-07-13; their v2 scores are backed up in
  `runs/judge-v2-backup/` (v1 in `runs/judge-v1-backup/`).
- **◊ = chunk-output run**: the agent saw 512-token chunks while the judge grounds
  in **full article texts** (for pre-v3 artifacts: the fallback top-`k_final` prefix
  of each call; for newer chunk runs: the recorded `seen_article_ids`) — generous,
  since the agent never read the full articles, and multiple calls/question widen
  what the judge counts extras against. ◊ numbers are comparable to each other but
  **inflated relative to the full-article runs** (#9–#11, honest seen-articles
  grounding). This is why the V1-era runs rescore so high: judge v2 had scored their
  relay of non-gold retrieved articles as hallucination.
- **Buckets**: correct = correctness 1 · on-track = correctness 0, partial 1 ·
  wrong = 0/0. The two flags are independent judge calls that overlap on correct
  answers — read buckets, never the partial mean.
- **Δ vs**: deltas compare each run to its *config predecessor* (the run it changed
  one thing against), named in the Δ-vs column — not always the previous row.
- 🟢 better · 🔴 worse · ⚪ within noise (<0.02 on rate metrics, <$0.005 cost,
  <2 s latency, <500 tokens). On-track and tool calls carry no color (residual
  bucket / UX metric). Cross-run deltas of a few points are inside judge noise
  (~10/49 questions flip buckets between near-identical runs).
- Tokens/cost/latency are generation-side (agent), averaged per question.

## Ladder

| # | run | date | n | Δ vs | correct | on-track | wrong | recall@5 | mrr@5 | calls | tok/q | $/ans | lat (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `baseline-train` ◊ | 07-13 | 50 | — | 0.66 | 0.28 | 0.06 | 0.743 | 0.543 | 1.54 | 5.7k | 0.039 | 16.6 |
| 2 | `voyage-reranker-train` ◊ | 07-12 | 50 | #1 | 0.74 🟢 +0.08 | 0.18 | 0.08 🔴 +0.02 | 0.793 🟢 +0.05 | 0.684 🟢 +0.14 | 2.32 | 12.6k 🔴 +6.9k | 0.056 🔴 +0.017 | 13.5 🟢 −3.1 |
| 3 | `system-prompt-v2a-train` ◊ | 07-12 | 49 | #2 | 0.69 🔴 −0.05 | 0.24 | 0.06 ⚪ | 0.840 🟢 +0.05 | 0.711 🟢 +0.03 | 2.84 | 18.3k 🔴 +5.7k | 0.080 🔴 +0.025 | 20.0 🔴 +6.5 |
| 4 | `system-prompt-v2b-train` ◊ | 07-12 | 50 | #2 | 0.70 🔴 −0.04 | 0.24 | 0.06 🟢 −0.02 | 0.823 🟢 +0.03 | 0.709 🟢 +0.03 | 3.12 | 23.0k 🔴 +10.4k | 0.092 🔴 +0.036 | 22.6 🔴 +9.1 |
| 5 | `system-prompt-v2c-train` ◊ | 07-12 | 49 | #2 | 0.67 🔴 −0.07 | 0.24 | 0.08 ⚪ | 0.840 🟢 +0.05 | 0.671 ⚪ | 2.67 | 17.3k 🔴 +4.7k | 0.080 🔴 +0.025 | 19.5 🔴 +6.0 |
| 6 | `hybrid-train` ◊ | 07-12 | 47 ‡ | #4 | 0.62 🔴 −0.08 | 0.32 | 0.06 ⚪ | 0.862 🟢 +0.04 | 0.695 ⚪ | 2.55 | 15.3k 🟢 −7.8k | 0.075 🟢 −0.017 | 38.0 🔴 +15.4 \* |
| 7 | `tool-call-constraint` ◊ | 07-12 | 49 ‡ | #6 | 0.63 ⚪ | 0.33 | 0.04 🟢 −0.02 | 0.816 🔴 −0.05 | 0.698 ⚪ | 1.49 | 6.9k 🟢 −8.3k | 0.051 🟢 −0.024 | 37.6 ⚪ \* |
| 8 | `system-v4` ◊ | 07-12 | 50 | #7 | 0.70 🟢 +0.07 | 0.24 | 0.06 ⚪ | 0.800 ⚪ | 0.731 🟢 +0.03 | 1.38 | 6.6k ⚪ | 0.050 ⚪ | 13.0 🟢 −24.6 \* |
| 9 | `full-articles-train` | 07-12 | 100 | #8 | 0.74 🟢 +0.04 | 0.23 | 0.03 🟢 −0.03 | 0.855 🟢 +0.06 | 0.700 🔴 −0.03 | 1.20 | 7.7k 🔴 +1.1k | 0.054 ⚪ | 24.2 🔴 +11.2 |
| 10 | **`prompt-v5-train`** ★ active | 07-13 | 100 | #9 | **0.83** 🟢 +0.09 | 0.14 | 0.03 ⚪ | 0.820 🔴 −0.04 | 0.686 ⚪ | 1.23 | 9.0k 🔴 +1.3k | 0.066 🔴 +0.012 | 18.0 🟢 −6.2 |
| 11 | `k8-articles-train` ✗ rejected | 07-13 | 100 | #10 | 0.81 🔴 −0.02 | 0.18 | 0.01 🟢 −0.02 | 0.830 ⚪ | 0.682 ⚪ | 1.14 | 10.7k 🔴 +1.7k | 0.074 🔴 +0.009 | 19.0 ⚪ |

‡ Transient OpenAI 520s during generation: `hybrid-train` is missing questions 27,
47, 49; `tool-call-constraint` is missing question 24.
\* Latency for runs 6–7 is inflated by those 520 retries; run 8's big latency "win"
is mostly their absence, not the prompt.

## Runs

### 1. `baseline-train` — 2026-07-13 (delta-chain root)
- The original agent shape on the first 50 **train** questions: V1 prompt, identity
  (no) reranker, pure vector search, 512-token chunk tool output, no search cap
  (run via env knobs `SEARCH_MODE=vector RERANKER=identity
  TOOL_OUTPUT_GRANULARITY=chunks MAX_SEARCHES=99` + prompt pointed at V1).
- Retrieval matches the era's control run `v1-identity-train` (judge-v2-scored)
  almost exactly — recall@5 0.743 in both, mrr@5 0.543 vs 0.563. But the agent
  searched much less than the era runs (1.54 avg calls vs 2.3–3.1), so #2's
  token/cost deltas partly reflect search-count variance, not the reranker itself.

### 2. `voyage-reranker-train` — 2026-07-12 09:48
- Added Voyage `rerank-2.5` as the reranking stage over the top-20 candidates
  (was identity passthrough).

### 3. `system-prompt-v2a-train` — 2026-07-12 10:02
- Prompt V2A: V1 plus targeted rules — literal click-paths, all methods, no extras.
- One of three V2 variants (#3–#5), all compared against #2.

### 4. `system-prompt-v2b-train` — 2026-07-12 10:08
- Prompt V2B: structural rewrite — the agent is a faithful messenger for one primary
  article. Won the three-way comparison under judge v2 and became the active prompt.
- Judge-v3 rescore epilogue: the V2 variants' big correctness wins over V1
  (0.46 vs 0.22 under judge v2) largely evaporate under v3 ◊ grounding — most of
  that gap was judge v2 punishing V1's relay of non-gold articles, not answer
  quality. Retrieval gains (recall@5 +0.03–0.05, from better agent queries) stand.

### 5. `system-prompt-v2c-train` — 2026-07-12 10:13
- Prompt V2C: V2B plus a pre-send self-check against hallucinated specifics.
  Lost to V2B; not adopted (the self-check idea returned in V5's verify pass).

### 6. `hybrid-train` — 2026-07-12 13:51
- First-stage search switched from pure vector to hybrid BM25+vector (α=0.5,
  Weaviate native fusion); V2B prompt unchanged.
- Retrieval win is real (+4–5 pts recall@5 over #4, confirmed on matched questions
  vs a vector control); answer buckets within judge noise. n=47 ‡, latency \*.

### 7. `tool-call-constraint` — 2026-07-12 14:30
- Capped the agent at 2 searches/question: prompt V3 (V2B + budget language) plus a
  deterministic "budget exhausted" tool error — a deliberate UX-over-quality trade.
- Cost of the cap vs #6: recall@5 −0.05 (hybrid's edge came partly through extra
  searches), tokens/cost nearly halved, avg calls 2.55→1.49. Kept in all later runs.
  (A vector-search control, `tool-call-constraint-just-vector`, sits outside this
  log: 0.40/0.833 under judge v2.)

### 8. `system-v4` — 2026-07-12 16:46
- Prompt V4: hard grounding cap, all-methods completeness, verbatim click-paths,
  two-article synthesis — from the failure analysis of #7's on-track bucket.
- Judge-history note: under judge v2 V4 looked like a regression (0.40 vs 0.43);
  the v3 rescore flipped the verdict to +0.07 — its completeness/synthesis
  behaviors were being punished as hallucination, not failing.

### 9. `full-articles-train` — 2026-07-12 17:24
- Tool output switched from 512-token chunks to **full articles** (top-`k_final`
  hits deduped to whole articles) — chunks were the dominant cause of missing-steps
  failures. Judge v3 (seen-articles grounding) landed in the same commit.
- First run on the full 100-question train split (earlier runs: first 50 only).
  Side effect: the agent searches less (1.38→1.20 calls) and the merged ranking gets
  cleaner. 16 of its 26 non-correct answers had every gold article in tool output —
  the remaining game is answer-side, not retrieval.

### 10. `prompt-v5-train` — 2026-07-13 15:06 ★ active baseline
- Prompt V5, from #9's failure analysis: full-fidelity relay (no "each briefly")
  + pre-send verify pass, ban on "the docs don't say X" meta-commentary,
  both-readings hedging for ambiguous questions + site-owner persona default.
- 14 of #9's 26 non-correct answers flipped to correct vs 5 losses; recall@5 dipped
  while correctness rose +0.09 — more evidence the game is answer-side.

### 11. `k8-articles-train` — 2026-07-13 15:28 ✗ rejected
- `k_final` 5→8: more full articles in tool output, retrieval pipeline unchanged.
- Rejected: seen-gold coverage rose exactly as predicted (0.772→0.874) but the
  answer layer gave it back via context dilution (compressed steps, tangent
  padding) — correct 0.83→0.81, cost +$0.009/answer. One real effect: wrong 3→1.
  Kept `k_final=5`; revisit only paired with prompt work against big-context
  compression.

## Not in this log

`baseline` / `voyage-reranker` ran on the **validation** split (script default at
the time) — zero question overlap with the train runs; don't cross-compare. Also
still on judge v2 (rescore on demand): `v1-identity-train` (the era's train-split
V1+identity control, superseded as chain root by #1) and
`tool-call-constraint-just-vector` (vector-search control for #7).
