# Sandbox Smoke Test — 2026-07-22

Agent branch: `agent/20260722-095123-29732`

---

## Step 1 — `.env` exists and is readable

**Result: PASS**

File found at `/sandbox/repo/.env`. Contents (secrets redacted):

```
OPENAI_API_KEY=<redacted>
ANTHROPIC_API_KEY=<redacted>
VOYAGE_API_KEY=<redacted>
WEAVIATE_HTTP_HOST=weaviate-20260722-095123-29732
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_PORT=50051
PHOENIX_ENDPOINT=http://phoenix-20260722-095123-29732:6006/v1/traces
```

---

## Step 2 — `uv run scripts/search.py "how do I add a member to my site"`

**Result: PASS**

Weaviate sidecar and embedding proxy both reachable. Hybrid BM25+vector search
returned 5 ranked articles as expected:

```
query      : 'how do I add a member to my site'
config     : collection=KB_chunk512o64_te3small, search_mode=hybrid, reranker=voyage, k=5
candidates : 20 chunks → 5 unique articles

[1] Site Members: Setting Contacts as Members  score=0.9102
[2] ADI: Adding and Setting Up Your Member's Area  score=0.8672
[3] Online Programs: Adding a Members Area  score=0.8438
[4] Site Members: Managing Your Member Roles  score=0.8008
[5] Site Members: Adding and Customizing the Members List Element  score=0.7930
```

There was also a benign `ResourceWarning: unclosed socket` at exit (urllib3
pool teardown), which is cosmetic and does not affect results.

---

## Step 3 — `export ANTHROPIC_BASE_URL=… && uv run scripts/run_eval.py --split train --limit 1`

**Result: FAIL — HuggingFace network blocked, dataset not cached**

Command run:
```
export ANTHROPIC_BASE_URL=http://llm-budget-20260722-095123-29732:4000/anthropic
uv run scripts/run_eval.py --split train --limit 1
```

The eval runner immediately tries to download `Wix/WixQA` from HuggingFace Hub
via `datasets.load_dataset(...)`. The sandbox network policy blocks egress to
external hosts. The dataset was not pre-cached locally (only a lock file was
present at `~/.cache/huggingface/datasets/Wix___wix_qa/`).

**Verbatim error (trimmed to relevant frames):**

```
File ".../customer_agent/data/wixqa.py", line 23, in _load
    return load_dataset(
        settings.dataset_name,   # "Wix/WixQA"
        ...
    )
  ...
httpcore.RemoteProtocolError: Server disconnected without sending a response.
```

Retry with `HF_DATASETS_OFFLINE=1` to confirm no cached data exists:

```
ConnectionError: Couldn't reach 'Wix/WixQA' on the Hub (OfflineModeIsEnabled)
```

The `ConnectionError` path (offline mode) confirms no usable on-disk cache is
present. This was tried twice — both runs produced the same failure.

**Root cause:** The sandbox environment provides a pre-indexed Weaviate but does
not include a pre-cached copy of the HuggingFace QA dataset (`Wix/WixQA`). The
eval pipeline cannot proceed without it, and external network access to
`huggingface.co` is blocked by sandbox policy.

**What was not tested:** The OpenAI agent path (GPT call via metered proxy) and
the Anthropic judge path (`ANTHROPIC_BASE_URL` proxy) could not be exercised
because the dataset load fails before any agent or judge call is made.

---

## Summary

| Step | Status | Notes |
|---|---|---|
| 1 — `.env` present | ✓ PASS | All expected keys present |
| 2 — Weaviate + embeddings | ✓ PASS | Hybrid search + Voyage rerank working |
| 3 — Agent eval (1 question) | ✗ FAIL | HuggingFace dataset not reachable or cached |
