# `run-agent` — sandboxed Claude Code → PR, on OpenShell

**Status: built and working (2026-07-21).** M1–M4 all done; the ✅ tables under [Milestones](#milestones) are measurements from real runs, not estimates. This file is the design record — rationale, what was measured, and what is still wrong. The usage docs are the "Sandboxed agent runs" section of the README.

Goal: one host command spins up an isolated [OpenShell](https://github.com/NVIDIA/openshell) sandbox with a **fully working, disposable dev environment** — its own Weaviate (KB pre-indexed), its own Phoenix, a budgeted LLM route — runs Claude Code headlessly against a fresh clone of this repo, and lets it open a PR on `jniechcial/customer-agent`. No Kubernetes, local Docker driver only.

```shell
scripts/run-agent --prompt "Change the chat.py UI so that agent answers are clearly labeled as 'agent>' and in another colour"
```

## Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Claude auth | `ANTHROPIC_API_KEY` via `provider create --type claude` | The supported path; Claude Code has full default-policy coverage. |
| Repo delivery | Fresh `git clone` from origin over HTTPS inside the sandbox | Reproducible; no `.env`, no `runs/`, no untracked junk leaks in. |
| PR mechanics | Claude does branch + commit + push + `gh pr create` itself | Matches the OpenShell GitHub tutorial; wrapper stays dumb and just reports whether a PR appeared. |
| Implementation | Bash, `scripts/run-agent` + payload dir | Mirrors OpenShell's own `examples/*/demo.sh`; zero deps. |
| **Execution model** | **Fire-and-forget: detached supervisor + `gc`, never a foreground trap** | The local stand-in for a Slack trigger — no caller holds the request open. `--no-keep` is client-side, so a dead launcher orphans the sandbox; cleanup must outlive the shell. |
| **Weaviate + Phoenix** | **Per-run sidecar containers on the gateway's bridge network** | Not Docker-in-Docker (needs privileged, negates OpenShell), not baked into the sandbox image (Phoenix UI would die with the sandbox). See below. |
| **KB seeding** | **Pre-seeded Weaviate image, tagged by collection name** | Sandbox starts with the index warm — no re-index, no per-run volume copy. |
| **Token budget** | **`claude --max-budget-usd` for the agent; a metered proxy sidecar + policy denying direct `api.openai.com` for the project's calls** | Claude Code has a native dollar cap (confirmed in M1). The project's own LLM calls need the proxy — prompt instructions and in-repo guards are advisory, since the agent edits code. |
| Compute driver | Docker (already running locally, 29.1.3) | macOS Homebrew gateway auto-detects it. |
| Secrets in sandbox | Providers only (env-var injection), never files | OpenShell purges them on sandbox delete. |

## How OpenShell gives us this

- `openshell sandbox create --from base --no-tty --no-keep -- <cmd>` runs `<cmd>` as the sandbox entrypoint non-interactively and **propagates its exit code**. That is the entire harness.
- `--upload <dir>:/sandbox` ships a payload directory in, preserving the basename → `./payload` lands at `/sandbox/payload/`.
- `--provider <name>` (repeatable) injects credentials as env vars at the network boundary. `--no-auto-providers` suppresses the interactive prompt — mandatory for an unattended run.
- `--env KEY=VALUE` (repeatable) sets sandbox-wide env vars. This is how the dev environment gets wired, with zero code change.
- The `base` image ships Claude Code, `gh`, `git`, `node`, Python, and `uv`.
- **The default policy allows GitHub reads only, and blocks all RFC1918 egress.** Both need a custom policy. That is the real work in this plan.
- `openshell logs <name> --tail` and `openshell term` show `l7_decision=deny` with the exact method + path — that is how we iterate on policy.

Prior art to copy from: `examples/agent-driven-policy-management/demo.sh` (host orchestration), `docs/get-started/tutorials/github-sandbox.mdx` (the `github_git`/`github_api` blocks), `examples/private-ip-routing/` (the `allowed_ips` mechanism).

## The per-run dev environment

### Why sidecars, not Docker-in-Docker or a baked image

Docker-in-sandbox is a non-starter: no Docker socket, non-root, landlock, and the gateway sets `enable_bind_mounts = false`. Running it would mean privileged containers, which negates the reason to use OpenShell at all.

The Docker driver puts every sandbox on a named bridge network (`network_name = "openshell-docker"` in the gateway TOML). It is a user-defined network, so **Docker's embedded DNS resolves container names on it**. `run-agent` starts `weaviate-<runid>`, `phoenix-<runid>` and `llm-budget-<runid>` on that network; the sandbox reaches them by name. Teardown is `docker rm -f` by the supervisor (see [Async by construction](#async-by-construction)).

Beats baking both into a custom sandbox image because: Phoenix stays reachable from the host browser on a published port (during *and* after the run — baked in, it dies with `--no-keep`); both images are already pulled locally so runs start in seconds with no build; and the sandbox image stays stock `base`, so the agent path and the environment path can be debugged independently.

Accepted cost: the network name is gateway-wide config, so concurrent runs share a network, though not containers. Run-ids keep names distinct.

### Seeding Weaviate

The host volume `customer-agent_weaviate_data` → `/var/lib/weaviate` already holds the full 10,081-chunk index. Bake it into an image once, in `scripts/build-weaviate-seed`:

```shell
docker stop customer-agent-weaviate-1                      # consistent LSM/WAL copy
docker run --rm -v customer-agent_weaviate_data:/src -v "$PWD/.seed":/dst alpine cp -a /src/. /dst/
docker build -t customer-agent/weaviate-kb:$COLLECTION .   # FROM weaviate:1.38.2 + COPY .seed /var/lib/weaviate
docker start customer-agent-weaviate-1
```

Tag with the collection name (`KB_chunk512o64_te3small`) so an index-config change shows up as a tag change and stale seeds are obvious. Rebuild only after a re-index. Weaviate's backup API is the clean alternative if stopping the local instance ever becomes annoying.

Phoenix needs no seed — a fresh, empty instance per run is exactly the point. Stock `arizephoenix/phoenix:latest`, no volume.

### Wiring (no code change)

`config.py` already exposes every knob as an env override:

```shell
--env WEAVIATE_HTTP_HOST=weaviate-<runid>
--env PHOENIX_ENDPOINT=http://phoenix-<runid>:6006/v1/traces
--env OPENAI_BASE_URL=http://llm-budget-<runid>:4000/v1
```

Weaviate ports (8080 HTTP, 50051 gRPC) keep their defaults.

### Policy block for the sidecars

The sandbox proxy blocks RFC1918 by default; `allowed_ips` is the documented escape hatch (loopback and link-local stay blocked regardless — which is why container names beat `host.docker.internal`). The subnet is discovered at render time with `docker network inspect openshell-docker -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}'` and substituted into the template:

```yaml
  dev_services:
    name: dev-services
    endpoints:
      - { port: 8080,  allowed_ips: ["__BRIDGE_CIDR__"] }   # weaviate REST  (observed: 172.20.0.0/16)
      - { port: 50051, allowed_ips: ["__BRIDGE_CIDR__"] }   # weaviate gRPC
      - { port: 6006,  allowed_ips: ["__BRIDGE_CIDR__"] }   # phoenix OTLP
      - { port: 4000,  allowed_ips: ["__BRIDGE_CIDR__"] }   # llm budget proxy
    binaries:
      - { path: "/sandbox/repo/.venv/bin/python*" }
      - { path: "/sandbox/.uv/python/**" }
```

### Token budget

Two different burns, capped differently:

| Burn | Cap | Enforcement |
|---|---|---|
| Claude Code's own session | `claude -p --max-budget-usd N` + wall-clock `timeout` on the entrypoint | **Native dollar cap** (confirmed present in 2.1.156, `--print` only). Left on the direct Anthropic route so a proxy bug can't brick a run mid-flight. |
| The project's eval/index calls | A `MAX_CALLS` cap in the `llm-budget` sidecar, and the policy above grants the venv Python **no route to `api.openai.com`** | Structural. The proxy is the only way out and lives outside the sandbox, so the agent cannot raise its own limit. `--unlimited-calls` turns the cap off for a run that genuinely needs a full eval. |

LiteLLM was the original plan; a ~180-line stdlib proxy replaced it, because LiteLLM's budget tracking wants a database and the enforcing layer here is the *policy*, not the proxy's feature set.

**The 429 is written for its reader.** A budget 429 is permanent — nothing replenishes — but that is the opposite of what `429` conventionally means, and by default the OpenAI/Anthropic SDKs retry it twice with backoff, so it reaches the agent looking like a transient rate limit. The refusal therefore carries `x-should-retry: false` (Stainless SDKs obey it: measured 0.27 s to fail instead of ~2 s of backoff), an `x-run-agent-*` header set naming the error, the counts and its permanence, and a body message that tells the agent what to *do* — stop, don't hunt for another route, and report the shortfall in the PR body. Same treatment for `403 wrong_run_token`.

**Budgets are per-run, not shared.** Every proxy listens on the gateway-wide bridge, so container-name resolution alone would let a concurrent run spend a neighbour's budget. Each run therefore generates a token at launch; it is what the sandbox presents as its OpenAI/Anthropic key, and the proxy rejects anything else with `403 wrong_run_token`. Verified with two proxies side by side: a neighbour's token scored `rejected: 2` on run A's tally **without consuming a single one of A's calls**, and the cap still fired at the boundary (`200 200 200 → 429`) while an `--unlimited-calls` proxy ran 5-for-5.

`run-agent --agent-budget 3.00 --max-calls 60` sets both. When the call cap is exhausted the proxy returns 429 and the eval fails loudly — the desired outcome, not a silent overspend.

## Files

All shipped in [PR #4](https://github.com/jniechcial/customer-agent/pull/4):

```
scripts/run-agent                      # host CLI: launcher, supervisor, status/logs/cancel/gc
scripts/build-weaviate-seed            # one-off: local volume → tagged seed image
scripts/run-agent-payload/
  entrypoint.sh                        # runs inside the sandbox
  policy.template.yaml                 # rendered per run with <org>/<repo> + bridge CIDR
  task.md.template                     # prompt scaffold
  llm-budget/proxy.py                  # metered egress proxy (stdlib only)
```

`runs/` was already gitignored, so per-run state under `runs/agent/<run-id>/` needed no change. This plan lives at `plans/OPENSHELL.md`; the user-facing docs are the "Sandboxed agent runs" section of the README.

Two gateway providers are created out of band and persist between runs: `run-agent-claude` (type `claude`, from `ANTHROPIC_API_KEY` in `.env`) and `run-agent-github` (type `github`, from `gh auth token`). `run-agent` does not create them — that is a gap, see below.

### `policy.template.yaml`

Default policy's `claude_code` and `pypi` blocks verbatim, drop `nvidia_inference`/`vscode`, add `dev_services` above, and replace the read-only GitHub blocks with two repo-scoped ones:

```yaml
  github_git:                     # Smart HTTP: clone, fetch, push
    name: github-git
    endpoints:
      - host: github.com
        port: 443
        protocol: rest
        enforcement: enforce
        rules:
          - allow: { method: GET,  path: "/__ORG__/__REPO__.git/info/refs*" }
          - allow: { method: POST, path: "/__ORG__/__REPO__.git/git-upload-pack" }
          - allow: { method: POST, path: "/__ORG__/__REPO__.git/git-receive-pack" }
    binaries:
      - { path: /usr/bin/git }

  github_api:                     # REST + GraphQL for `gh pr create`
    name: github-api
    endpoints:
      - host: api.github.com
        port: 443
        protocol: rest
        enforcement: enforce
        rules:
          - allow: { method: "*", path: "/repos/__ORG__/__REPO__/**" }
          - allow: { method: GET, path: "/user" }        # gh auth probe
      - host: api.github.com
        port: 443
        path: "/graphql"
        protocol: graphql
        enforcement: enforce
        rules:
          - allow: { operation_type: query }
          - allow: { operation_type: mutation, fields: [createPullRequest, addComment] }
        deny_rules:
          - operation_type: mutation
            fields: [deleteRepository, deleteRef, updateBranchProtectionRule]
    binaries:
      - { path: /usr/local/bin/claude }
      - { path: /usr/bin/gh }
      - { path: /usr/bin/git }
      - { path: /usr/bin/curl }
```

Blast radius is one repo: writes to any other repo are denied by the proxy even though the token carries full `repo` scope. That containment is the point.

### `entrypoint.sh` (inside the sandbox)

```
1. require_env ANTHROPIC_API_KEY GITHUB_TOKEN            # fail fast, never echo values
2. export GH_TOKEN="$GITHUB_TOKEN"
3. git config --global user.name/user.email "run-agent"
4. git clone https://x-access-token:${GITHUB_TOKEN}@github.com/<org>/<repo>.git /sandbox/repo
   cd /sandbox/repo && git checkout -b "$BRANCH" "origin/$BASE"
5. uv sync                                               # venv at /sandbox/repo/.venv (policy expects this path)
6. wait-for weaviate-<runid>:8080/v1/.well-known/ready   # bounded retry, fail fast if the sidecar is missing
7. timeout "$WALL_CLOCK" claude -p "$(cat /sandbox/payload/task.md)" \
       --dangerously-skip-permissions --max-budget-usd "$AGENT_BUDGET" \
       --output-format stream-json --verbose --model "${AGENT_MODEL:-opus}"
8. git -C /sandbox/repo log --oneline "origin/$BASE".. | head
9. gh pr view --json url -q .url 2>/dev/null | sed 's/^/PR_URL=/'   # machine-readable last line
```

Notes:

- Claude Code refuses `--dangerously-skip-permissions` as root. M1 confirmed the sandbox runs as `sandbox` (uid 998), and M2 confirmed that flag alone is enough — the session reported `permissionMode: bypassPermissions`.
- The clone URL carries the token, so the entrypoint immediately rewrites `origin` to a tokenless URL and moves auth into a `credential.helper` shim. Without that, the token sits in `.git/config` where the agent can read it and echo it into a PR body.
- Step 9 is what makes "did it open a PR" observable, since `--no-keep` deletes the sandbox on exit.
- Step 5 costs a cold `uv sync` per run. If it's slow, bake the venv into a custom sandbox image later — but only after the stock path works.

### `task.md.template`

Wraps `--prompt` in the non-interactive contract:

- Where the repo is (`/sandbox/repo`), which branch is checked out, what the base is.
- **What the environment provides**: a private Weaviate with the KB already indexed, a private Phoenix, a budgeted LLM route. So `run_eval.py --limit N` genuinely works — and the budget is finite, so run the smallest eval that supports the change and do not re-index the KB.
- Rules: never ask the user anything — assume, and record the assumption in the PR body. No files outside the repo. No force-push. One branch, one PR.
- Finish with `git add -A && git commit`, `git push -u origin <branch>`, `gh pr create` with a body quoting the original prompt verbatim plus assumptions and any eval numbers produced.

### `scripts/run-agent` (host)

```
Flags: --prompt <text>          (required)
       --repo <org/repo>        (default: parsed from `git remote get-url origin`)
       --base <branch>          (default: main)
       --branch <name>          (default: agent/<run-id>)
       --model <name>           (default: sonnet)
       --agent-budget <usd>     (default: 3.00 — passed to claude --max-budget-usd)
       --max-calls <n>          (default: 60 — hard cap in the llm-budget sidecar)
       --unlimited-calls        (turn that cap off for a run that needs a full eval)
       --timeout <duration>     (default: 45m — wall clock around the claude call)
       --cpu / --memory         (default: 2 / 4Gi)
       --no-services            (skip sidecars for pure-UI changes — faster, cheaper)
       --keep-phoenix           (leave the traces container up after the run)
       --keep                   (leave the sandbox up for post-mortem)
       --dry-run                (render policy + prompt, print where, stop)

Subcommands: `status [run-id]`, `logs <run-id> [-f]`, `cancel <run-id>`, `gc`.
```

### Async by construction

The CLI is **fire-and-forget**: it launches a run and hands the terminal back in ~1 second. This is the local stand-in for triggering a run from Slack — no caller holds a lock on the request, and nothing depends on the invoking shell staying alive.

That rules out the obvious cleanup design. A `trap EXIT` in the launching shell is useless when the shell returns immediately, and **`--no-keep` cannot cover for it: it is enforced by the OpenShell CLI client, not the gateway.** Measured directly — SIGKILL the client and the sandbox sits in `Ready` indefinitely, 50 s after its entrypoint had already finished. Kill the launcher and you orphan a sandbox *and* three containers.

So ownership moves to a detached supervisor, with a garbage collector behind it:

| Layer | Owns | Covers |
|---|---|---|
| Front door (`run-agent --prompt …`) | preflight, render, `docker run -d` the sidecars, spawn supervisor, print run id + Phoenix URL, exit | fast failure on misconfiguration (missing token, no seed image, gateway down) |
| Supervisor (detached, `nohup`) | waits for sidecar readiness, blocks on `sandbox create`, records result, then tears down sandbox + sidecars | normal completion, agent failure, `cancel`, SIGTERM |
| `gc` (automatic on every launch, or manual) | sweeps any run whose supervisor is no longer alive, plus containers whose run directory is gone | SIGKILL, crash, reboot, a launcher that died mid-setup |

Every teardown path is idempotent, so overlapping cleanups are harmless. Run state lives in `runs/agent/<run-id>/` (`status.json`, `run.log`, `policy.yaml`, `payload/`), which is what makes `status` and `gc` work across terminals and reboots. Sidecars carry `run-agent.managed=1` and `run-agent.runid=<id>` labels so the collector can identify its own garbage without guessing from names.

## Milestones

**M1 — install and smoke test. ✅ Done (2026-07-21).** OpenShell 0.0.86 installed via Homebrew; gateway running on `https://127.0.0.1:17670`, registered as `openshell`, status Connected. Two throwaway sandboxes (`--from base --no-tty --no-keep --no-auto-providers`) established:

| Finding | Value | Consequence |
|---|---|---|
| Sandbox user | `sandbox`, uid 998 — **not root** | `--dangerously-skip-permissions` is viable. |
| Base image toolchain | Claude Code 2.1.156, gh 2.93.0, git 2.43.0, Python 3.14.3, uv 0.10.8 | Nothing extra to install for the agent path. |
| Agent budget flag | **`--max-turns` does not exist**; `--max-budget-usd <amount>` does (`--print` only) | Better than planned — a native dollar cap replaces the turn heuristic. |
| Permission flags | `--dangerously-skip-permissions`, `--allow-dangerously-skip-permissions`, `--permission-mode bypassPermissions` | Which combination the headless path needs is an M2 question. |
| Bridge network | `openshell-docker`, subnet **172.20.0.0/16** | The concrete `allowed_ips` CIDR for `dev_services`. Still discovered at render time — do not hardcode. |
| Image pull | ~52 s cold, instant warm | Per-run cost is negligible after the first. |

**M2 — one hand-driven PR. ✅ Done (2026-07-21).** [PR #1](https://github.com/jniechcial/customer-agent/pull/1) opened by Claude Code from inside the sandbox, first attempt, **zero policy denials** — the derived policy was right without iteration, so the anticipated deny-diagnose-widen loop never happened.

| Finding | Value |
|---|---|
| Permission flags | `--dangerously-skip-permissions` **alone** yields `permissionMode: bypassPermissions`. The separate `--allow-dangerously-skip-permissions` is not needed. |
| Cost / latency | $0.117 and 30 s wall for a README edit on `--model sonnet` (6 turns, resolved to `claude-sonnet-4-6`). A stray `claude-haiku-4-5` call also appeared — same `api.anthropic.com` endpoint, so policy-invisible. |
| `--max-budget-usd` | Accepted without complaint. Not actually exercised — the run cost a tenth of the $1.00 cap. **Still unproven under pressure.** |
| Default `claude_code` block | Sufficient. No extra Anthropic hosts needed. |
| PR quality | Body quoted the request verbatim and listed three real assumptions, per the task template. Diff was correctly scoped to one file. |

**Containment verified by negative test** (a separate LLM-free sandbox on the same policy): `git ls-remote` on `jniechcial/customer-agent` **allowed**; `git ls-remote` on `NVIDIA/OpenShell` **denied**; `gh api repos/NVIDIA/OpenShell` **denied**; `curl https://example.com` **denied**. The single-repo blast radius is real, not just asserted.

**M3 — the dev environment. ✅ Done (2026-07-21).** Seed image, three sidecars, `dev_services` policy, and an agent run that used all of it: [PR #2](https://github.com/jniechcial/customer-agent/pull/2), again with zero policy denials.

| Piece | Result |
|---|---|
| Seed image | `customer-agent/weaviate-kb:KB_chunk512o64_te3small`, 436 MB, serves all 10,081 chunks with **no re-indexing**. Build takes ~30 s; host Weaviate was down for ~20 s of it and came back with data intact. |
| Sidecar reachability | Sandbox reaches `weaviate-m3:8080` and `phoenix-m3:6006` **by container name** over the `openshell-docker` bridge (`172.20.0.0/16`). An undeclared port on the *same host* returns `403 policy_denied` — the `allowed_ips` block is port-scoped, not host-scoped. |
| Budget cap | Verified under pressure with `MAX_CALLS=4`: calls 1–4 returned 200, call 5 returned **429 `budget_exhausted`**. Direct `https://api.openai.com` from the sandbox fails outright. The proxy holds the real keys; the sandbox's `.env` gets a placeholder. |
| Acceptance run | `scripts/search.py` built and **verified against the live KB with real queries** whose output is quoted in the PR body. $1.05, 43 turns, 7.4 min on sonnet. |
| Metering | 4 OpenAI calls, 47 input tokens, 0 denied against a cap of 60 — an eval-shaped task fits comfortably. |
| Tracing | The per-run Phoenix ended with a `search` project holding 6 spans / 6 traces, browsable on the host at `localhost:6008`. |

**The one real gotcha, found by the agent mid-run:** sandbox egress goes through an HTTP proxy, and any client whose HTTP layer ignores proxy environment variables cannot reach *anything* — including the sidecars, since even DNS for `weaviate-m3` resolves through it. The Weaviate Python client builds its httpx instance with `trust_env=False` by default, so it failed until the agent added `AdditionalConfig(trust_env=True)` and `skip_init_checks=True` to `indexer.py`. **Generalize this before adding any new SDK to a sandbox run.**

Deliberately not wired: `ANTHROPIC_BASE_URL` is *not* set sandbox-wide, because Claude Code reads it too and would route its own session through the meter. The eval judge therefore needs it exported explicitly; the policy denies the venv Python direct Anthropic access, so forgetting fails loudly rather than silently unmetered.

**M4 — the async CLI. ✅ Done (2026-07-21).** `scripts/run-agent` with the launcher / supervisor / gc split above. All four lifecycle paths exercised end to end:

| Path | Result |
|---|---|
| Launch | Front door returns in **1 s**; sidecars started, supervisor detached, Phoenix URL printed. |
| Completion | Background run finished in ~41 s and opened [PR #3](https://github.com/jniechcial/customer-agent/pull/3). Sandbox deleted and all three sidecars removed **with no terminal attached**. |
| `cancel` | Supervisor killed, sandbox and containers torn down, state recorded as `cancelled`. |
| `gc` | Tested by accident, which is the best kind: a real bug (an unbound `MAX_CALLS` crashed the launcher after the sidecars were already up) left orphaned containers. `gc` identified the run as `unregistered` with a dead supervisor and swept them. |

The `trust_env` lesson from M3 is now baked into the generated task briefing, so future agents are told up front that egress is proxied.

**M5 — budget isolation and refusal semantics. ✅ Done (2026-07-21).** Two gaps closed after M4:

| Change | Result |
|---|---|
| Per-run token on the metered proxy | Budgets were already counted per run, but the proxy was unauthenticated on a shared bridge, so any sandbox could spend a neighbour's calls by resolving `llm-budget-<other>:4000`. Each run now generates a token that the sandbox presents *as* its API key. Verified with two proxies side by side: a neighbour's token scored `rejected: 2` on run A's tally **without consuming one of A's calls**, while A's own cap still fired (`200 200 200 → 429`). The proxy refuses to start without a token. |
| `--unlimited-calls` | Cap off for runs that genuinely need a full eval (`MAX_CALLS=0`). Verified 5-for-5 against an unlimited proxy while the capped one refused. The task briefing swaps the hard-cap warning for an instruction to spend deliberately. |
| 429 written for its reader | See [Token budget](#token-budget) — `x-should-retry: false` plus `x-run-agent-*` headers. Measured 0.27 s to surface instead of ~2 s of SDK backoff. |

## Known limitations (accepted for the prototype)

- **Concurrency is shallow.** Sidecars are per-run, but the bridge network and provider names are gateway-wide. Two simultaneous runs work; ten would need namespacing. The budget half of this is fixed (M5), but Weaviate and Phoenix sidecars remain reachable across runs, so a neighbour could read another run's traces or query its KB. Both are read-mostly and carry no credentials. The proper fix is a per-run Docker network, which is awkward because the gateway owns the sandbox's network attachment.
- **`run-agent` does not bootstrap its providers.** `run-agent-claude` and `run-agent-github` must be created once by hand (see the README); the CLI checks for credentials but not for the providers themselves, so a fresh machine fails inside the sandbox rather than at preflight.
- **One repo per policy render.** Cross-repo work needs a wider policy; deliberately not built.
- **Containment is per-binary, and two of the binaries are general-purpose interpreters.** Every grant is a (host, path-rule, binary) triple, so a block is only as narrow as its `binaries:` list. The default policy's `pypi` block granted unrestricted `github.com`/`api.github.com` to the *project's* interpreter as well as to the installers — a hole in the repo-scoped `github_git`/`github_api` blocks, since `python -c` could then reach the API for any repository the token allows. Split into `pypi_installers` (uv/pip only, which need GitHub for managed CPython downloads) after review; no installer path is lost, as nothing here resolves from GitHub. **The equivalent hole on `/usr/bin/node` is left open**: the `claude_code` block grants it full `api.anthropic.com`, and Claude Code is a Node application, so removing it would brick every run. Hand-written JS therefore has an unmetered Anthropic route around the proxy's call cap. Closing it properly means routing Claude Code itself through the proxy — see the open question below.
- **`main` is protected on the server, not by the policy.** Git pushes carry ref updates in the request body, so `git-receive-pack` is allow-or-deny with no branch granularity: within the one repository the policy admits, "do not force-push, do not push to base" is a prompt rule (`task.md`) that only GitHub branch protection can actually enforce. Enabled on this repo out of band, like the two providers — and, like them, `run-agent` does not check for it at preflight.
- **The GitHub token is over-scoped relative to the policy.** `gh auth token` carries full `repo`; the L7 proxy — not the token — confines the agent to one repository. A fine-grained PAT scoped to `customer-agent` is the belt-and-braces version.
- **The seed image goes stale.** It captures the index at build time; re-indexing means rebuilding. The collection-name tag makes staleness visible but does not prevent it.
- **No PR review loop.** `run-agent` opens the PR and exits. Iterating on review comments would be a `--continue` mode against an existing branch — out of scope.
- **The budget caps spend, not damage.** A confused agent can still burn the full budget on a useless eval. `sonnet` is the default model for exactly this reason.

## Open questions for later

- ~~Does `claude -p` need hosts beyond the default `claude_code` block?~~ No — M2 ran clean on it.
- ~~Does the LLM call cap actually hold?~~ Yes — M3 verified the 429 at the boundary. **`--max-budget-usd` itself is still untested at its limit**, though it now matters less: the proxy is the enforcing layer, and the agent's own spend is observable per run ($0.12 and $1.05 so far).
- `uv sync` per run was not a bottleneck inside a 7.4-minute run, so the venv does not need baking into a custom image yet.
- **Voyage reranking is unmetered.** The `voyageai` SDK takes an explicit key and exposes no base-URL override, so it goes direct and the real `VOYAGE_API_KEY` lands in the sandbox. Cheap and non-LLM, but it is the one credential the proxy does not hold back.
- Should the metered proxy also count Voyage and expose a dollar figure rather than a call count? Call count was what was asked for and is what the eval loop actually spends.
- Should Claude Code's own traffic also route through the budget proxy? Structurally tidier (one budget for the whole run), but a proxy failure then kills the agent. Revisit once the proxy has proven stable.
- Is `--approval-mode auto` worth enabling, so the agent can *propose* policy widenings via `policy.local` instead of failing outright? Interesting follow-up; adds a host-side approval step. Skip for v1.
