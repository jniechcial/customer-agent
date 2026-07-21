#!/usr/bin/env bash
# Runs INSIDE the OpenShell sandbox as the entrypoint command.
#
# Clones the target repo, puts it on a fresh branch, and hands the task to
# Claude Code headlessly. Claude does the commit/push/PR itself; this script
# only reports whether that actually happened, because --no-keep destroys the
# sandbox the moment we exit.
#
# Inputs arrive as sandbox env vars (--env on `openshell sandbox create`) and
# provider-injected credentials (ANTHROPIC_API_KEY, GITHUB_TOKEN).

set -uo pipefail

require_env() {
    local missing=0
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            echo "run-agent: missing required env: $name" >&2
            missing=1
        fi
    done
    [[ "$missing" == 0 ]] || exit 78   # EX_CONFIG
}

require_env ANTHROPIC_API_KEY GITHUB_TOKEN REPO BRANCH BASE

AGENT_MODEL="${AGENT_MODEL:-opus}"
AGENT_BUDGET="${AGENT_BUDGET:-3.00}"
WALL_CLOCK="${WALL_CLOCK:-45m}"
WORKDIR=/sandbox/repo

# `gh` reads GH_TOKEN; git uses the credential in the clone URL below.
export GH_TOKEN="$GITHUB_TOKEN"

git config --global user.name  "run-agent"
git config --global user.email "run-agent@localhost"
git config --global advice.detachedHead false

echo "run-agent: cloning ${REPO} (base ${BASE}) -> ${WORKDIR}"
git clone --quiet "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO}.git" "$WORKDIR" || {
    echo "run-agent: clone failed" >&2; exit 1
}
cd "$WORKDIR" || exit 1

# Keep the token out of .git/config so it cannot leak into a commit or the PR.
git remote set-url origin "https://github.com/${REPO}.git"
git config credential.helper '!f() { echo "username=x-access-token"; echo "password=${GITHUB_TOKEN}"; }; f'

git checkout --quiet -b "$BRANCH" "origin/${BASE}" || {
    echo "run-agent: could not branch ${BRANCH} from origin/${BASE}" >&2; exit 1
}

# When the run has a dev environment, point the project at the sidecars. The
# repo reads .env through pydantic-settings; the OpenAI SDK reads OPENAI_BASE_URL
# from the process environment (set sandbox-wide with --env), which is why the
# API key here is a placeholder — the real one lives in the llm-budget proxy.
if [[ -n "${WEAVIATE_HOST:-}" ]]; then
    echo "run-agent: writing .env for the dev environment"
    # RUN_TOKEN stands in for the API keys: the proxy swaps it for the real
    # credential, and it is what stops this run from spending another run's
    # budget through a neighbouring proxy on the shared bridge network.
    cat > "${WORKDIR}/.env" <<EOF
OPENAI_API_KEY=${RUN_TOKEN:-no-run-token}
ANTHROPIC_API_KEY=${RUN_TOKEN:-no-run-token}
VOYAGE_API_KEY=${VOYAGE_API_KEY:-}
WEAVIATE_HTTP_HOST=${WEAVIATE_HOST}
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_PORT=50051
PHOENIX_ENDPOINT=${PHOENIX_ENDPOINT:-}
EOF
    # .env is gitignored, but be explicit: it must never reach a commit.
    grep -qxF '.env' "${WORKDIR}/.gitignore" 2>/dev/null || echo '.env' >> "${WORKDIR}/.gitignore"
fi

echo "run-agent: starting claude (model=${AGENT_MODEL} budget=\$${AGENT_BUDGET} timeout=${WALL_CLOCK})"
timeout "$WALL_CLOCK" claude -p "$(cat /sandbox/payload/task.md)" \
    --dangerously-skip-permissions \
    --max-budget-usd "$AGENT_BUDGET" \
    --model "$AGENT_MODEL" \
    --output-format stream-json --verbose
claude_rc=$?
echo "run-agent: claude exited rc=${claude_rc}"

echo "run-agent: commits on ${BRANCH} vs origin/${BASE}:"
git log --oneline "origin/${BASE}..HEAD" | head -20

# Last line of the log is the machine-readable result the host greps for.
pr_url="$(gh pr view --json url -q .url 2>/dev/null)"
if [[ -n "$pr_url" ]]; then
    echo "PR_URL=${pr_url}"
else
    echo "PR_URL="
fi

exit "$claude_rc"
