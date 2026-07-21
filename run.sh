#!/usr/bin/env bash
#
# Launch one deployment profile of the shared codebase.
#
#   ./run.sh glucowise up -d       # start the diabetes stack
#   ./run.sh glucowise generate    # build its KV snapshots (one-shot job)
#   ./run.sh aiciblock up -d       # start the hemorroides + cirugía stack
#   ./run.sh aiciblock logs -f
#   ./run.sh glucowise down
#
# Everything after the profile name is passed straight to docker compose;
# `generate` is the one shorthand, for the one-shot snapshot job.
#
# Each profile gets its own snapshots subdir, container names and LB port,
# so both stacks can run side by side.
set -euo pipefail

cd "$(dirname "$0")"

known_profiles() {
  find profiles -name '*.env' -exec basename {} .env \; 2>/dev/null | sort | tr '\n' ' '
}

usage() {
  echo "usage: $0 <profile> <docker-compose args...>" >&2
  echo "known profiles: $(known_profiles)" >&2
  exit 1
}

PROFILE="${1:-}"
[[ -n "$PROFILE" ]] || usage
shift

ENV_FILE="profiles/${PROFILE}.env"
[[ -f "$ENV_FILE" ]] || { echo "unknown profile: $PROFILE" >&2; usage; }
[[ -f .env ]] || { echo "ERROR: .env missing — copy env.example and set RAG_API_KEY" >&2; exit 1; }
[[ $# -gt 0 ]] || usage

# The compose file mounts this; create it so docker doesn't make a root-owned one.
mkdir -p "snapshots/${PROFILE}"

if [[ "$1" == "generate" ]]; then
  shift
  exec docker compose --env-file "$ENV_FILE" -p "$PROFILE" \
    --profile generate run --rm rag-generate "$@"
fi

exec docker compose --env-file "$ENV_FILE" -p "$PROFILE" "$@"
