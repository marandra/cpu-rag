#!/usr/bin/env bash
###############################################################################
# stop_pool.sh — stop a CPU-RAG Apptainer pool started by launch_pool.sh.
#
# launch_pool.sh already tears the pool down on Ctrl-C / scancel. Use this only
# when the launcher was detached (nohup/&) or you lost the foreground shell.
# Matches this user's apptainer processes for the RAG service and nginx LB.
###############################################################################
set -uo pipefail

STAGE_ROOT="${STAGE_ROOT:-/tmp/cpu-rag}"

echo "==> Stopping CPU-RAG pool (user: $USER)"
# Replicas: apptainer exec ... uvicorn app.main:app
pkill -u "$USER" -f 'apptainer.*app\.main:app' && echo "  replicas signalled" || echo "  no replicas found"
# nginx LB: identified by the generated conf path under STAGE_ROOT
pkill -u "$USER" -f "apptainer.*nginx -c ${STAGE_ROOT}" && echo "  nginx signalled" || echo "  no nginx found"

echo "==> Done. Verify with: pgrep -u $USER -af apptainer"
