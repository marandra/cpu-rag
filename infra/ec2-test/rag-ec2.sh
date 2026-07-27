#!/usr/bin/env bash
#
# Start / stop / inspect the cpu-rag v2 trial EC2.
#
# The instance id + region come from OpenTofu state (no hardcoded IDs — survives
# a destroy/recreate). Stopping kills the ~$0.53/h compute charge; only EBS+EIP
# (~$0.25/day) remain. The Elastic IP is stable, so `ssh rag` keeps working.
#
# Usage:
#   ./rag-ec2.sh start     # power on, wait until running, show IP + health
#   ./rag-ec2.sh stop      # power off (idle cost only)
#   ./rag-ec2.sh status    # current state + public IP
#   ./rag-ec2.sh ssh       # ssh into it (once running)
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="$(tofu -chdir="$DIR" output -raw region 2>/dev/null || echo eu-central-1)"
IID="$(tofu -chdir="$DIR" output -raw instance_id 2>/dev/null || true)"

if [[ -z "$IID" ]]; then
  echo "!! No instance_id in tofu state at $DIR — has the box been applied?" >&2
  exit 1
fi

state() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text
}
ip() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

case "${1:-status}" in
  start)
    echo "Starting $IID ..."
    aws ec2 start-instances --region "$REGION" --instance-ids "$IID" >/dev/null
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
    echo "Running. IP: $(ip)"
    echo "  ssh rag"
    echo "  health: curl http://$(ip):8001/health  |  :8002/health"
    echo "  (give the app ~1-2 min to warm the snapshots on a fresh boot)"
    ;;
  stop)
    echo "Stopping $IID ..."
    aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null
    aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$IID"
    echo "Stopped. Compute charge is now \$0 — only EBS+EIP (~\$0.25/day) remain."
    ;;
  status)
    echo "instance: $IID   region: $REGION"
    echo "state:    $(state)"
    echo "ip:       $(ip)"
    ;;
  ssh)
    [[ "$(state)" == "running" ]] || { echo "Not running — './rag-ec2.sh start' first." >&2; exit 1; }
    exec ssh rag
    ;;
  *)
    echo "Usage: $0 {start|stop|status|ssh}" >&2
    exit 2
    ;;
esac
