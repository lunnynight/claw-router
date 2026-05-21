#!/bin/bash
# Deploy claw-router to production server
set -e

TARGET="${DEPLOY_TARGET:-root@your-server-ip}"
REMOTE_DIR="${DEPLOY_DIR:-/opt/claw-router}"

echo "==> Syncing to $TARGET:$REMOTE_DIR"
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.env' --exclude '*.egg-info' \
    -e ssh "$(dirname "$(dirname "$0")")/" "$TARGET:$REMOTE_DIR/"

echo "==> Installing dependencies"
ssh "$TARGET" "cd $REMOTE_DIR && pip install -e . 2>&1 | tail -3"

echo "==> Copying .env if not exists"
scp -n "$(dirname "$(dirname "$0")")/.env" "$TARGET:$REMOTE_DIR/.env" 2>/dev/null || true

echo "==> Installing systemd service"
ssh "$TARGET" "cp $REMOTE_DIR/deploy/claw-router.service /etc/systemd/system/ && systemctl daemon-reload"

echo "==> Restarting service"
ssh "$TARGET" "systemctl restart claw-router && sleep 2 && systemctl status claw-router --no-pager"

echo "==> Done! Test: curl http://\$TARGET_IP:3456/health"
