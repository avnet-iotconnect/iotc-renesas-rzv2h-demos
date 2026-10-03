#!/bin/bash
# Copy this repo to the board over SSH. Run on your PC (Linux, macOS, WSL or
# Git Bash) from anywhere inside the repo — the RZ/V AI SDK image has no git,
# so the code is pushed rather than cloned:
#     deploy/push_to_board.sh <board-ip> [dest-dir]
# Sends tracked + new files and skips everything in .gitignore, so the
# board's own calibration JSON files and /IOTCONNECT credentials are never
# touched. Re-run it to update the board after a `git pull`.
set -e
BOARD="${1:?usage: deploy/push_to_board.sh <board-ip> [dest-dir]}"
DEST="${2:-/root/iotc-renesas-rzv2h-robotic-arm}"
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

SSH_OPTS="-o StrictHostKeyChecking=accept-new"
git ls-files -z --cached --others --exclude-standard \
    | tar --null -T - -cf - \
    | ssh $SSH_OPTS "root@$BOARD" "mkdir -p '$DEST' && tar -xf - -C '$DEST' && chmod +x '$DEST'/*.sh '$DEST'/deploy/*.sh"
echo "Pushed to root@$BOARD:$DEST"
echo "First time on this board? Run:  ssh root@$BOARD 'cd $DEST && bash setup_board.sh'"
