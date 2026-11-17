#!/bin/bash
# Skidanje adaptera sa LUMI scratcha: k-pro -> oracle -> LUMI, katran kroz stdout.
# Upotreba: fetch_adapter.sh <adapter-dir-na-scratchu> <lokalni-izlaz.tar.gz>
# Primjer: fetch_adapter.sh /scratch/project_465003647/bh-vision/smo2b-v1/out/adapter-smo2b-v1 /tmp/opencode/adapter-smo2b-v1.tar.gz
set -euo pipefail
REMOTE_DIR="${1:?upotreba: fetch_adapter.sh <remote-adapter-dir> <lokalni-tar.gz>}"
LOCAL_TGZ="${2:?upotreba: fetch_adapter.sh <remote-adapter-dir> <lokalni-tar.gz>}"
SSH="ssh -i ~/.ssh/agentmujo_pro -o IdentitiesOnly=yes -o BatchMode=yes ubuntu@140.238.222.253"
$SSH "~/lumi.sh 'tar czf - -C $(dirname "$REMOTE_DIR") $(basename "$REMOTE_DIR")'" > "$LOCAL_TGZ"
echo "skinuto: $LOCAL_TGZ ($(du -h "$LOCAL_TGZ" | cut -f1))"
tar tzf "$LOCAL_TGZ" | head -8
