#!/usr/bin/env bash
# Download the packaged LIBERO bowl-task dataset (task_2, 50 demos, JPG, ~0.4 GB)
# and extract it into ./data. This is all training needs — no simulator.
set -euo pipefail
TAG="${BIND_DATA_TAG:-v0.1}"
URL="https://github.com/cameronosmith/BIND/releases/download/${TAG}/bind_bowl_task2.tar.gz"
DEST="${1:-./data}"
mkdir -p "$DEST"
echo "Downloading $URL"
curl -L --fail -o /tmp/bind_bowl_task2.tar.gz "$URL"
echo "Extracting into $DEST"
tar xzf /tmp/bind_bowl_task2.tar.gz -C "$DEST"
rm -f /tmp/bind_bowl_task2.tar.gz
echo "Done -> $DEST/libero_spatial/task_2/  (train with: python train.py --cache_root $DEST --task_ids 2)"
