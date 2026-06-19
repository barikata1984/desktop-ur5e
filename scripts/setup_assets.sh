#!/usr/bin/env bash
# Create symlinks from assets/ to the MuJoCo Menagerie model directories
# needed by the scene XMLs (meshdir="../assets").
#
# The menagerie is at assets/mujoco_menagerie/ (git submodule).
# This script links mesh files from the UR5e and Robotiq 2F-85 models
# into assets/ so that MuJoCo can find them.
#
# Idempotent: safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MENAGERIE="assets/mujoco_menagerie"
MODELS=(universal_robots_ur5e robotiq_2f85)

if [ ! -d "$MENAGERIE" ]; then
  echo "Error: $MENAGERIE not found. Initialize the git submodule first:"
  echo "  git submodule update --init"
  exit 1
fi

# Link mesh files from each model's assets/ into the top-level assets/.
for d in "${MODELS[@]}"; do
  src="$MENAGERIE/$d/assets"
  if [ ! -d "$src" ]; then
    echo "Warning: $src not found, skipping"
    continue
  fi
  for f in "$src/"*; do
    ln -sf "$f" "assets/$(basename "$f")"
  done
done

echo "assets/ ready ($(find assets -maxdepth 1 -type l | wc -l) mesh symlinks)"
