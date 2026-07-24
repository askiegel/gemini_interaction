#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cleanup_generated_test_files() {
    find . -maxdepth 1 -type f \( \
        -name 'test_*_state.json' -o \
        -name 'test_*_state.json.lock' -o \
        -name 'world_model_state.json.lock' \
    \) -delete
    find . -type d -name '__pycache__' -prune -exec rm -rf {} +
}

trap cleanup_generated_test_files EXIT
cleanup_generated_test_files

if [ -f .env ]; then
    echo "INFO: Local .env detected; verifying it is ignored."
    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git check-ignore -q .env || {
            echo "ERROR: .env is not ignored by Git."
            exit 1
        }
    fi
fi

if find . -type f \( -name '*.lock' -o -name '*backup*' -o -name '*:Zone.Identifier' \) | grep -q .; then
    echo "ERROR: Release contains lock, backup, or Zone.Identifier files before testing."
    find . -type f \( -name '*.lock' -o -name '*backup*' -o -name '*:Zone.Identifier' \)
    exit 1
fi

python3 -m compileall -q .

if command -v node >/dev/null 2>&1; then
    node --check voice_relay/operator_console.js
fi

python3 -m unittest discover -p 'test_*.py'

echo "PASS: Version 1.0 release verification completed."
