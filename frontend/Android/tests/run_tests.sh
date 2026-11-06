#!/usr/bin/env bash
# Checks the Android on-device pipeline on the JVM (no Android SDK needed):
#   - live-guidance Kotlin port vs. the Python backend (fixtures from the iOS guidance_fixtures.py)
#   - ONNX detectors (float and int8) vs. PyTorch reference detections (the iOS detector_golden.json)
#   - ONNX braille block segmenter vs. PyTorch reference masks (fixtures/braille_golden.json)
#   - text reading rules (aiming hints, duplicate text, number safety)
# Requires JDK 17+, Gradle and python3 (standard library only).
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$TESTS_DIR/../../.." && pwd)"
IOS_TESTS="$REPO_ROOT/frontend/IOS_Swift/Tests"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

python3 "$IOS_TESTS/guidance_fixtures.py" "$WORK_DIR/guidance_fixtures.json"

cd "$TESTS_DIR"
gradle --quiet --max-workers 1 test \
    -Dgilnun.fixtures="$WORK_DIR/guidance_fixtures.json" \
    -Dgilnun.model="$REPO_ROOT/frontend/Android/app/src/main/assets/sidewalk_detector.onnx" \
    -Dgilnun.fastModel="$REPO_ROOT/frontend/Android/app/src/main/assets/sidewalk_detector_int8.onnx" \
    -Dgilnun.brailleModel="$REPO_ROOT/frontend/Android/app/src/main/assets/braille_block.onnx" \
    -Dgilnun.golden="$IOS_TESTS/detector_golden.json" \
    -Dgilnun.repoRoot="$REPO_ROOT" \
    "$@"
echo "Android tests passed."
