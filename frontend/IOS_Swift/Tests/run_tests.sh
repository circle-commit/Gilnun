#!/usr/bin/env bash
# Checks the iOS on-device pipeline on macOS:
#   - live-guidance Swift port vs. the Python backend (fixtures from guidance_fixtures.py)
#   - Core ML detector vs. PyTorch reference detections (detector_golden.json)
# Requires Xcode command line tools and python3 (standard library only).
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$TESTS_DIR/../../.." && pwd)"
APP_DIR="$REPO_ROOT/frontend/IOS_Swift/Gilnun"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

python3 "$TESTS_DIR/guidance_fixtures.py" "$WORK_DIR/guidance_fixtures.json"
xcrun coremlcompiler compile "$APP_DIR/SidewalkDetector.mlpackage" "$WORK_DIR" > /dev/null

# Same isolation settings as the app target.
xcrun swiftc -swift-version 5 -default-isolation MainActor -O \
    "$APP_DIR/SceneDetection.swift" \
    "$APP_DIR/ApproachTracker.swift" \
    "$APP_DIR/GuidanceEngine.swift" \
    "$APP_DIR/SceneAnalyzer.swift" \
    "$APP_DIR/ObjectDetector.swift" \
    "$TESTS_DIR/GuidanceTests.swift" \
    "$TESTS_DIR/DetectorTests.swift" \
    "$TESTS_DIR/main.swift" \
    -o "$WORK_DIR/GilnunTests"

"$WORK_DIR/GilnunTests" \
    "$WORK_DIR/guidance_fixtures.json" \
    "$WORK_DIR/SidewalkDetector.mlmodelc" \
    "$TESTS_DIR/detector_golden.json" \
    "$REPO_ROOT"
