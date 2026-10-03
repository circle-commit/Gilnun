import Foundation

// Usage: GilnunTests <guidance_fixtures.json> <SidewalkDetector.mlmodelc> <detector_golden.json> <repo root>
// Run through run_tests.sh, which generates the fixtures and compiles the model.

var checkCount = 0
var failureCount = 0

func expect(_ condition: Bool, _ message: @autoclosure () -> String) {
    checkCount += 1
    guard !condition else { return }

    failureCount += 1
    if failureCount <= 40 {
        print("FAIL: \(message())")
    }
}

let arguments = CommandLine.arguments
guard arguments.count == 5 else {
    print("usage: GilnunTests <guidance_fixtures.json> <model.mlmodelc> <detector_golden.json> <repo root>")
    exit(2)
}

runGuidanceUnitTests()
runDepthTests()
runGuidanceParityTests(fixturesURL: URL(fileURLWithPath: arguments[1]))
runDetectorTests(
    modelURL: URL(fileURLWithPath: arguments[2]),
    goldenURL: URL(fileURLWithPath: arguments[3]),
    repoRoot: URL(fileURLWithPath: arguments[4])
)

print("\(checkCount) checks, \(failureCount) failures")
exit(failureCount == 0 ? 0 : 1)
