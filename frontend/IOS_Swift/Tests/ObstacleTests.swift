import CoreGraphics
import Foundation

// MARK: - Close obstacles from LiDAR depth

func runObstacleTests() {
    runObstacleDetectorTests()
    runObstacleTrackerTests()
    runObstacleAnalyzerTests()
}

/// A box in the world around the user: meters ahead, to the side, and in height relative
/// to the phone (negative is below it).
private struct WorldBox {
    var ahead: ClosedRange<Double>
    var across: ClosedRange<Double>
    var height: ClosedRange<Double>
}

private struct Scene {
    var cameraHeight = 1.2
    /// Degrees the camera looks below the horizon.
    var pitchDown = 0.0
    var boxes: [WorldBox] = []

    var gravity: (x: Double, y: Double, z: Double) {
        let angle = pitchDown * .pi / 180
        return (0, -cos(angle), -sin(angle))
    }
}

/// 320x180 map with a 65° by 40° field of view, like the LiDAR stream.
private let sceneIntrinsics = DepthIntrinsics(fx: 250, fy: 250, cx: 160, cy: 90)

/// Renders the scene's depth (distance along the camera axis) for every pixel.
private func render(_ scene: Scene, intrinsics: DepthIntrinsics? = sceneIntrinsics) -> DepthMap {
    let width = 320
    let height = 180
    let down = scene.gravity
    let lookDown = -down.z
    let length = (1 - lookDown * lookDown).squareRoot()
    let forward = (x: -lookDown * down.x / length, y: -lookDown * down.y / length, z: (-1 - lookDown * down.z) / length)
    let side = (
        x: forward.y * down.z - forward.z * down.y,
        y: forward.z * down.x - forward.x * down.z,
        z: forward.x * down.y - forward.y * down.x
    )

    var meters = [Float](repeating: 0, count: width * height)
    for row in 0..<height {
        for column in 0..<width {
            let rayX = (Double(column) + 0.5 - sceneIntrinsics.cx) / sceneIntrinsics.fx
            let rayY = (Double(row) + 0.5 - sceneIntrinsics.cy) / sceneIntrinsics.fy
            // Per meter of depth: the device-frame direction of this pixel's ray.
            let ray = (x: -rayY, y: -rayX, z: -1.0)
            let ahead = ray.x * forward.x + ray.y * forward.y + ray.z * forward.z
            let across = ray.x * side.x + ray.y * side.y + ray.z * side.z
            let up = -(ray.x * down.x + ray.y * down.y + ray.z * down.z)

            var nearest = Double.infinity
            if up < 0 {
                nearest = scene.cameraHeight / -up
            }
            for box in scene.boxes {
                var enter = 0.0
                var exit = Double.infinity
                for (rate, range) in [(ahead, box.ahead), (across, box.across), (up, box.height)] {
                    if abs(rate) < 1e-9 {
                        if !range.contains(0) { exit = -1 }
                        continue
                    }
                    let a = range.lowerBound / rate
                    let b = range.upperBound / rate
                    enter = max(enter, min(a, b))
                    exit = min(exit, max(a, b))
                }
                if enter <= exit && enter > 0 {
                    nearest = min(nearest, enter)
                }
            }
            meters[row * width + column] = nearest.isFinite && nearest < 20 ? Float(nearest) : 0
        }
    }
    return DepthMap(width: width, height: height, meters: meters, intrinsics: intrinsics)
}

private func runObstacleDetectorTests() {
    func find(_ scene: Scene, intrinsics: DepthIntrinsics? = sceneIntrinsics) -> CloseObstacle? {
        CloseObstacleDetector.find(in: render(scene, intrinsics: intrinsics), gravity: scene.gravity)
    }
    func near(_ value: Double?, _ expected: Double) -> Bool {
        guard let value else { return false }
        return abs(value - expected) <= 0.05
    }

    // The ground alone is never an obstacle, however the phone is held.
    expect(find(Scene()) == nil, "obstacle: ground with the phone upright")
    expect(find(Scene(pitchDown: 30)) == nil, "obstacle: ground with the phone tilted 30° down")
    expect(find(Scene(cameraHeight: 0.9, pitchDown: 20)) == nil, "obstacle: ground with the phone held low")

    let wall = WorldBox(ahead: 1.0...1.3, across: -0.4...0.4, height: -1.2...0.5)
    let wallAhead = find(Scene(boxes: [wall]))
    expect(near(wallAhead?.distance, 1.0) && wallAhead?.isHeadHeight == false, "obstacle: wall 1 m ahead, got \(String(describing: wallAhead))")

    let person = WorldBox(ahead: 0.8...1.0, across: -0.2...0.2, height: -1.2...0.4)
    let personTilted = find(Scene(pitchDown: 25, boxes: [person]))
    expect(near(personTilted?.distance, 0.8), "obstacle: person 0.8 m ahead with the phone tilted, got \(String(describing: personTilted))")

    // A sign hanging at 1.5-1.9 m with nothing below it: what a white cane misses.
    let sign = WorldBox(ahead: 1.1...1.15, across: -0.3...0.3, height: 0.3...0.7)
    let signAhead = find(Scene(boxes: [sign]))
    expect(near(signAhead?.distance, 1.1) && signAhead?.isHeadHeight == true, "obstacle: sign at head height, got \(String(describing: signAhead))")

    expect(find(Scene(boxes: [WorldBox(ahead: 1.0...1.3, across: 0.6...1.0, height: -1.2...0.5)])) == nil, "obstacle: beside the path")
    expect(find(Scene(boxes: [WorldBox(ahead: 2.0...2.3, across: -0.4...0.4, height: -1.2...0.5)])) == nil, "obstacle: farther than 1.5 m")
    expect(find(Scene(boxes: [WorldBox(ahead: 1.0...1.3, across: -0.4...0.4, height: -1.2 ... -1.05)])) == nil, "obstacle: a low curb")
    expect(find(Scene(pitchDown: 70, boxes: [wall])) == nil, "obstacle: camera pointed at the ground")
    expect(find(Scene(boxes: [wall]), intrinsics: nil) == nil, "obstacle: no intrinsics")
}

private func runObstacleTrackerTests() {
    let tracker = CloseObstacleTracker()
    func step(_ time: Double, _ distance: Double?, headHeight: Bool = false) -> CloseObstacle? {
        tracker.update(distance.map { CloseObstacle(distance: $0, isHeadHeight: headHeight) }, now: time)
    }
    expect(step(0.0, 1.2) == nil, "obstacle tracker: first sighting waits")
    expect(step(0.2, 1.2)?.distance == 1.2, "obstacle tracker: second sighting is announced")
    expect(step(0.4, 1.1) == nil, "obstacle tracker: a little closer is not announced again")
    expect(step(0.6, 0.65)?.distance == 0.65, "obstacle tracker: half a meter closer is announced")
    expect(step(0.8, nil) == nil, "obstacle tracker: gone")
    expect(step(2.0, nil) == nil, "obstacle tracker: still gone")
    expect(step(2.2, 1.3) == nil, "obstacle tracker: a new obstacle waits for confirmation")
    expect(step(2.4, 1.3)?.distance == 1.3, "obstacle tracker: a new obstacle is announced")

    expect(
        CloseObstacleTracker.message(for: CloseObstacle(distance: 0.8, isHeadHeight: false)) == "멈춰 주세요. 바로 앞에 장애물이 있어요.",
        "obstacle message: within a meter"
    )
    expect(
        CloseObstacleTracker.message(for: CloseObstacle(distance: 1.4, isHeadHeight: true)) == "앞 머리 높이에 장애물이 있어요. 약 1.5미터 거리예요.",
        "obstacle message: head height"
    )
}

private func runObstacleAnalyzerTests() {
    let frame = CGSize(width: 1080, height: 1920)
    let scene = Scene(boxes: [WorldBox(ahead: 0.9...1.2, across: -0.3...0.3, height: -1.2...0.5)])
    let depth = render(scene)

    // Nothing detected: the warning comes from the depth alone, after two frames.
    let analyzer = SceneAnalyzer()
    let first = analyzer.analyze([], frameSize: frame, now: 1.0, depth: depth, gravity: scene.gravity)
    expect(first.voiceGuide.isEmpty && first.closeObstacle != nil, "analyzer obstacle: first frame shows but does not speak")
    let second = analyzer.analyze([], frameSize: frame, now: 1.2, depth: depth, gravity: scene.gravity)
    expect(second.voiceGuide == "멈춰 주세요. 바로 앞에 장애물이 있어요.", "analyzer obstacle: spoken on the second frame, got \(second.voiceGuide)")
    expect(second.voiceUrgency == .critical, "analyzer obstacle: within a meter is critical")

    // The detector sees the same thing: its guidance names it instead.
    let person = RawDetection(label: "person", confidence: 0.9, bbox: BoundingBox(x1: 340, y1: 500, x2: 740, y2: 1500))
    let named = SceneAnalyzer()
    var spoken: [String] = []
    // Later timestamps: guidance waits 1.5 s after time zero before its first sentence.
    for (index, time) in [100.0, 100.2, 100.4].enumerated() {
        let result = named.analyze([person], frameSize: frame, now: time, depth: depth, gravity: scene.gravity)
        expect(result.closeObstacle != nil, "analyzer obstacle: still shown with a detection (frame \(index + 1))")
        spoken.append(result.voiceGuide)
    }
    expect(!spoken.joined().contains("장애물이"), "analyzer obstacle: no generic warning when the detector names it: \(spoken)")
    expect(spoken.joined().contains("사람"), "analyzer obstacle: the detector's guidance speaks instead: \(spoken)")

    // Without gravity there is no obstacle check, exactly as before.
    let noGravity = SceneAnalyzer().analyze([], frameSize: frame, now: 1.0, depth: depth)
    expect(noGravity.closeObstacle == nil && noGravity.detections.isEmpty, "analyzer obstacle: needs gravity")
}
