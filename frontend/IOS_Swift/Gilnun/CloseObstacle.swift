import Foundation

/// Pinhole intrinsics of a depth map, in its pixels (sensor orientation).
nonisolated struct DepthIntrinsics: Equatable {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
}

/// Something solid in the walking path found in LiDAR depth, whatever the detector
/// thinks it is. Catches what the detector misses, such as objects so close they fill
/// the frame, and things a white cane misses, such as signs or branches at head height.
nonisolated struct CloseObstacle: Equatable {
    /// Horizontal distance ahead to its nearest part, in meters.
    let distance: Double
    /// Nothing of it is lower than about a meter above the ground.
    let isHeadHeight: Bool
}

nonisolated enum CloseObstacleDetector {
    /// Half the width of the path checked, a little wider than a person's shoulders.
    static let pathHalfWidth = 0.35
    /// How far ahead to warn, about a second and a half of walking.
    static let range = 1.5
    /// Points closer than this are the user's own hand or arm.
    static let minimumDistance = 0.2
    /// Points this close to the ground are the ground, curbs or low steps.
    static let groundClearance = 0.2
    /// Points this far above the phone are above head height.
    static let maximumHeightAboveCamera = 0.6
    /// Phone height when the ground is not in view: held at chest height.
    static let defaultCameraHeight = 1.2
    /// An obstacle whose lowest point is this high above the ground hangs at head height.
    static let headHeightAboveGround = 1.0
    /// Share of the depth map an obstacle must cover, so a few noisy points do not count.
    static let minimumShare = 0.0025

    /// - Parameters:
    ///   - depth: LiDAR depth with its intrinsics.
    ///   - gravity: CoreMotion gravity in device coordinates (x right, y up, z out of the
    ///     screen in portrait), pointing down.
    static func find(in depth: DepthMap, gravity: (x: Double, y: Double, z: Double)) -> CloseObstacle? {
        guard let intrinsics = depth.intrinsics, intrinsics.fx > 0, intrinsics.fy > 0 else { return nil }
        let gravityLength = (gravity.x * gravity.x + gravity.y * gravity.y + gravity.z * gravity.z).squareRoot()
        guard gravityLength > 0.1 else { return nil }
        let down = (x: gravity.x / gravityLength, y: gravity.y / gravityLength, z: gravity.z / gravityLength)

        // The back camera looks along the device's -z axis.
        let lookDown = -down.z
        // Pointed steeply up or down (more than about 50°), the camera does not see the path ahead.
        guard abs(lookDown) < 0.77 else { return nil }
        let forwardLength = (1 - lookDown * lookDown).squareRoot()
        let forward = (x: -lookDown * down.x / forwardLength, y: -lookDown * down.y / forwardLength, z: (-1 - lookDown * down.z) / forwardLength)
        let side = (
            x: forward.y * down.z - forward.z * down.y,
            y: forward.z * down.x - forward.x * down.z,
            z: forward.x * down.y - forward.y * down.x
        )

        var pathHeights: [Double] = []
        var nearPoints: [(distance: Double, height: Double)] = []
        for row in 0..<depth.height {
            let rayY = (Double(row) + 0.5 - intrinsics.cy) / intrinsics.fy
            for column in 0..<depth.width {
                let z = Double(depth.meters[row * depth.width + column])
                guard z.isFinite, z > 0 else { continue }

                // Camera coordinates (x along the sensor's width, y down it, z forward) to
                // device coordinates: the sensor's x runs down the portrait screen and its y
                // runs from right to left.
                let rayX = (Double(column) + 0.5 - intrinsics.cx) / intrinsics.fx
                let point = (x: -rayY * z, y: -rayX * z, z: -z)
                let ahead = point.x * forward.x + point.y * forward.y + point.z * forward.z
                let across = point.x * side.x + point.y * side.y + point.z * side.z
                guard ahead > minimumDistance, abs(across) <= pathHalfWidth else { continue }

                let height = -(point.x * down.x + point.y * down.y + point.z * down.z)
                if ahead <= 4.0 {
                    pathHeights.append(height)
                }
                if ahead <= range {
                    nearPoints.append((ahead, height))
                }
            }
        }

        // The ground is the lowest surface in the path; without it in view, assume chest height.
        var cameraHeight = defaultCameraHeight
        if pathHeights.count >= 20 {
            pathHeights.sort()
            let lowest = pathHeights[pathHeights.count / 20]
            if lowest < -0.5 && lowest > -2.0 {
                cameraHeight = -lowest
            }
        }

        let groundTop = -cameraHeight + groundClearance
        let obstacle = nearPoints.filter { $0.height > groundTop && $0.height < maximumHeightAboveCamera }
        guard Double(obstacle.count) >= minimumShare * Double(depth.width * depth.height) else { return nil }

        // Low percentiles instead of minimums, so a few stray points do not decide.
        let distances = obstacle.map(\.distance).sorted()
        let heights = obstacle.map(\.height).sorted()
        let nearest = distances[distances.count / 20]
        let lowest = heights[heights.count / 20]
        return CloseObstacle(
            distance: (nearest * 100).rounded() / 100,
            isHeadHeight: lowest + cameraHeight > headHeightAboveGround
        )
    }
}

/// Decides when to speak about a close obstacle: once it has been seen in two frames in a
/// row, and again when it is half a meter closer than when it was last announced.
nonisolated final class CloseObstacleTracker {
    private var consecutiveFrames = 0
    private var lastSeenAt = -Double.infinity
    private var spokenDistance: Double?

    func reset() {
        consecutiveFrames = 0
        lastSeenAt = -.infinity
        spokenDistance = nil
    }

    /// Returns the obstacle to announce now, if any.
    func update(_ obstacle: CloseObstacle?, now: TimeInterval) -> CloseObstacle? {
        guard let obstacle else {
            // Gone for over a second: the next obstacle is a new one.
            if now - lastSeenAt > 1.0 {
                consecutiveFrames = 0
                spokenDistance = nil
            }
            return nil
        }

        consecutiveFrames = now - lastSeenAt <= 0.6 ? consecutiveFrames + 1 : 1
        lastSeenAt = now
        guard consecutiveFrames >= 2 else { return nil }

        if let spokenDistance, spokenDistance - obstacle.distance < 0.5 {
            return nil
        }
        spokenDistance = obstacle.distance
        return obstacle
    }

    static func message(for obstacle: CloseObstacle) -> String {
        let place = obstacle.isHeadHeight ? "머리 높이에" : "바로 앞에"
        if obstacle.distance < 1.0 {
            return "멈춰 주세요. \(place) 장애물이 있어요."
        }
        return "\(obstacle.isHeadHeight ? "앞 머리 높이에" : "앞에") 장애물이 있어요. \(GuidanceRules.distancePhrase(meters: obstacle.distance))"
    }
}
