import Foundation

/// Axis-aligned box in upright frame pixel coordinates (origin top-left), stored as
/// corner points like the backend's `bbox_xyxy`.
nonisolated struct BoundingBox: Equatable {
    var x1: Double
    var y1: Double
    var x2: Double
    var y2: Double

    var area: Double { max(0, x2 - x1) * max(0, y2 - y1) }
    var centerX: Double { (x1 + x2) / 2 }

    func intersectionOverUnion(with other: BoundingBox) -> Double {
        let intersection = max(0, min(x2, other.x2) - max(x1, other.x1))
            * max(0, min(y2, other.y2) - max(y1, other.y1))
        let union = area + other.area - intersection
        return union > 0 ? intersection / union : 0
    }
}

/// One object found by the detector, before any guidance logic is applied.
nonisolated struct RawDetection {
    let label: String
    let confidence: Double
    let bbox: BoundingBox
}

nonisolated enum HorizontalPosition: String {
    case left
    case center
    case right

    /// Splits the frame into thirds, like the backend's `_position_from_bbox`.
    init(centerX: Double, frameWidth: Double) {
        if centerX < frameWidth / 3 {
            self = .left
        } else if centerX > frameWidth * 2 / 3 {
            self = .right
        } else {
            self = .center
        }
    }

    var koreanName: String {
        switch self {
        case .left: return "왼쪽"
        case .center: return "정면"
        case .right: return "오른쪽"
        }
    }
}

nonisolated enum RiskLevel: String, Comparable {
    case low
    case medium
    case high
    case critical

    init(score: Int) {
        if score >= 85 {
            self = .critical
        } else if score >= 68 {
            self = .high
        } else if score >= 48 {
            self = .medium
        } else {
            self = .low
        }
    }

    var rank: Int {
        switch self {
        case .low: return 0
        case .medium: return 1
        case .high: return 2
        case .critical: return 3
        }
    }

    static func < (lhs: RiskLevel, rhs: RiskLevel) -> Bool {
        lhs.rank < rhs.rank
    }
}

nonisolated enum DistanceLevel: String {
    case far
    case near
    case close
    case veryClose = "very_close"
}

/// A detection enriched with the safety metadata used for live guidance.
nonisolated struct SceneDetection {
    let label: String
    let koreanLabel: String
    let confidence: Double
    let bbox: BoundingBox
    let frameWidth: Double
    let position: HorizontalPosition
    let areaRatio: Double
    /// How low the box bottom sits in the frame (0...1); lower boxes are usually closer.
    let verticalRatio: Double
    /// Distance measured by the LiDAR camera (iPhone Pro models), nil when unavailable.
    var distanceMeters: Double?
    var approaching = false
    var growthRatio: Double?
    var frontDangerZone = false
    var riskScore = 0
    var riskLevel = RiskLevel.low
}
