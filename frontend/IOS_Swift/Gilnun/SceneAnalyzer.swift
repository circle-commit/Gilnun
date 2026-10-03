import CoreGraphics
import Foundation

nonisolated struct LiveSceneResult {
    /// Sentence to speak now; empty when nothing new is worth announcing.
    let voiceGuide: String
    /// Highest risk level among the announced objects, used to decide whether to interrupt speech.
    let voiceUrgency: RiskLevel
    /// Detections sorted by guidance priority (riskiest first).
    let detections: [SceneDetection]

    static let empty = LiveSceneResult(voiceGuide: "", voiceUrgency: .low, detections: [])
}

/// Turns detector output into prioritized detections and spoken guidance.
///
/// Port of the backend live pipeline (`vision_service` → `safety_service` →
/// `guidance_message_service`). Call it from one serial queue; it keeps state
/// across frames to detect approaching objects and to avoid repeating itself.
nonisolated final class SceneAnalyzer {
    static let koreanLabels: [String: String] = [
        "person": "사람",
        "car": "차량",
        "truck": "트럭",
        "bus": "버스",
        "bicycle": "자전거",
        "motorcycle": "오토바이",
        "scooter": "전동 킥보드",
        "wheelchair": "휠체어",
        "stroller": "유모차",
        "traffic_light": "신호등",
        "traffic_sign": "교통 표지판",
        "pole": "기둥",
        "bollard": "볼라드 / 차단봉",
        "bench": "벤치",
        "tree_trunk": "나무",
        "movable_signage": "입간판",
        "potted_plant": "화분",
        "parking_meter": "주차 정산기",
        "stop": "버스 정류장",
        "table": "테이블",
        "barricade": "바리케이드",
        "chair": "의자",
        "fire_hydrant": "소화전",
        "kiosk": "가판대",
        "carrier": "카트",
        "dog": "강아지",
        "traffic_light_controller": "신호 제어함",
        "power_controller": "전기 분전함",
    ]

    /// Boxes smaller than 1% of the frame are ignored.
    private static let minimumAreaRatio = 0.01

    private let approachTracker = ApproachTracker(minGrowthRatio: 1.28)
    private let eventTracker: GuidanceEventTracker

    init(eventTracker: GuidanceEventTracker = GuidanceEventTracker()) {
        self.eventTracker = eventTracker
    }

    func reset() {
        approachTracker.reset()
        eventTracker.reset()
    }

    /// - Parameters:
    ///   - rawDetections: Detector output in `frameSize` pixel coordinates.
    ///   - now: Monotonic timestamp in seconds.
    func analyze(_ rawDetections: [RawDetection], frameSize: CGSize, now: TimeInterval) -> LiveSceneResult {
        let frameWidth = Double(frameSize.width)
        let frameHeight = Double(frameSize.height)
        let frameArea = max(1, frameWidth * frameHeight)

        // The backend rounded these values for its JSON response before applying
        // thresholds; rounding the same way keeps iOS decisions identical to it.
        var detections = rawDetections.compactMap { raw -> SceneDetection? in
            let areaRatio = rounded(min(1, raw.bbox.area / frameArea), places: 6)
            guard areaRatio >= Self.minimumAreaRatio else { return nil }

            let bbox = BoundingBox(
                x1: rounded(raw.bbox.x1, places: 2),
                y1: rounded(raw.bbox.y1, places: 2),
                x2: rounded(raw.bbox.x2, places: 2),
                y2: rounded(raw.bbox.y2, places: 2)
            )
            return SceneDetection(
                label: raw.label,
                koreanLabel: Self.koreanLabels[raw.label] ?? raw.label,
                confidence: rounded(raw.confidence, places: 4),
                bbox: bbox,
                frameWidth: frameWidth,
                position: HorizontalPosition(centerX: raw.bbox.centerX, frameWidth: frameWidth),
                areaRatio: areaRatio,
                verticalRatio: rounded(max(0, min(1, raw.bbox.y2 / max(1, frameHeight))), places: 4)
            )
        }

        let alerts = approachTracker.update(detections.map { (label: $0.label, bbox: $0.bbox) }, now: now)
        for alert in alerts {
            detections[alert.detectionIndex].approaching = true
            detections[alert.detectionIndex].growthRatio = alert.growthRatio
        }

        let prioritized = GuidanceRules.enrichAndPrioritize(detections)
        guard !prioritized.isEmpty else { return .empty }

        let events = eventTracker.chooseEvents(prioritized, now: now, limit: 2)
        return LiveSceneResult(
            voiceGuide: events.map(\.message).joined(separator: " "),
            voiceUrgency: events.map(\.detection.riskLevel).max() ?? .low,
            detections: prioritized
        )
    }

    private func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10, Double(places))
        return (value * scale).rounded(.toNearestOrEven) / scale
    }
}
