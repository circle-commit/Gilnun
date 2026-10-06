import Foundation

/// Risk scoring, distance estimation and Korean speech templates for live guidance.
///
/// Port of `backend/services/guidance_message_service.py`. Keep the two in sync;
/// `frontend/IOS_Swift/Tests/run_tests.sh` checks this file against the Python code.
nonisolated enum GuidanceRules {
    nonisolated enum ObjectGroup: String {
        case vehicle
        case mobility
        case person
        case obstacle
        case traffic
    }

    nonisolated struct DistanceThresholds {
        let veryCloseArea: Double
        let veryCloseVertical: Double
        let closeArea: Double
        let closeVertical: Double
        let nearArea: Double
        let nearVertical: Double
    }

    static let highestRiskLabels: Set<String> = ["car", "truck", "bus"]
    static let highRiskLabels: Set<String> = ["motorcycle", "scooter", "bicycle"]
    static let mediumHighRiskLabels: Set<String> = [
        "bollard", "pole", "movable_signage", "tree_trunk", "barricade", "fire_hydrant",
    ]
    static let mediumRiskLabels: Set<String> = ["person", "wheelchair", "stroller", "carrier", "dog"]
    static let lowerRiskLabels: Set<String> = ["bench", "potted_plant", "traffic_sign", "traffic_light"]
    static let immediateSpeechLabels = highestRiskLabels.union(["motorcycle", "scooter"])
    static let lowConfidenceSpeechLabels = highestRiskLabels.union(highRiskLabels).union(["wheelchair", "stroller"])
    static let noisySpeechLabels: Set<String> = ["pole", "tree_trunk", "traffic_sign", "traffic_light", "movable_signage"]
    /// Fixed objects without a risk set of their own. "stop" is a bus stop.
    static let otherObstacleLabels: Set<String> = [
        "parking_meter", "stop", "table", "chair", "kiosk", "traffic_light_controller", "power_controller",
    ]
    static let obstacleLabels = mediumHighRiskLabels.union(["bench", "potted_plant"]).union(otherObstacleLabels)
    static let mobilityLabels = highRiskLabels.union(["wheelchair", "stroller", "carrier", "dog"])
    static let guidanceLabels = highestRiskLabels
        .union(highRiskLabels)
        .union(mediumHighRiskLabels)
        .union(mediumRiskLabels)
        .union(lowerRiskLabels)
        .union(otherObstacleLabels)

    // Small ground obstacles can be close while their boxes are still small.
    static let groundObstacleDistanceLabels: Set<String> = [
        "bollard", "pole", "tree_trunk", "movable_signage", "bench",
        "potted_plant", "parking_meter", "table", "fire_hydrant",
    ]

    static let defaultDistanceThresholds = DistanceThresholds(
        veryCloseArea: 0.09, veryCloseVertical: 0.86,
        closeArea: 0.04, closeVertical: 0.68,
        nearArea: 0.018, nearVertical: 0.52
    )

    // Large objects occupy a lot of pixels even when they are not immediately near.
    static let vehicleDistanceThresholds = DistanceThresholds(
        veryCloseArea: 0.16, veryCloseVertical: 0.90,
        closeArea: 0.08, closeVertical: 0.76,
        nearArea: 0.035, nearVertical: 0.58
    )
    static let personDistanceThresholds = DistanceThresholds(
        veryCloseArea: 0.11, veryCloseVertical: 0.88,
        closeArea: 0.055, closeVertical: 0.72,
        nearArea: 0.025, nearVertical: 0.54
    )
    static let mobilityDistanceThresholds = DistanceThresholds(
        veryCloseArea: 0.08, veryCloseVertical: 0.84,
        closeArea: 0.035, closeVertical: 0.64,
        nearArea: 0.016, nearVertical: 0.48
    )
    static let groundObstacleDistanceThresholds = DistanceThresholds(
        veryCloseArea: 0.035, veryCloseVertical: 0.80,
        closeArea: 0.018, closeVertical: 0.60,
        nearArea: 0.01, nearVertical: 0.42
    )
    static let trafficDistanceThresholds = DistanceThresholds(
        veryCloseArea: 0.08, veryCloseVertical: 0.90,
        closeArea: 0.035, closeVertical: 0.76,
        nearArea: 0.016, nearVertical: 0.60
    )

    // MARK: Risk

    static func objectGroup(for label: String) -> ObjectGroup {
        if highestRiskLabels.contains(label) {
            return .vehicle
        }
        if mobilityLabels.contains(label) {
            return .mobility
        }
        if label == "person" {
            return .person
        }
        if obstacleLabels.contains(label) {
            return .obstacle
        }
        if label == "traffic_sign" || label == "traffic_light" {
            return .traffic
        }
        return .obstacle
    }

    static func distanceLevel(label: String, areaRatio: Double, verticalRatio: Double) -> DistanceLevel {
        let thresholds = distanceThresholds(for: label)
        let area = clamp01(areaRatio)
        let vertical = clamp01(verticalRatio)

        if area >= thresholds.veryCloseArea || vertical >= thresholds.veryCloseVertical {
            return .veryClose
        }
        if area >= thresholds.closeArea || vertical >= thresholds.closeVertical {
            return .close
        }
        if area >= thresholds.nearArea || vertical >= thresholds.nearVertical {
            return .near
        }
        return .far
    }

    /// Uses the LiDAR distance when there is one; otherwise estimates from the box like the backend.
    static func distanceLevel(of detection: SceneDetection) -> DistanceLevel {
        if let meters = detection.distanceMeters {
            return distanceLevel(meters: meters)
        }
        return distanceLevel(label: detection.label, areaRatio: detection.areaRatio, verticalRatio: detection.verticalRatio)
    }

    /// At walking speed (about 1.2 m/s) these are roughly 1, 2 and 4 seconds away.
    static func distanceLevel(meters: Double) -> DistanceLevel {
        if meters < 1.2 {
            return .veryClose
        }
        if meters < 2.5 {
            return .close
        }
        if meters < 5.0 {
            return .near
        }
        return .far
    }

    static func isFrontDangerZone(_ detection: SceneDetection) -> Bool {
        guard detection.position == .center else { return false }
        let distance = distanceLevel(of: detection)
        return distance == .close || distance == .veryClose
    }

    static func riskScore(for detection: SceneDetection) -> Int {
        let confidence = clamp01(detection.confidence)
        let areaRatio = clamp01(detection.areaRatio)
        let verticalRatio = clamp01(detection.verticalRatio)

        var score = baseRisk(for: detection.label)
        score += positionBonus(for: detection.position)
        score += min(24, Int(areaRatio * 200))
        score += min(14, Int(verticalRatio * 14))
        score += Int(confidence * 8)

        if detection.approaching {
            score += detection.position == .center ? 24 : 13
        }

        if isFrontDangerZone(detection) {
            score += 10
        }

        return max(0, min(100, score))
    }

    /// Attaches safety metadata and sorts by live-guidance priority (riskiest first).
    static func enrichAndPrioritize(_ detections: [SceneDetection]) -> [SceneDetection] {
        let scored = detections.map { detection -> SceneDetection in
            var enriched = detection
            enriched.frontDangerZone = isFrontDangerZone(detection)
            enriched.riskScore = riskScore(for: detection)
            enriched.riskLevel = RiskLevel(score: enriched.riskScore)
            return enriched
        }

        return stableSorted(scored) { lhs, rhs in
            if lhs.riskScore != rhs.riskScore {
                return lhs.riskScore > rhs.riskScore
            }
            if (lhs.position == .center) != (rhs.position == .center) {
                return lhs.position == .center
            }
            if lhs.areaRatio != rhs.areaRatio {
                return lhs.areaRatio > rhs.areaRatio
            }
            return lhs.confidence > rhs.confidence
        }
    }

    // MARK: Speech

    nonisolated struct MessageContext {
        let label: String
        let koreanLabel: String
        let particle: String
        let position: HorizontalPosition
        let riskLevel: RiskLevel
        let objectGroup: ObjectGroup
        let eventType: GuidanceEventType
        let approaching: Bool
        let newlyDetected: Bool
        let distanceLevel: DistanceLevel
        let frontWord: String
    }

    nonisolated enum TemplateCondition {
        case lowLeft
        case lowRight
        case mediumVisible
        case frontObstacle
        case frontObstacleNamed
        case centerApproaching
        case criticalFront
        case sideApproaching
        case sidePerson
        case frontVehicle
        case frontHighRisk
        case highRisk
        case highNear
        case anyGuidance

        func matches(_ context: MessageContext) -> Bool {
            let isCenter = context.position == .center
            let isNear = context.distanceLevel == .close || context.distanceLevel == .veryClose

            switch self {
            case .lowLeft:
                return !context.approaching && context.position == .left && context.riskLevel == .low
            case .lowRight:
                return !context.approaching && context.position == .right && context.riskLevel == .low
            case .mediumVisible:
                return !context.approaching && context.riskLevel == .medium && context.distanceLevel != .veryClose
            case .frontObstacle:
                return isCenter && context.objectGroup == .obstacle && context.riskLevel >= .medium
            case .frontObstacleNamed:
                return !context.approaching && isCenter && context.objectGroup == .obstacle && context.riskLevel >= .high
            case .centerApproaching:
                return context.approaching && isCenter
            case .criticalFront:
                return context.riskLevel == .critical && isCenter && isNear
            case .sideApproaching:
                return context.approaching && !isCenter
            case .sidePerson:
                return context.objectGroup == .person && !isCenter
            case .frontVehicle:
                return !context.approaching && context.objectGroup == .vehicle && isCenter && context.riskLevel != .critical
            case .frontHighRisk:
                return !context.approaching && isCenter && context.riskLevel >= .high
            case .highRisk:
                return !context.approaching && context.riskLevel == .high
            case .highNear:
                return !context.approaching && context.riskLevel == .high && isNear
            case .anyGuidance:
                return !context.approaching && context.riskLevel <= .medium
            }
        }
    }

    nonisolated struct MessageTemplate {
        let text: String
        let baseWeight: Int
        let condition: TemplateCondition
        let tags: Set<String>
    }

    static let messageTemplates: [MessageTemplate] = [
        MessageTemplate(text: "왼쪽에 {korean_label}{particle} 있어요.", baseWeight: 18, condition: .lowLeft, tags: ["low", "side", "natural"]),
        MessageTemplate(text: "오른쪽에 {korean_label}{particle} 있어요.", baseWeight: 18, condition: .lowRight, tags: ["low", "side", "natural"]),
        MessageTemplate(text: "{position_ko}에 {korean_label}{particle} 보여요.", baseWeight: 13, condition: .mediumVisible, tags: ["medium", "new"]),
        MessageTemplate(text: "{position_ko}에 {korean_label}{particle} 보여요. 조심해서 지나가 주세요.", baseWeight: 12, condition: .mediumVisible, tags: ["medium", "action"]),
        MessageTemplate(text: "{front_word}에 {korean_label}{particle} 있어요. 천천히 이동해 주세요.", baseWeight: 26, condition: .frontObstacle, tags: ["obstacle", "front", "action", "medium"]),
        MessageTemplate(text: "{front_word}에 장애물이 있어요. 천천히 이동해 주세요.", baseWeight: 24, condition: .frontObstacle, tags: ["obstacle", "front", "action", "medium"]),
        MessageTemplate(text: "정면에 {korean_label}{particle} 있어요. 조심해 주세요.", baseWeight: 24, condition: .frontObstacleNamed, tags: ["obstacle", "front", "urgent"]),
        MessageTemplate(text: "정면 {korean_label}{particle} 가까워지고 있어요. 잠시 멈춰 주세요.", baseWeight: 34, condition: .centerApproaching, tags: ["approaching", "front", "urgent", "action"]),
        MessageTemplate(text: "정면의 {korean_label}{particle} 가까워지고 있어요. 주의해 주세요.", baseWeight: 22, condition: .centerApproaching, tags: ["approaching", "front", "urgent"]),
        MessageTemplate(text: "주의해 주세요. {korean_label}{particle} 바로 앞에 있어요.", baseWeight: 34, condition: .criticalFront, tags: ["critical", "front", "urgent", "action"]),
        MessageTemplate(text: "잠시 멈춰 주세요. 정면에 {korean_label}{particle} 가까워요.", baseWeight: 30, condition: .criticalFront, tags: ["critical", "front", "urgent", "action"]),
        MessageTemplate(text: "{position_from_ko} {korean_label}{particle} 다가오고 있어요.", baseWeight: 18, condition: .sideApproaching, tags: ["approaching", "side", "natural"]),
        MessageTemplate(text: "{position_ko}의 {korean_label}{particle} 가까워지고 있어요.", baseWeight: 18, condition: .sideApproaching, tags: ["approaching", "side"]),
        MessageTemplate(text: "{position_from_ko} {korean_label}{particle} 지나가고 있어요.", baseWeight: 20, condition: .sidePerson, tags: ["person", "side", "natural"]),
        MessageTemplate(text: "{position_ko}에 사람이 있어요. 살짝 주의해 주세요.", baseWeight: 11, condition: .sidePerson, tags: ["person", "side", "medium"]),
        MessageTemplate(text: "{position_ko}에 {korean_label}{particle} 보여요.", baseWeight: 9, condition: .sidePerson, tags: ["person", "side", "new"]),
        MessageTemplate(text: "{front_word}에 {korean_label}{particle} 보여요. 주의해 주세요.", baseWeight: 24, condition: .frontVehicle, tags: ["vehicle", "front", "new", "urgent"]),
        MessageTemplate(text: "정면에 {korean_label}{particle} 가까워요. 조심해 주세요.", baseWeight: 22, condition: .frontHighRisk, tags: ["front", "urgent", "near"]),
        MessageTemplate(text: "{position_ko}에 {korean_label}{particle} 있어요. 주의해 주세요.", baseWeight: 13, condition: .highRisk, tags: ["urgent"]),
        MessageTemplate(text: "{position_ko}에 {korean_label}{particle} 가까이 있어요.", baseWeight: 14, condition: .highNear, tags: ["urgent", "near", "natural"]),
        MessageTemplate(text: "{position_ko}에 {korean_label}{particle} 있어요.", baseWeight: 8, condition: .anyGuidance, tags: ["natural"]),
        MessageTemplate(text: "{position_ko}에 {korean_label}{particle} 보여요.", baseWeight: 7, condition: .anyGuidance, tags: ["new", "natural"]),
    ]

    static let fallbackTemplate = "{position_ko}에 {korean_label}{particle} 있어요."

    static func messageContext(for detection: SceneDetection, eventType: GuidanceEventType) -> MessageContext {
        let koreanLabel = detection.koreanLabel.isEmpty ? "장애물" : detection.koreanLabel
        return MessageContext(
            label: detection.label,
            koreanLabel: koreanLabel,
            particle: particle(for: koreanLabel),
            position: detection.position,
            riskLevel: detection.riskLevel,
            objectGroup: objectGroup(for: detection.label),
            eventType: eventType,
            approaching: detection.approaching || eventType == .approaching,
            newlyDetected: eventType == .newObject,
            distanceLevel: distanceLevel(of: detection),
            frontWord: highestRiskLabels.contains(detection.label) ? "앞쪽" : "앞"
        )
    }

    /// Templates that fit the situation, with their selection weights.
    static func weightedTemplates(for context: MessageContext) -> [(text: String, weight: Int)] {
        let weighted = messageTemplates
            .filter { $0.condition.matches(context) }
            .map { (text: $0.text, weight: templateWeight($0.baseWeight, context: context, tags: $0.tags)) }
        return weighted.isEmpty ? [(text: fallbackTemplate, weight: 1)] : weighted
    }

    /// Picks a weighted template so repeated guidance does not sound robotic.
    /// `random` returns a uniform value in `0...upperBound`.
    static func detectionMessage(
        for detection: SceneDetection,
        eventType: GuidanceEventType,
        random: (Double) -> Double
    ) -> String {
        let context = messageContext(for: detection, eventType: eventType)
        let template = chooseWeightedTemplate(weightedTemplates(for: context), random: random)
        let message = format(template, with: context)
        guard let meters = detection.distanceMeters else { return message }
        return "\(message) \(distancePhrase(meters: meters))"
    }

    /// Spoken LiDAR distance: half meters up close, whole meters farther away.
    static func distancePhrase(meters: Double) -> String {
        if meters < 1.0 {
            return "1미터 안이에요."
        }
        let rounded = meters < 3.0 ? (meters * 2).rounded() / 2 : meters.rounded()
        let number = rounded == rounded.rounded() ? String(Int(rounded)) : String(format: "%.1f", rounded)
        return "약 \(number)미터 거리예요."
    }

    static func particle(for label: String) -> String {
        guard let last = label.unicodeScalars.last else { return "이" }

        let code = Int(last.value)
        if (0xAC00...0xD7A3).contains(code) {
            return (code - 0xAC00) % 28 != 0 ? "이" : "가"
        }
        return "이"
    }

    // MARK: Helpers

    private static func distanceThresholds(for label: String) -> DistanceThresholds {
        if groundObstacleDistanceLabels.contains(label) {
            return groundObstacleDistanceThresholds
        }

        switch objectGroup(for: label) {
        case .vehicle: return vehicleDistanceThresholds
        case .person: return personDistanceThresholds
        case .mobility: return mobilityDistanceThresholds
        case .traffic: return trafficDistanceThresholds
        case .obstacle: return defaultDistanceThresholds
        }
    }

    private static func baseRisk(for label: String) -> Int {
        if highestRiskLabels.contains(label) { return 70 }
        if highRiskLabels.contains(label) { return 58 }
        if mediumHighRiskLabels.contains(label) { return 48 }
        if mediumRiskLabels.contains(label) { return 36 }
        if lowerRiskLabels.contains(label) { return 22 }
        return 28
    }

    private static func positionBonus(for position: HorizontalPosition) -> Int {
        switch position {
        case .center: return 15
        case .right: return 6
        case .left: return 5
        }
    }

    private static func templateWeight(_ baseWeight: Int, context: MessageContext, tags: Set<String>) -> Int {
        var weight = baseWeight
        let isUrgentOrAction = tags.contains("urgent") || tags.contains("action")

        switch context.eventType {
        case .approaching:
            weight += tags.contains("approaching") ? 24 : -4
        case .riskIncreased, .closer, .enteredFrontZone:
            weight += isUrgentOrAction ? 18 : 2
        case .newObject:
            weight += tags.contains("new") ? 12 : 3
        }

        switch context.riskLevel {
        case .critical:
            weight += tags.contains("critical") ? 34 : isUrgentOrAction ? 18 : -8
        case .high:
            weight += isUrgentOrAction ? 16 : 0
        case .medium:
            weight += tags.contains("medium") || tags.contains("action") ? 12 : 2
        case .low:
            weight += tags.contains("low") || tags.contains("natural") ? 16 : -6
        }

        switch context.position {
        case .center:
            weight += tags.contains("front") || tags.contains("action") ? 16 : 2
        case .right:
            weight += tags.contains("side") ? 8 : 1
        case .left:
            weight += tags.contains("side") ? 7 : 1
        }

        switch context.objectGroup {
        case .vehicle:
            weight += tags.contains("vehicle") || tags.contains("urgent") ? 16 : 0
        case .mobility, .obstacle:
            weight += tags.contains("obstacle") || tags.contains("action") ? 13 : 3
        case .person:
            weight += tags.contains("person") || tags.contains("natural") ? 12 : 0
        case .traffic:
            break
        }

        switch context.distanceLevel {
        case .veryClose:
            weight += tags.contains("critical") || tags.contains("near") || tags.contains("action") ? 16 : -4
        case .close:
            weight += tags.contains("near") || isUrgentOrAction ? 10 : 0
        case .far:
            weight += tags.contains("low") || tags.contains("natural") ? 8 : -4
        case .near:
            break
        }

        return max(1, weight)
    }

    private static func chooseWeightedTemplate(
        _ templates: [(text: String, weight: Int)],
        random: (Double) -> Double
    ) -> String {
        let total = templates.reduce(0) { $0 + max(0, $1.weight) }
        guard total > 0 else { return templates[0].text }

        let cursor = random(Double(total))
        var running = 0.0
        for template in templates {
            running += Double(max(0, template.weight))
            if cursor <= running {
                return template.text
            }
        }
        return templates[templates.count - 1].text
    }

    private static func format(_ template: String, with context: MessageContext) -> String {
        template
            .replacingOccurrences(of: "{korean_label}", with: context.koreanLabel)
            .replacingOccurrences(of: "{particle}", with: context.particle)
            .replacingOccurrences(of: "{position_ko}", with: context.position.koreanName)
            .replacingOccurrences(of: "{position_from_ko}", with: context.position.koreanName + "에서")
            .replacingOccurrences(of: "{front_word}", with: context.frontWord)
    }

    private static func clamp01(_ value: Double) -> Double {
        max(0, min(1, value))
    }
}

/// Python's `sorted` is stable; Swift's sort is not guaranteed to be, so ties keep input order.
nonisolated func stableSorted<Element>(
    _ elements: [Element],
    by areInIncreasingOrder: (Element, Element) -> Bool
) -> [Element] {
    elements.enumerated()
        .sorted { lhs, rhs in
            if areInIncreasingOrder(lhs.element, rhs.element) { return true }
            if areInIncreasingOrder(rhs.element, lhs.element) { return false }
            return lhs.offset < rhs.offset
        }
        .map(\.element)
}

nonisolated enum GuidanceEventType: String {
    case newObject = "new_object"
    case closer
    case riskIncreased = "risk_increased"
    case enteredFrontZone = "entered_front_zone"
    case approaching
}

nonisolated struct GuidanceEvent {
    let type: GuidanceEventType
    let detection: SceneDetection
    let priority: Int
    var message = ""
}

/// Event-based speech suppression for live guidance.
///
/// The detector has no stable object IDs, so situations are keyed by object class
/// and a coarse horizontal bucket. That is enough to avoid repeating the same
/// narration without a heavy multi-object tracker.
nonisolated final class GuidanceEventTracker {
    nonisolated struct Cooldowns {
        var global: TimeInterval = 1.5
        var object: TimeInterval = 6.0
        var situation: TimeInterval = 8.0
        var signature: TimeInterval = 10.0
        var staleAfter: TimeInterval = 12.0
    }

    private nonisolated struct SituationState {
        var lastSeenAt: TimeInterval
        var lastSpokenAt: TimeInterval
        var riskScore: Int
        var riskLevel: RiskLevel
        var position: HorizontalPosition
        var areaRatio: Double
        var approaching: Bool
        var seenCount: Int
        /// LiDAR distance when this object was last announced.
        var spokenDistanceMeters: Double?
    }

    /// The backend bucketed box centers every 192 px on 1080 px wide frames.
    private static let referenceFrameWidth = 1080.0
    private static let objectBucketWidth = 192.0

    private let cooldowns: Cooldowns
    private let random: (Double) -> Double
    private var states: [String: SituationState] = [:]
    private var lastSituationAt: [String: TimeInterval] = [:]
    private var lastSignatureAt: [String: TimeInterval] = [:]
    private var lastSpokenAt: TimeInterval = 0

    init(
        cooldowns: Cooldowns = Cooldowns(),
        random: @escaping (Double) -> Double = { Double.random(in: 0...$0) }
    ) {
        self.cooldowns = cooldowns
        self.random = random
    }

    func reset() {
        states.removeAll()
        lastSituationAt.removeAll()
        lastSignatureAt.removeAll()
        lastSpokenAt = 0
    }

    /// Returns up to `limit` events worth announcing now, each with its sentence.
    func chooseEvents(_ detections: [SceneDetection], now: TimeInterval, limit: Int = 2) -> [GuidanceEvent] {
        states = states.filter { now - $0.value.lastSeenAt <= cooldowns.staleAfter }

        var candidates: [GuidanceEvent] = []
        for detection in detections {
            guard passesSpeechConfidence(detection) else {
                rememberSeen(detection, now: now, speechEligible: false)
                continue
            }

            guard isTemporallyConfirmed(detection) else {
                rememberSeen(detection, now: now)
                continue
            }

            let event = classifyEvent(detection, now: now)
            rememberSeen(detection, now: now)
            if let event {
                candidates.append(event)
            }
        }

        let ordered = stableSorted(candidates) { lhs, rhs in
            if lhs.priority != rhs.priority {
                return lhs.priority > rhs.priority
            }
            if lhs.detection.riskScore != rhs.detection.riskScore {
                return lhs.detection.riskScore > rhs.detection.riskScore
            }
            return lhs.detection.confidence > rhs.detection.confidence
        }

        var selected: [GuidanceEvent] = []
        for var event in ordered {
            event.message = GuidanceRules.detectionMessage(for: event.detection, eventType: event.type, random: random)
            guard !event.message.isEmpty, canSpeak(event, now: now) else { continue }

            selected.append(event)
            markSpoken(event, now: now)
            if selected.count >= limit {
                break
            }
        }

        return selected
    }

    private func classifyEvent(_ detection: SceneDetection, now: TimeInterval) -> GuidanceEvent? {
        guard GuidanceRules.guidanceLabels.contains(detection.label) else { return nil }

        let riskScore = detection.riskScore
        let isCenterFront = detection.frontDangerZone || detection.position == .center

        guard let state = states[objectKey(detection)] else {
            if riskScore >= 35 || isCenterFront {
                return GuidanceEvent(type: .newObject, detection: detection, priority: 45)
            }
            return nil
        }

        if now - state.lastSeenAt > cooldowns.staleAfter {
            return GuidanceEvent(type: .newObject, detection: detection, priority: 45)
        }

        if state.lastSpokenAt <= 0 && (riskScore >= 35 || isCenterFront) {
            return GuidanceEvent(type: .newObject, detection: detection, priority: 45)
        }

        if detection.approaching && !state.approaching {
            return GuidanceEvent(type: .approaching, detection: detection, priority: 95)
        }

        if isCenterFront && state.position != .center {
            return GuidanceEvent(type: .enteredFrontZone, detection: detection, priority: 85)
        }

        let scoreDelta = riskScore - state.riskScore
        let levelDelta = detection.riskLevel.rank - state.riskLevel.rank
        let areaDelta = detection.areaRatio - state.areaRatio

        if scoreDelta >= 18 || (levelDelta >= 1 && scoreDelta >= 8) {
            return GuidanceEvent(type: .riskIncreased, detection: detection, priority: 80 + max(levelDelta, 0) * 5)
        }

        // With LiDAR, announce again once an object is a meter closer than last time and near.
        if let meters = detection.distanceMeters, let spoken = state.spokenDistanceMeters,
           spoken - meters >= 1.0, meters < 3.0, riskScore >= 35 {
            return GuidanceEvent(type: .closer, detection: detection, priority: 72)
        }

        if areaDelta >= 0.035 && riskScore >= 45 {
            return GuidanceEvent(type: .closer, detection: detection, priority: 72)
        }

        return nil
    }

    private func rememberSeen(_ detection: SceneDetection, now: TimeInterval, speechEligible: Bool = true) {
        let key = objectKey(detection)
        let previous = states[key]
        let previousSeenCount = previous?.seenCount ?? 0

        states[key] = SituationState(
            lastSeenAt: now,
            lastSpokenAt: previous?.lastSpokenAt ?? 0,
            riskScore: detection.riskScore,
            riskLevel: detection.riskLevel,
            position: detection.position,
            areaRatio: detection.areaRatio,
            approaching: detection.approaching,
            seenCount: speechEligible ? previousSeenCount + 1 : previousSeenCount,
            spokenDistanceMeters: previous?.spokenDistanceMeters
        )
    }

    private func canSpeak(_ event: GuidanceEvent, now: TimeInterval) -> Bool {
        if now - lastSpokenAt < cooldowns.global {
            return false
        }

        let objectCooldown = event.type == .approaching || event.type == .enteredFrontZone
            ? 2.5
            : cooldowns.object
        if let state = states[objectKey(event.detection)], now - state.lastSpokenAt < objectCooldown {
            return false
        }

        if now - (lastSituationAt[situationKey(event)] ?? 0) < cooldowns.situation {
            return false
        }

        if now - (lastSignatureAt[speechSignature(event)] ?? 0) < cooldowns.signature {
            return false
        }

        return true
    }

    private func markSpoken(_ event: GuidanceEvent, now: TimeInterval) {
        states[objectKey(event.detection)]?.lastSpokenAt = now
        states[objectKey(event.detection)]?.spokenDistanceMeters = event.detection.distanceMeters
        lastSituationAt[situationKey(event)] = now
        lastSignatureAt[speechSignature(event)] = now
        lastSpokenAt = now
    }

    private func passesSpeechConfidence(_ detection: SceneDetection) -> Bool {
        let confidence = max(0, min(1, detection.confidence))

        if GuidanceRules.lowConfidenceSpeechLabels.contains(detection.label) {
            return confidence >= 0.25
        }
        if GuidanceRules.noisySpeechLabels.contains(detection.label) {
            return confidence >= 0.55
        }
        return confidence >= 0.35
    }

    /// Requires a few consecutive sightings before speaking, except for urgent objects.
    private func isTemporallyConfirmed(_ detection: SceneDetection) -> Bool {
        let confidence = max(0, min(1, detection.confidence))

        if GuidanceRules.immediateSpeechLabels.contains(detection.label) && confidence >= 0.35 {
            return true
        }
        if detection.approaching {
            return true
        }

        let seenCount = (states[objectKey(detection)]?.seenCount ?? 0) + 1
        let requiredCount = GuidanceRules.noisySpeechLabels.contains(detection.label) ? 3 : 2
        return seenCount >= requiredCount
    }

    private func objectKey(_ detection: SceneDetection) -> String {
        let scale = detection.frameWidth > 0 ? Self.referenceFrameWidth / detection.frameWidth : 1
        let bucket = Int((detection.bbox.centerX * scale / Self.objectBucketWidth).rounded(.down))
        return "\(detection.label):\(bucket)"
    }

    private func situationKey(_ event: GuidanceEvent) -> String {
        "\(event.type.rawValue):\(event.detection.label):\(event.detection.position.rawValue)"
    }

    private func speechSignature(_ event: GuidanceEvent) -> String {
        let detection = event.detection
        let distance = GuidanceRules.distanceLevel(of: detection)
        return "\(event.type.rawValue):\(detection.label):\(detection.position.rawValue):\(detection.riskLevel.rawValue):\(distance.rawValue)"
    }
}
