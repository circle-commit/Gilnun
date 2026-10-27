import CoreGraphics
import Foundation

// MARK: - Ports of backend/tests

func runGuidanceUnitTests() {
    // test_guidance_distance_estimation.py
    expect(GuidanceRules.distanceLevel(label: "car", areaRatio: 0.05, verticalRatio: 0.65) == .near, "car at 5% area is near")
    expect(GuidanceRules.distanceLevel(label: "car", areaRatio: 0.08, verticalRatio: 0.65) == .close, "car at 8% area is close")
    expect(GuidanceRules.distanceLevel(label: "bollard", areaRatio: 0.011, verticalRatio: 0.61) == .close, "low bollard is close")
    expect(GuidanceRules.distanceLevel(label: "bollard", areaRatio: 0.011, verticalRatio: 0.43) == .near, "higher bollard is near")
    expect(GuidanceRules.distanceLevel(label: "traffic_sign", areaRatio: 0.02, verticalRatio: 0.55) == .near, "sign is near")
    expect(GuidanceRules.distanceLevel(label: "traffic_sign", areaRatio: 0.01, verticalRatio: 0.55) == .far, "small sign is far")

    let front = GuidanceRules.enrichAndPrioritize([
        unitDetection("car", areaRatio: 0.05, verticalRatio: 0.65),
        unitDetection("bollard", areaRatio: 0.011, verticalRatio: 0.61),
    ])
    let dangerByLabel = Dictionary(uniqueKeysWithValues: front.map { ($0.label, $0.frontDangerZone) })
    expect(dangerByLabel["car"] == false, "car is not in the front danger zone")
    expect(dangerByLabel["bollard"] == true, "bollard is in the front danger zone")

    // test_guidance_temporal_confidence.py
    var now = 100.0
    func events(_ tracker: GuidanceEventTracker, _ label: String, confidence: Double = 0.8) -> Int {
        now += 0.2
        let detections = GuidanceRules.enrichAndPrioritize([
            unitDetection(label, confidence: confidence, areaRatio: 0.05, verticalRatio: 0.7),
        ])
        return tracker.chooseEvents(detections, now: now).count
    }
    func zeroCooldownTracker() -> GuidanceEventTracker {
        GuidanceEventTracker(cooldowns: .init(global: 0, object: 0, situation: 0, signature: 0))
    }

    let person = zeroCooldownTracker()
    expect(events(person, "person") == 0, "person needs a second frame")
    expect(events(person, "person") == 1, "person speaks on the second frame")

    let pole = zeroCooldownTracker()
    expect(events(pole, "pole") == 0, "pole frame 1")
    expect(events(pole, "pole") == 0, "pole frame 2")
    expect(events(pole, "pole") == 1, "pole speaks on the third frame")

    let noisyPole = zeroCooldownTracker()
    for frame in 1...3 {
        expect(events(noisyPole, "pole", confidence: 0.4) == 0, "low-confidence pole frame \(frame) is ignored")
    }
    expect(events(noisyPole, "pole") == 0, "confident pole frame 1")
    expect(events(noisyPole, "pole") == 0, "confident pole frame 2")
    expect(events(noisyPole, "pole") == 1, "confident pole speaks on the third frame")

    expect(events(zeroCooldownTracker(), "car") == 1, "car speaks on the first frame")

    // Calmer speech (the iOS app): fixed objects beside the path only within reach,
    // a calm gap between ordinary sentences, and poles that do not "come closer".
    func object(_ label: String, meters: Double, _ position: HorizontalPosition = .left, right: Double = 300) -> SceneDetection {
        var detection = SceneDetection(
            label: label, koreanLabel: SceneAnalyzer.koreanLabels[label] ?? label, confidence: 0.8,
            bbox: BoundingBox(x1: 100, y1: 900, x2: right, y2: 1300), frameWidth: 1080,
            position: position, areaRatio: 0.05, verticalRatio: 0.7
        )
        detection.distanceMeters = meters
        return detection
    }
    func spoken(_ tracker: GuidanceEventTracker, _ detection: SceneDetection, frames: Int) -> [String] {
        (0..<frames).flatMap { _ -> [String] in
            now += 0.2
            return tracker.chooseEvents(GuidanceRules.enrichAndPrioritize([detection]), now: now).map(\.message)
        }
    }
    func tracker(calmer: Bool) -> GuidanceEventTracker {
        GuidanceEventTracker(cooldowns: .init(global: 0, object: 0, situation: 0, signature: 0), calmerSpeech: calmer) { _ in 0 }
    }
    expect(spoken(tracker(calmer: false), object("pole", meters: 3), frames: 4).count == 1, "backend: pole beside the path is spoken")
    expect(spoken(tracker(calmer: true), object("pole", meters: 3), frames: 4).isEmpty, "calmer: pole beside the path at 3 m is quiet")
    expect(spoken(tracker(calmer: true), object("pole", meters: 1), frames: 4).count == 1, "calmer: pole within reach is spoken")
    expect(spoken(tracker(calmer: true), object("pole", meters: 3, .center), frames: 4).count == 1, "calmer: pole in the path is spoken")
    expect(
        spoken(tracker(calmer: true), object("movable_signage", meters: 2, right: 600), frames: 4).count == 1,
        "calmer: a sign beside the path whose box crosses the middle reaches into the path and is spoken"
    )
    let reach = tracker(calmer: true)
    expect(spoken(reach, object("pole", meters: 2), frames: 4).isEmpty, "calmer: pole at 2 m beside the path is quiet")
    expect(spoken(reach, object("pole", meters: 1.4), frames: 1).count == 1, "calmer: spoken on its first frame within 1.5 m")
    let longWalk = tracker(calmer: true)
    expect(spoken(longWalk, object("pole", meters: 2), frames: 65).isEmpty, "calmer: pole beside the path for 13 s is quiet")
    expect(spoken(longWalk, object("pole", meters: 1.4), frames: 1).count == 1, "calmer: still spoken on its first frame within reach")
    expect(spoken(tracker(calmer: true), object("car", meters: 6), frames: 1).count == 1, "calmer: vehicles beside the path are still spoken")

    let gap = tracker(calmer: true)
    expect(spoken(gap, object("person", meters: 4), frames: 2).count == 1, "calmer: person beside the path is spoken")
    expect(spoken(gap, object("dog", meters: 4, .right), frames: 3).isEmpty, "calmer: another calm sentence waits")
    now += 4
    expect(spoken(gap, object("dog", meters: 4, .right), frames: 1).count == 1, "calmer: spoken after the calm gap")
    expect(spoken(gap, object("bollard", meters: 2, .center), frames: 3).count == 1, "calmer: something in the path does not wait")
    expect(spoken(gap, object("car", meters: 6, .right), frames: 1).isEmpty, "calmer: a car beside the path and not coming waits")
    var comingPerson = object("person", meters: 4)
    comingPerson.approaching = true
    expect(spoken(gap, comingPerson, frames: 1).count == 1, "calmer: a person coming closer does not wait")

    // The screen keeps "위험" (85 and up) for what is in the way or coming at the user.
    let parkedCar = GuidanceRules.enrichAndPrioritize([object("car", meters: 6)])[0]
    let personAhead = GuidanceRules.enrichAndPrioritize([object("person", meters: 2, .center)])[0]
    expect(parkedCar.riskScore >= 85 && GuidanceRules.displayRisk(parkedCar) == 84, "display: a car beside the path shows 주의")
    expect(GuidanceRules.displayRisk(personAhead) >= 85, "display: a person in the way shows 위험")
    var sideSign = object("movable_signage", meters: 2)
    sideSign.approaching = true
    var crossingSign = object("movable_signage", meters: 2, right: 600)
    crossingSign.approaching = true
    let signs = GuidanceRules.enrichAndPrioritize([sideSign, crossingSign])
    expect(signs.allSatisfy { $0.riskScore >= 85 }, "display: both signs score high")
    expect(GuidanceRules.displayRisk(signs.first { !GuidanceRules.reachesPath($0) }!) == 84, "display: a sign beside the path shows 주의")
    expect(GuidanceRules.displayRisk(signs.first { GuidanceRules.reachesPath($0) }!) >= 85, "display: a sign reaching into the path shows 위험")

    var movingPole = object("pole", meters: 1)
    movingPole.approaching = true
    let pole1m = GuidanceRules.enrichAndPrioritize([movingPole])[0]
    let backendWords = GuidanceRules.detectionMessage(for: pole1m, eventType: .approaching) { _ in 0 }
    let calmerWords = GuidanceRules.detectionMessage(for: pole1m, eventType: .approaching, calmerSpeech: true) { _ in 0 }
    expect(backendWords.contains("다가오고"), "backend: \(backendWords)")
    expect(!calmerWords.contains("다가오고") && calmerWords.contains("가까워지고"), "calmer: a pole does not come closer: \(calmerWords)")
}

private func unitDetection(
    _ label: String,
    confidence: Double = 0.8,
    areaRatio: Double,
    verticalRatio: Double
) -> SceneDetection {
    SceneDetection(
        label: label,
        koreanLabel: label,
        confidence: confidence,
        bbox: BoundingBox(x1: 100, y1: 100, x2: 300, y2: 400),
        frameWidth: 1080,
        position: .center,
        areaRatio: areaRatio,
        verticalRatio: verticalRatio
    )
}

// MARK: - Parity with the Python backend

private struct Fixtures: Decodable {
    let frameWidth: Double
    let frameHeight: Double
    let rules: [RuleCase]
    let priority: [PriorityCase]
    let templates: [TemplateCase]
    let eventSequences: [EventSequence]
    let approachSequences: [ApproachSequence]
    let sceneSequences: [SceneSequence]
}

private struct FixtureDetection: Decodable {
    let label: String
    let koreanLabel: String
    let confidence: Double
    let bboxXyxy: [Double]
    let position: String
    let areaRatio: Double
    let verticalRatio: Double
    let approaching: Bool

    func sceneDetection(frameWidth: Double, koreanLabel overrideLabel: String? = nil) -> SceneDetection {
        var detection = SceneDetection(
            label: label,
            koreanLabel: overrideLabel ?? koreanLabel,
            confidence: confidence,
            bbox: BoundingBox(x1: bboxXyxy[0], y1: bboxXyxy[1], x2: bboxXyxy[2], y2: bboxXyxy[3]),
            frameWidth: frameWidth,
            position: HorizontalPosition(rawValue: position) ?? .center,
            areaRatio: areaRatio,
            verticalRatio: verticalRatio
        )
        detection.approaching = approaching
        return detection
    }
}

private struct RuleCase: Decodable {
    let detection: FixtureDetection
    let frontDangerZone: Bool
    let riskScore: Int
    let riskLevel: String
    let distanceLevel: String
    let objectGroup: String
    let particle: String
}

private struct PriorityCase: Decodable {
    let detections: [FixtureDetection]
    let order: [Int]
}

private struct WeightedTemplate: Decodable, Equatable {
    let text: String
    let weight: Int

    init(text: String, weight: Int) {
        self.text = text
        self.weight = weight
    }

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        text = try container.decode(String.self)
        weight = try container.decode(Int.self)
    }
}

private struct TemplateCase: Decodable {
    let detection: FixtureDetection
    let eventType: String
    let templates: [WeightedTemplate]
    let message: String
}

private struct CooldownFixture: Decodable {
    let global: Double
    let object: Double
    let situation: Double
    let signature: Double
    let staleAfter: Double
}

private struct ExpectedEvent: Decodable {
    let event: String
    let label: String
    let position: String
    let message: String
}

private struct EventFrame: Decodable {
    let now: Double
    let detections: [FixtureDetection]
    let events: [ExpectedEvent]
}

private struct EventSequence: Decodable {
    let cooldowns: CooldownFixture
    let frames: [EventFrame]
}

private struct ApproachDetection: Decodable {
    let label: String
    let bbox: [Double]
}

private struct ExpectedAlert: Decodable {
    let index: Int
    let trackId: Int
    let growthRatio: Double
}

private struct ApproachFrame: Decodable {
    let now: Double
    let detections: [ApproachDetection]
    let alerts: [ExpectedAlert]
}

private struct ApproachSequence: Decodable {
    let frames: [ApproachFrame]
}

private struct RawFixture: Decodable {
    let label: String
    let confidence: Double
    let bbox: [Double]
}

private struct ExpectedSceneDetection: Decodable {
    let label: String
    let position: String
    let approaching: Bool
    let riskScore: Int
    let riskLevel: String
}

private struct SceneFrame: Decodable {
    let now: Double
    let raw: [RawFixture]
    let voiceGuide: String
    let detections: [ExpectedSceneDetection]
}

private struct SceneSequence: Decodable {
    let frames: [SceneFrame]
}

private func box(_ values: [Double]) -> BoundingBox {
    BoundingBox(x1: values[0], y1: values[1], x2: values[2], y2: values[3])
}

func runGuidanceParityTests(fixturesURL: URL) {
    let fixtures: Fixtures
    do {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        fixtures = try decoder.decode(Fixtures.self, from: Data(contentsOf: fixturesURL))
    } catch {
        expect(false, "could not read guidance fixtures: \(error)")
        return
    }

    let frameWidth = fixtures.frameWidth

    for (index, testCase) in fixtures.rules.enumerated() {
        let detection = GuidanceRules.enrichAndPrioritize([testCase.detection.sceneDetection(frameWidth: frameWidth)])[0]
        let name = "rules[\(index)] \(testCase.detection.label)"
        expect(detection.frontDangerZone == testCase.frontDangerZone, "\(name) front danger zone")
        expect(detection.riskScore == testCase.riskScore, "\(name) risk score \(detection.riskScore) != \(testCase.riskScore)")
        expect(detection.riskLevel.rawValue == testCase.riskLevel, "\(name) risk level")
        expect(GuidanceRules.distanceLevel(of: detection).rawValue == testCase.distanceLevel, "\(name) distance level")
        expect(GuidanceRules.objectGroup(for: detection.label).rawValue == testCase.objectGroup, "\(name) object group")
        expect(GuidanceRules.particle(for: detection.koreanLabel) == testCase.particle, "\(name) particle")
    }

    for (index, testCase) in fixtures.priority.enumerated() {
        // The Korean label carries the input index; it does not affect ordering.
        let detections = testCase.detections.enumerated().map { offset, detection in
            detection.sceneDetection(frameWidth: frameWidth, koreanLabel: String(offset))
        }
        let order = GuidanceRules.enrichAndPrioritize(detections).compactMap { Int($0.koreanLabel) }
        expect(order == testCase.order, "priority[\(index)] order \(order) != \(testCase.order)")
    }

    for (index, testCase) in fixtures.templates.enumerated() {
        let detection = GuidanceRules.enrichAndPrioritize([testCase.detection.sceneDetection(frameWidth: frameWidth)])[0]
        guard let eventType = GuidanceEventType(rawValue: testCase.eventType) else {
            expect(false, "templates[\(index)] unknown event \(testCase.eventType)")
            continue
        }
        let context = GuidanceRules.messageContext(for: detection, eventType: eventType)
        let templates = GuidanceRules.weightedTemplates(for: context).map { WeightedTemplate(text: $0.text, weight: $0.weight) }
        expect(templates == testCase.templates, "templates[\(index)] candidates \(templates) != \(testCase.templates)")
        let message = GuidanceRules.detectionMessage(for: detection, eventType: eventType, random: { _ in 0 })
        expect(message == testCase.message, "templates[\(index)] message \(message) != \(testCase.message)")
    }

    for (sequenceIndex, sequence) in fixtures.eventSequences.enumerated() {
        let cooldowns = GuidanceEventTracker.Cooldowns(
            global: sequence.cooldowns.global,
            object: sequence.cooldowns.object,
            situation: sequence.cooldowns.situation,
            signature: sequence.cooldowns.signature,
            staleAfter: sequence.cooldowns.staleAfter
        )
        let tracker = GuidanceEventTracker(cooldowns: cooldowns, random: { _ in 0 })
        for (frameIndex, frame) in sequence.frames.enumerated() {
            let detections = GuidanceRules.enrichAndPrioritize(frame.detections.map { $0.sceneDetection(frameWidth: frameWidth) })
            let actual = tracker.chooseEvents(detections, now: frame.now).map {
                "\($0.type.rawValue)|\($0.detection.label)|\($0.detection.position.rawValue)|\($0.message)"
            }
            let expected = frame.events.map { "\($0.event)|\($0.label)|\($0.position)|\($0.message)" }
            expect(actual == expected, "event_sequences[\(sequenceIndex)][\(frameIndex)] \(actual) != \(expected)")
        }
    }

    for (sequenceIndex, sequence) in fixtures.approachSequences.enumerated() {
        let tracker = ApproachTracker(minGrowthRatio: 1.28)
        for (frameIndex, frame) in sequence.frames.enumerated() {
            let alerts = tracker.update(frame.detections.map { (label: $0.label, bbox: box($0.bbox)) }, now: frame.now)
            let actual = alerts.map { "\($0.detectionIndex)|\($0.trackID)|\($0.growthRatio)" }
            let expected = frame.alerts.map { "\($0.index)|\($0.trackId)|\($0.growthRatio)" }
            expect(actual == expected, "approach_sequences[\(sequenceIndex)][\(frameIndex)] \(actual) != \(expected)")
        }
    }

    let frameSize = CGSize(width: fixtures.frameWidth, height: fixtures.frameHeight)
    for (sequenceIndex, sequence) in fixtures.sceneSequences.enumerated() {
        let analyzer = SceneAnalyzer(eventTracker: GuidanceEventTracker(random: { _ in 0 }))
        for (frameIndex, frame) in sequence.frames.enumerated() {
            let raw = frame.raw.map { RawDetection(label: $0.label, confidence: $0.confidence, bbox: box($0.bbox)) }
            let result = analyzer.analyze(raw, frameSize: frameSize, now: frame.now)
            let name = "scene_sequences[\(sequenceIndex)][\(frameIndex)]"
            expect(result.voiceGuide == frame.voiceGuide, "\(name) voice guide \"\(result.voiceGuide)\" != \"\(frame.voiceGuide)\"")

            let actual = result.detections.map {
                "\($0.label)|\($0.position.rawValue)|\($0.approaching)|\($0.riskScore)|\($0.riskLevel.rawValue)"
            }
            let expected = frame.detections.map {
                "\($0.label)|\($0.position)|\($0.approaching)|\($0.riskScore)|\($0.riskLevel)"
            }
            expect(actual == expected, "\(name) detections \(actual) != \(expected)")
        }
    }

    let sceneFrames = fixtures.sceneSequences.reduce(0) { $0 + $1.frames.count }
    let spokenFrames = fixtures.sceneSequences.reduce(0) { $0 + $1.frames.filter { !$0.voiceGuide.isEmpty }.count }
    let approachAlerts = fixtures.approachSequences.reduce(0) { $0 + $1.frames.reduce(0) { $0 + $1.alerts.count } }
    print(
        "guidance parity: \(fixtures.rules.count) rule cases, \(fixtures.priority.count) orderings, " +
        "\(fixtures.templates.count) template cases, \(fixtures.eventSequences.count) event sequences, " +
        "\(approachAlerts) approach alerts, \(sceneFrames) scene frames (\(spokenFrames) with speech)"
    )
}
