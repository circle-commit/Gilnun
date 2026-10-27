import CoreGraphics
import Foundation

struct TextStabilityDecision {
    let status: LiveOCRStatus
    let shouldRunFullOCR: Bool
    /// Spoken help for aiming the camera, already spaced out; nil when nothing should be said.
    var hint: TextAimingHint? = nil
}

/// Tells users who cannot see the screen how to move the phone so the text fits the frame.
enum TextAimingHint: String {
    case moveLeft = "휴대폰을 왼쪽으로 조금 옮겨 주세요."
    case moveRight = "휴대폰을 오른쪽으로 조금 옮겨 주세요."
    case moveUp = "휴대폰을 위쪽으로 조금 옮겨 주세요."
    case moveDown = "휴대폰을 아래쪽으로 조금 옮겨 주세요."
    case moveFarther = "휴대폰을 조금 멀리 떼 주세요."
    case moveCloser = "휴대폰을 글자에 더 가까이 대 주세요."
    case glare = "빛이 반사돼요. 휴대폰을 조금 기울여 주세요."
}

final class TextStabilityTracker {
    private var detectedStartDate: Date?
    private var stableStartDate: Date?
    private var smoothedRegion: CGRect?
    private let requiredStableDuration: TimeInterval = 0.75
    /// Text that never becomes readable is read anyway after this long. Long enough for a
    /// spoken hint and the user's move, since partial text is better than none.
    private let persistentTextFallbackDuration: TimeInterval = 3.0
    private static let minimumRegionArea: CGFloat = 0.035
    private static let maximumCenterDistance: CGFloat = 0.28
    /// Text within this distance of the frame's edge is likely cut off.
    private static let edgeMargin: CGFloat = 0.02
    private static let maximumGlareRatio = 0.05
    private let minimumBlurScore = 5.0
    private let maximumMovementScore = 13.0
    /// A problem must last this long before it is spoken, so passing frames stay quiet.
    private let hintDelay: TimeInterval = 0.5
    private let hintInterval: TimeInterval = 2.0
    private var hintCandidate: (hint: TextAimingHint, since: Date)?
    private var lastHintDate: Date = .distantPast

    func reset() {
        detectedStartDate = nil
        stableStartDate = nil
        smoothedRegion = nil
        hintCandidate = nil
    }

    /// What keeps `region` (Vision coordinates, origin bottom-left) from being read well, most important first.
    static func aimingHint(for region: CGRect, glareRatio: Double) -> TextAimingHint? {
        let cutLeft = region.minX <= edgeMargin
        let cutRight = region.maxX >= 1 - edgeMargin
        let cutBottom = region.minY <= edgeMargin
        let cutTop = region.maxY >= 1 - edgeMargin
        if (cutLeft && cutRight) || (cutTop && cutBottom) { return .moveFarther }
        if cutLeft { return .moveLeft }
        if cutRight { return .moveRight }
        if cutTop { return .moveUp }
        if cutBottom { return .moveDown }
        if region.width * region.height < minimumRegionArea { return .moveCloser }

        let dx = region.midX - 0.5
        let dy = region.midY - 0.5
        if sqrt(dx * dx + dy * dy) > maximumCenterDistance {
            if abs(dx) >= abs(dy) { return dx < 0 ? .moveLeft : .moveRight }
            return dy > 0 ? .moveUp : .moveDown
        }
        return glareRatio >= maximumGlareRatio ? .glare : nil
    }

    func update(with analysis: OCRFrameAnalysis) -> TextStabilityDecision {
        guard let region = analysis.textRegion else {
            reset()
            return TextStabilityDecision(status: .searching, shouldRunFullOCR: false)
        }

        if detectedStartDate == nil {
            detectedStartDate = analysis.timestamp
        }

        let aimingHint = Self.aimingHint(for: region, glareRatio: analysis.glareRatio)
        let isReadableCandidate = aimingHint == nil
            && analysis.blurScore >= minimumBlurScore
            && analysis.movementScore <= maximumMovementScore
            && analysis.confidence >= 0.45

        guard isReadableCandidate else {
            stableStartDate = nil
            smoothedRegion = smooth(region, previous: smoothedRegion)

            let detectedDuration = analysis.timestamp.timeIntervalSince(detectedStartDate ?? analysis.timestamp)
            if detectedDuration >= persistentTextFallbackDuration {
                detectedStartDate = nil
                hintCandidate = nil
                return TextStabilityDecision(status: .reading, shouldRunFullOCR: true)
            }

            return TextStabilityDecision(status: .detected, shouldRunFullOCR: false, hint: spokenHint(aimingHint, now: analysis.timestamp))
        }
        hintCandidate = nil

        let isSameTarget = smoothedRegion.map { intersectionOverUnion($0, region) >= 0.55 } ?? true
        smoothedRegion = smooth(region, previous: isSameTarget ? smoothedRegion : nil)

        if !isSameTarget {
            stableStartDate = analysis.timestamp
        } else if stableStartDate == nil {
            stableStartDate = analysis.timestamp
        }

        let stableDuration = analysis.timestamp.timeIntervalSince(stableStartDate ?? analysis.timestamp)
        if stableDuration >= requiredStableDuration {
            detectedStartDate = nil
            stableStartDate = nil
            return TextStabilityDecision(status: .reading, shouldRunFullOCR: true)
        }

        return TextStabilityDecision(status: .stabilizing, shouldRunFullOCR: false)
    }

    /// `hint` once it has lasted `hintDelay`, at most once every `hintInterval`.
    private func spokenHint(_ hint: TextAimingHint?, now: Date) -> TextAimingHint? {
        guard let hint else {
            hintCandidate = nil
            return nil
        }
        guard let candidate = hintCandidate, candidate.hint == hint else {
            hintCandidate = (hint, now)
            return nil
        }
        guard now.timeIntervalSince(candidate.since) >= hintDelay,
              now.timeIntervalSince(lastHintDate) >= hintInterval else { return nil }

        lastHintDate = now
        return hint
    }

    private func smooth(_ region: CGRect, previous: CGRect?) -> CGRect {
        guard let previous else { return region }

        let weight: CGFloat = 0.65
        return CGRect(
            x: previous.origin.x * weight + region.origin.x * (1 - weight),
            y: previous.origin.y * weight + region.origin.y * (1 - weight),
            width: previous.width * weight + region.width * (1 - weight),
            height: previous.height * weight + region.height * (1 - weight)
        )
    }

    private func intersectionOverUnion(_ first: CGRect, _ second: CGRect) -> CGFloat {
        let intersection = first.intersection(second)
        guard !intersection.isNull else { return 0 }

        let intersectionArea = intersection.width * intersection.height
        let unionArea = first.width * first.height + second.width * second.height - intersectionArea
        return unionArea <= 0 ? 0 : intersectionArea / unionArea
    }
}
