import CoreGraphics
import Foundation

struct TextStabilityDecision {
    let status: LiveOCRStatus
    let shouldRunFullOCR: Bool
}

final class TextStabilityTracker {
    private var detectedStartDate: Date?
    private var stableStartDate: Date?
    private var smoothedRegion: CGRect?
    private let requiredStableDuration: TimeInterval = 0.75
    private let persistentTextFallbackDuration: TimeInterval = 1.5
    private let minimumRegionArea: CGFloat = 0.035
    private let maximumCenterDistance: CGFloat = 0.28
    private let minimumBlurScore = 5.0
    private let maximumMovementScore = 13.0

    func reset() {
        detectedStartDate = nil
        stableStartDate = nil
        smoothedRegion = nil
    }

    func update(with analysis: OCRFrameAnalysis) -> TextStabilityDecision {
        guard let region = analysis.textRegion else {
            reset()
            return TextStabilityDecision(status: .searching, shouldRunFullOCR: false)
        }

        if detectedStartDate == nil {
            detectedStartDate = analysis.timestamp
        }

        let isReadableCandidate = isLargeEnough(region)
            && isCentered(region)
            && analysis.blurScore >= minimumBlurScore
            && analysis.movementScore <= maximumMovementScore
            && analysis.confidence >= 0.45

        guard isReadableCandidate else {
            stableStartDate = nil
            smoothedRegion = smooth(region, previous: smoothedRegion)

            let detectedDuration = analysis.timestamp.timeIntervalSince(detectedStartDate ?? analysis.timestamp)
            if detectedDuration >= persistentTextFallbackDuration {
                detectedStartDate = nil
                return TextStabilityDecision(status: .reading, shouldRunFullOCR: true)
            }

            return TextStabilityDecision(status: .detected, shouldRunFullOCR: false)
        }

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

    private func isLargeEnough(_ region: CGRect) -> Bool {
        region.width * region.height >= minimumRegionArea
    }

    private func isCentered(_ region: CGRect) -> Bool {
        let center = CGPoint(x: region.midX, y: region.midY)
        let dx = center.x - 0.5
        let dy = center.y - 0.5
        return sqrt(dx * dx + dy * dy) <= maximumCenterDistance
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
