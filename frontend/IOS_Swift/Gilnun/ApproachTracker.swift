import Foundation

nonisolated struct ApproachAlert {
    /// Index of the detection passed to `update` that triggered the alert.
    let detectionIndex: Int
    let trackID: Int
    let label: String
    let growthRatio: Double
}

/// Lightweight approach tracking for monocular camera detections.
///
/// Port of `vision/tracker_logic.py`: boxes are matched to tracks by IoU, and a
/// track whose box area keeps growing over recent frames is likely getting closer.
nonisolated final class ApproachTracker {
    private nonisolated final class Track {
        let id: Int
        let label: String
        var bbox: BoundingBox
        var lastSeen: TimeInterval
        var areaHistory: [Double] = []

        init(id: Int, label: String, bbox: BoundingBox, lastSeen: TimeInterval) {
            self.id = id
            self.label = label
            self.bbox = bbox
            self.lastSeen = lastSeen
        }
    }

    private let minIoU: Double
    private let maxTrackAge: TimeInterval
    private let historySize: Int
    private let minGrowthRatio: Double
    private var tracks: [Track] = []
    private var nextTrackID = 1

    init(
        minIoU: Double = 0.25,
        maxTrackAge: TimeInterval = 1.0,
        historySize: Int = 5,
        minGrowthRatio: Double = 1.35
    ) {
        self.minIoU = minIoU
        self.maxTrackAge = maxTrackAge
        self.historySize = historySize
        self.minGrowthRatio = minGrowthRatio
    }

    func update(_ detections: [(label: String, bbox: BoundingBox)], now: TimeInterval) -> [ApproachAlert] {
        tracks.removeAll { now - $0.lastSeen > maxTrackAge }

        var alerts: [ApproachAlert] = []
        for (index, detection) in detections.enumerated() {
            let track = matchOrCreateTrack(label: detection.label, bbox: detection.bbox, now: now)
            track.bbox = detection.bbox
            track.lastSeen = now
            track.areaHistory.append(detection.bbox.area)
            if track.areaHistory.count > historySize {
                track.areaHistory.removeFirst(track.areaHistory.count - historySize)
            }

            if let growthRatio = approachGrowthRatio(of: track) {
                alerts.append(
                    ApproachAlert(
                        detectionIndex: index,
                        trackID: track.id,
                        label: track.label,
                        growthRatio: growthRatio
                    )
                )
            }
        }

        return alerts
    }

    func reset() {
        tracks.removeAll()
        nextTrackID = 1
    }

    private func matchOrCreateTrack(label: String, bbox: BoundingBox, now: TimeInterval) -> Track {
        var bestTrack: Track?
        var bestIoU = 0.0

        for track in tracks where track.label == label {
            let score = track.bbox.intersectionOverUnion(with: bbox)
            if score > bestIoU {
                bestIoU = score
                bestTrack = track
            }
        }

        if let bestTrack, bestIoU >= minIoU {
            return bestTrack
        }

        let track = Track(id: nextTrackID, label: label, bbox: bbox, lastSeen: now)
        nextTrackID += 1
        tracks.append(track)
        return track
    }

    private func approachGrowthRatio(of track: Track) -> Double? {
        let history = track.areaHistory
        guard history.count >= 3 else { return nil }

        let growthRatio = history[history.count - 1] / max(history[0], 1.0)
        let isMonotonicEnough = history[history.count - 1] > history[history.count - 2]
            && history[history.count - 2] > history[history.count - 3]
        guard growthRatio >= minGrowthRatio, isMonotonicEnough else { return nil }

        return (growthRatio * 100).rounded(.toNearestOrEven) / 100
    }
}
