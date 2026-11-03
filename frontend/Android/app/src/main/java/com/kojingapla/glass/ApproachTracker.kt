package com.kojingapla.glass

import kotlin.math.max

data class ApproachAlert(
    /** Index of the detection passed to `update` that triggered the alert. */
    val detectionIndex: Int,
    val trackId: Int,
    val label: String,
    val growthRatio: Double
)

/**
 * Lightweight approach tracking for monocular camera detections.
 *
 * Port of `vision/tracker_logic.py` (via iOS `ApproachTracker.swift`): boxes are matched
 * to tracks by IoU, and a track whose box area keeps growing over recent frames is
 * likely getting closer. Times are in seconds.
 */
class ApproachTracker(
    private val minIoU: Double = 0.25,
    private val maxTrackAge: Double = 1.0,
    private val historySize: Int = 5,
    private val minGrowthRatio: Double = 1.35
) {
    private class Track(val id: Int, val label: String, var bbox: BoundingBox, var lastSeen: Double) {
        val areaHistory = ArrayList<Double>()
    }

    private val tracks = ArrayList<Track>()
    private var nextTrackId = 1

    fun update(detections: List<Pair<String, BoundingBox>>, now: Double): List<ApproachAlert> {
        tracks.removeAll { now - it.lastSeen > maxTrackAge }

        val alerts = ArrayList<ApproachAlert>()
        detections.forEachIndexed { index, (label, bbox) ->
            val track = matchOrCreateTrack(label, bbox, now)
            track.bbox = bbox
            track.lastSeen = now
            track.areaHistory.add(bbox.area)
            while (track.areaHistory.size > historySize) track.areaHistory.removeAt(0)

            approachGrowthRatio(track)?.let { ratio ->
                alerts.add(ApproachAlert(index, track.id, track.label, ratio))
            }
        }
        return alerts
    }

    fun reset() {
        tracks.clear()
        nextTrackId = 1
    }

    private fun matchOrCreateTrack(label: String, bbox: BoundingBox, now: Double): Track {
        var bestTrack: Track? = null
        var bestIoU = 0.0
        for (track in tracks) {
            if (track.label != label) continue
            val score = track.bbox.intersectionOverUnion(bbox)
            if (score > bestIoU) {
                bestIoU = score
                bestTrack = track
            }
        }
        if (bestTrack != null && bestIoU >= minIoU) return bestTrack

        val track = Track(nextTrackId, label, bbox, now)
        nextTrackId += 1
        tracks.add(track)
        return track
    }

    private fun approachGrowthRatio(track: Track): Double? {
        val history = track.areaHistory
        if (history.size < 3) return null

        val last = history.size - 1
        val growthRatio = history[last] / max(history[0], 1.0)
        val isMonotonicEnough = history[last] > history[last - 1] && history[last - 1] > history[last - 2]
        if (growthRatio < minGrowthRatio || !isMonotonicEnough) return null

        return roundHalfEven(growthRatio, 2)
    }
}
