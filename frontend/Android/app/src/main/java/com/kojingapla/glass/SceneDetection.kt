package com.kojingapla.glass

import kotlin.math.max
import kotlin.math.min

/**
 * Axis-aligned box in upright frame pixel coordinates (origin top-left), stored as
 * corner points like the backend's `bbox_xyxy`. Port of iOS `SceneDetection.swift`.
 */
data class BoundingBox(val x1: Double, val y1: Double, val x2: Double, val y2: Double) {
    val area: Double get() = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    val centerX: Double get() = (x1 + x2) / 2

    fun intersectionOverUnion(other: BoundingBox): Double {
        val intersection = max(0.0, min(x2, other.x2) - max(x1, other.x1)) *
            max(0.0, min(y2, other.y2) - max(y1, other.y1))
        val union = area + other.area - intersection
        return if (union > 0) intersection / union else 0.0
    }
}

/** One object found by the detector, before any guidance logic is applied. */
data class RawDetection(val label: String, val confidence: Double, val bbox: BoundingBox)

enum class HorizontalPosition(val rawValue: String, val koreanName: String) {
    LEFT("left", "왼쪽"),
    CENTER("center", "정면"),
    RIGHT("right", "오른쪽");

    companion object {
        /** Splits the frame into thirds, like the backend's `_position_from_bbox`. */
        fun of(centerX: Double, frameWidth: Double): HorizontalPosition = when {
            centerX < frameWidth / 3 -> LEFT
            centerX > frameWidth * 2 / 3 -> RIGHT
            else -> CENTER
        }

        fun fromRaw(value: String): HorizontalPosition = entries.firstOrNull { it.rawValue == value } ?: CENTER
    }
}

/** Declared from low to high, so `compareTo` follows the risk rank. */
enum class RiskLevel(val rawValue: String) {
    LOW("low"),
    MEDIUM("medium"),
    HIGH("high"),
    CRITICAL("critical");

    val rank: Int get() = ordinal

    companion object {
        fun of(score: Int): RiskLevel = when {
            score >= 85 -> CRITICAL
            score >= 68 -> HIGH
            score >= 48 -> MEDIUM
            else -> LOW
        }
    }
}

enum class DistanceLevel(val rawValue: String) {
    FAR("far"),
    NEAR("near"),
    CLOSE("close"),
    VERY_CLOSE("very_close")
}

/**
 * A detection enriched with the safety metadata used for live guidance.
 *
 * Android phones have no LiDAR, so `distanceMeters` is only set by tests; the rules
 * then estimate distance from the box like the backend does.
 */
data class SceneDetection(
    val label: String,
    val koreanLabel: String,
    val confidence: Double,
    val bbox: BoundingBox,
    val frameWidth: Double,
    val position: HorizontalPosition,
    val areaRatio: Double,
    /** How low the box bottom sits in the frame (0...1); lower boxes are usually closer. */
    val verticalRatio: Double,
    val distanceMeters: Double? = null,
    val approaching: Boolean = false,
    val growthRatio: Double? = null,
    val frontDangerZone: Boolean = false,
    val riskScore: Int = 0,
    val riskLevel: RiskLevel = RiskLevel.LOW
)

/** Rounds half to even like Python's `round` and Swift's `.toNearestOrEven`. */
internal fun roundHalfEven(value: Double, places: Int): Double {
    var scale = 1.0
    repeat(places) { scale *= 10 }
    return Math.rint(value * scale) / scale
}
