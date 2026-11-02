package com.kojingapla.glass

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * On-device text reading: frame quality, line ordering, aiming help and number safety.
 *
 * Port of iOS `OCRFrameAnalyzer.swift` and `TextStabilityTracker.swift`. Regions are
 * normalized to the upright frame with the origin at the top-left (iOS Vision uses the
 * bottom-left, so "up" and "down" are flipped relative to the Swift code).
 */

/** One recognized line in the upright frame. */
data class OcrLine(val text: String, val box: RectRatio, val confidence: Float)

data class OcrFrameAnalysis(
    val textRegion: RectRatio?,
    val confidence: Float,
    val blurScore: Double,
    val movementScore: Double,
    /** Mean luma (0-255) of the whole frame. */
    val brightness: Double,
    /** Share of the text region (0-1) that is blown out to white, as glare on glossy paper. */
    val glareRatio: Double,
    val timestampMs: Long,
    /** Lines confident enough to read aloud, in reading order. */
    val readableLines: List<String>
) {
    val readableText: String get() = readableLines.joinToString(" ")
}

/** A close reading of the text region, made safe to speak. */
data class DetailRead(
    /** Lines in reading order. */
    val lines: List<String>,
    /** Lines the recognizer saw but could not read. */
    val missedLines: Int,
    /** Lines with digits left out because they were not certain. */
    val droppedNumberLines: Int
) {
    /**
     * A misread digit can change a dose or a phone number, so a line with digits is kept only
     * when `previous`, a read of the next or last frame, has the very same line.
     */
    fun confirmedBy(previous: DetailRead): DetailRead {
        val earlier = previous.lines.map(::normalized).toSet()
        val kept = lines.filter { !hasDigit(it) || normalized(it) in earlier }
        return DetailRead(kept, missedLines, droppedNumberLines + lines.size - kept.size)
    }

    companion object {
        val EMPTY = DetailRead(emptyList(), 0, 0)

        fun hasDigit(text: String): Boolean = text.any { it.isDigit() }

        private fun normalized(text: String): String = text.filterNot { it.isWhitespace() }
    }
}

object OcrText {
    /** Same cutoff the backend used for PaddleOCR (`OCR_MIN_CONFIDENCE`). */
    const val MINIMUM_READABLE_CONFIDENCE = 0.5f

    /** Lines between this and the readable cutoff are text seen but not read; lower is mostly texture. */
    const val MINIMUM_MISSED_CONFIDENCE = 0.3f

    /** Lines with digits need a high score before they are read at all. */
    const val MINIMUM_NUMBER_CONFIDENCE = 0.8f

    /** Lines this confident count as text when deciding where the text is. */
    const val MINIMUM_REGION_CONFIDENCE = 0.45f

    /** Orders confident lines top-to-bottom, then left-to-right within a row. */
    fun readableLines(lines: List<OcrLine>, minimumNumberConfidence: Float = 0f): List<String> {
        val readable = lines.mapNotNull { line ->
            val text = line.text.trim()
            val confident = line.confidence >= MINIMUM_READABLE_CONFIDENCE &&
                (line.confidence >= minimumNumberConfidence || !DetailRead.hasDigit(text))
            if (confident && text.isNotEmpty()) line.copy(text = text) else null
        }

        val rows = ArrayList<MutableList<OcrLine>>()
        for (line in readable.sortedBy { it.box.centerY }) {
            val rowLine = rows.lastOrNull()?.first()
            if (rowLine != null && abs(rowLine.box.centerY - line.box.centerY) < min(rowLine.box.height, line.box.height) / 2) {
                rows.last().add(line)
            } else {
                rows.add(mutableListOf(line))
            }
        }
        return rows.flatMap { row -> row.sortedBy { it.box.left }.map { it.text } }
    }

    /** Union of the lines confident enough to count as text, or null without any. */
    fun textRegion(lines: List<OcrLine>): RectRatio? =
        lines.filter { it.confidence >= MINIMUM_REGION_CONFIDENCE }
            .map { it.box }
            .reduceOrNull { a, b -> RectRatio(min(a.left, b.left), min(a.top, b.top), max(a.right, b.right), max(a.bottom, b.bottom)) }

    fun detailRead(lines: List<OcrLine>): DetailRead {
        val missed = lines.count { it.confidence >= MINIMUM_MISSED_CONFIDENCE && it.confidence < MINIMUM_READABLE_CONFIDENCE }
        val uncertainNumbers = lines.count {
            it.confidence >= MINIMUM_READABLE_CONFIDENCE && it.confidence < MINIMUM_NUMBER_CONFIDENCE && DetailRead.hasDigit(it.text)
        }
        return DetailRead(readableLines(lines, MINIMUM_NUMBER_CONFIDENCE), missed, uncertainNumbers)
    }
}

/**
 * A coarse grid of luma values sampled from the camera's sensor-oriented Y plane,
 * row by row, `COLUMNS` per row.
 */
class LumaSample(val values: DoubleArray) {
    val brightness: Double get() = if (values.isEmpty()) 0.0 else values.average()

    /** Mean absolute difference between neighbors; low on blurry frames. */
    fun blurScore(): Double {
        if (values.isEmpty()) return 0.0
        var edgeEnergy = 0.0
        var comparisons = 0
        for (index in values.indices) {
            if (index % COLUMNS != COLUMNS - 1 && index + 1 < values.size) {
                edgeEnergy += abs(values[index] - values[index + 1])
                comparisons += 1
            }
            val lower = index + COLUMNS
            if (lower < values.size) {
                edgeEnergy += abs(values[index] - values[lower])
                comparisons += 1
            }
        }
        return if (comparisons == 0) 0.0 else edgeEnergy / comparisons
    }

    /** Mean absolute change since `previous`; high while the phone moves. */
    fun movementScore(previous: LumaSample?): Double {
        if (previous == null || previous.values.size != values.size || values.isEmpty()) return 0.0
        var total = 0.0
        for (index in values.indices) total += abs(previous.values[index] - values[index])
        return total / values.size
    }

    /**
     * Share of the samples inside `region` (upright, top-left origin) that are saturated.
     * `rotationDegrees` turns the sensor image upright, as CameraX reports it.
     */
    fun glareRatio(region: RectRatio, rotationDegrees: Int): Double {
        val rows = values.size / COLUMNS
        if (rows == 0) return 0.0
        // A little margin: glare can wash out the text next to the lines still found.
        val left = region.left - 0.05f
        val top = region.top - 0.05f
        val right = region.right + 0.05f
        val bottom = region.bottom + 0.05f
        var inside = 0
        var saturated = 0
        for (index in values.indices) {
            val sensorX = ((index % COLUMNS) + 0.5) / COLUMNS
            val sensorY = ((index / COLUMNS) + 0.5) / rows
            val (x, y) = when (((rotationDegrees % 360) + 360) % 360) {
                90 -> (1 - sensorY) to sensorX
                180 -> (1 - sensorX) to (1 - sensorY)
                270 -> sensorY to (1 - sensorX)
                else -> sensorX to sensorY
            }
            if (x < left || x > right || y < top || y > bottom) continue
            inside += 1
            if (values[index] >= 250) saturated += 1
        }
        return if (inside == 0) 0.0 else saturated.toDouble() / inside
    }

    companion object {
        const val COLUMNS = 24
        const val ROWS = 32
    }
}

data class TextStabilityDecision(
    val status: LiveOcrStatus,
    val shouldRunFullOcr: Boolean,
    /** Spoken help for aiming the camera, already spaced out; null when nothing should be said. */
    val hint: TextAimingHint? = null
)

/** Tells users who cannot see the screen how to move the phone so the text fits the frame. */
enum class TextAimingHint(val message: String) {
    MOVE_LEFT("휴대폰을 왼쪽으로 조금 옮겨 주세요."),
    MOVE_RIGHT("휴대폰을 오른쪽으로 조금 옮겨 주세요."),
    MOVE_UP("휴대폰을 위쪽으로 조금 옮겨 주세요."),
    MOVE_DOWN("휴대폰을 아래쪽으로 조금 옮겨 주세요."),
    MOVE_FARTHER("휴대폰을 조금 멀리 떼 주세요."),
    MOVE_CLOSER("휴대폰을 글자에 더 가까이 대 주세요."),
    GLARE("빛이 반사돼요. 휴대폰을 조금 기울여 주세요.")
}

class TextStabilityTracker {
    private var detectedStartMs: Long? = null
    private var stableStartMs: Long? = null
    private var smoothedRegion: RectRatio? = null
    private var hintCandidate: TextAimingHint? = null
    private var hintCandidateSinceMs = 0L
    private var lastHintMs = Long.MIN_VALUE / 2

    fun reset() {
        detectedStartMs = null
        stableStartMs = null
        smoothedRegion = null
        hintCandidate = null
    }

    fun update(analysis: OcrFrameAnalysis): TextStabilityDecision {
        val region = analysis.textRegion ?: run {
            reset()
            return TextStabilityDecision(LiveOcrStatus.SEARCHING, false)
        }
        val now = analysis.timestampMs
        if (detectedStartMs == null) detectedStartMs = now

        val aimingHint = aimingHint(region, analysis.glareRatio)
        val readableCandidate = aimingHint == null &&
            analysis.blurScore >= MINIMUM_BLUR_SCORE &&
            analysis.movementScore <= MAXIMUM_MOVEMENT_SCORE &&
            analysis.confidence >= OcrText.MINIMUM_REGION_CONFIDENCE

        if (!readableCandidate) {
            stableStartMs = null
            smoothedRegion = smooth(region, smoothedRegion)

            val detectedDuration = now - (detectedStartMs ?: now)
            if (detectedDuration >= PERSISTENT_TEXT_FALLBACK_MS) {
                detectedStartMs = null
                hintCandidate = null
                return TextStabilityDecision(LiveOcrStatus.READING, true)
            }
            return TextStabilityDecision(LiveOcrStatus.DETECTED, false, spokenHint(aimingHint, now))
        }
        hintCandidate = null

        val sameTarget = smoothedRegion?.let { intersectionOverUnion(it, region) >= 0.55f } ?: true
        smoothedRegion = smooth(region, if (sameTarget) smoothedRegion else null)
        if (!sameTarget || stableStartMs == null) stableStartMs = now

        val stableDuration = now - (stableStartMs ?: now)
        if (stableDuration >= REQUIRED_STABLE_MS) {
            detectedStartMs = null
            stableStartMs = null
            return TextStabilityDecision(LiveOcrStatus.READING, true)
        }
        return TextStabilityDecision(LiveOcrStatus.STABILIZING, false)
    }

    /** `hint` once it has lasted `HINT_DELAY_MS`, at most once every `HINT_INTERVAL_MS`. */
    private fun spokenHint(hint: TextAimingHint?, now: Long): TextAimingHint? {
        if (hint == null) {
            hintCandidate = null
            return null
        }
        if (hintCandidate != hint) {
            hintCandidate = hint
            hintCandidateSinceMs = now
            return null
        }
        if (now - hintCandidateSinceMs < HINT_DELAY_MS || now - lastHintMs < HINT_INTERVAL_MS) return null

        lastHintMs = now
        return hint
    }

    private fun smooth(region: RectRatio, previous: RectRatio?): RectRatio {
        previous ?: return region
        val weight = 0.65f
        return RectRatio(
            left = previous.left * weight + region.left * (1 - weight),
            top = previous.top * weight + region.top * (1 - weight),
            right = previous.right * weight + region.right * (1 - weight),
            bottom = previous.bottom * weight + region.bottom * (1 - weight)
        )
    }

    private fun intersectionOverUnion(first: RectRatio, second: RectRatio): Float {
        val left = max(first.left, second.left)
        val top = max(first.top, second.top)
        val right = min(first.right, second.right)
        val bottom = min(first.bottom, second.bottom)
        if (right <= left || bottom <= top) return 0f

        val intersection = (right - left) * (bottom - top)
        val union = first.width * first.height + second.width * second.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    companion object {
        private const val REQUIRED_STABLE_MS = 750L

        /**
         * Text that never becomes readable is read anyway after this long. Long enough for a
         * spoken hint and the user's move, since partial text is better than none.
         */
        private const val PERSISTENT_TEXT_FALLBACK_MS = 3_000L
        private const val MINIMUM_REGION_AREA = 0.035f
        private const val MAXIMUM_CENTER_DISTANCE = 0.28f

        /** Text within this distance of the frame's edge is likely cut off. */
        private const val EDGE_MARGIN = 0.02f
        private const val MAXIMUM_GLARE_RATIO = 0.05
        private const val MINIMUM_BLUR_SCORE = 5.0
        private const val MAXIMUM_MOVEMENT_SCORE = 13.0

        /** A problem must last this long before it is spoken, so passing frames stay quiet. */
        private const val HINT_DELAY_MS = 500L
        private const val HINT_INTERVAL_MS = 2_000L

        /** What keeps `region` (upright, top-left origin) from being read well, most important first. */
        fun aimingHint(region: RectRatio, glareRatio: Double): TextAimingHint? {
            val cutLeft = region.left <= EDGE_MARGIN
            val cutRight = region.right >= 1 - EDGE_MARGIN
            val cutTop = region.top <= EDGE_MARGIN
            val cutBottom = region.bottom >= 1 - EDGE_MARGIN
            if ((cutLeft && cutRight) || (cutTop && cutBottom)) return TextAimingHint.MOVE_FARTHER
            if (cutLeft) return TextAimingHint.MOVE_LEFT
            if (cutRight) return TextAimingHint.MOVE_RIGHT
            if (cutTop) return TextAimingHint.MOVE_UP
            if (cutBottom) return TextAimingHint.MOVE_DOWN
            if (region.width * region.height < MINIMUM_REGION_AREA) return TextAimingHint.MOVE_CLOSER

            val dx = region.centerX - 0.5f
            val dy = region.centerY - 0.5f
            if (sqrt(dx * dx + dy * dy) > MAXIMUM_CENTER_DISTANCE) {
                if (abs(dx) >= abs(dy)) return if (dx < 0) TextAimingHint.MOVE_LEFT else TextAimingHint.MOVE_RIGHT
                return if (dy < 0) TextAimingHint.MOVE_UP else TextAimingHint.MOVE_DOWN
            }
            return if (glareRatio >= MAXIMUM_GLARE_RATIO) TextAimingHint.GLARE else null
        }
    }
}
