package com.kojingapla.glass

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Reads the pedestrian signal across the street and says when it changes.
 *
 * Follows the two-stage approach of earlier pedestrian-signal aids: a detector finds the
 * light, then the lit lamp decides its state (YOLO plus color rules in Kyungwoon University's
 * 2023 YOLOv4-tiny system and the minji-o-j "system-for-visually-impaired" project; a CNN in
 * LYTNet, Yu, Lee & Kim, CAIP 2019; Oko on iOS). Korean pedestrian signals are a vertical
 * box with a red standing figure on top and a green walking figure below, while vehicle
 * signals are horizontal, so a tall box with its top or bottom lamp lit is read as one.
 *
 * Safety rules, since a wrong "green" is the worst mistake this app can make:
 * - a state is spoken only after several frames agree, and only when it changes;
 * - green is never "go": the user is told to listen for cars before crossing;
 * - a green first seen mid-phase or flashing means "wait for the next one", because
 *   its remaining time is unknown;
 * - anything unclear is not spoken at all.
 */
enum class SignalLamp {
    RED,
    GREEN,
    /** A pedestrian signal with neither lamp lit, as between flashes. */
    DARK,
    UNKNOWN
}

object PedestrianSignalReader {
    /** Pedestrian signal boxes are about twice as tall as wide; vehicle signals are wide. */
    const val MINIMUM_ASPECT = 1.6
    const val MINIMUM_CONFIDENCE = 0.35

    /** Smaller boxes are too far or too few pixels to read the lamp reliably. */
    const val MINIMUM_HEIGHT_RATIO = 0.025

    /**
     * Only signals toward the street the user faces. Standing at a crossing, the signal on
     * the far side is often well off-center (a third of the way in, in the dataset photo the
     * tests use), so this is wide; the announcement is panned toward the signal it read.
     */
    const val MAXIMUM_CENTER_OFFSET = 0.4

    /** Share of a lamp's pixels that must glow in its color for the lamp to count as lit. */
    const val LIT_SHARE = 0.06

    /** How much brighter in its color the lit half must be than the other half. */
    const val DOMINANCE = 3.0

    /** The signal the user is facing: tall, close to the middle, and the largest of those. */
    fun mainSignal(detections: List<RawDetection>, frameWidth: Int, frameHeight: Int): RawDetection? =
        detections
            .filter { it.label == "traffic_light" && it.confidence >= MINIMUM_CONFIDENCE }
            .filter { detection ->
                val box = detection.bbox
                val width = box.x2 - box.x1
                val height = box.y2 - box.y1
                width > 0 && height / width >= MINIMUM_ASPECT && height >= frameHeight * MINIMUM_HEIGHT_RATIO &&
                    abs(box.centerX / frameWidth - 0.5) <= MAXIMUM_CENTER_OFFSET
            }
            .maxByOrNull { it.bbox.area }

    /** Which lamp of the signal in `box` is lit, from ARGB `pixels` of a `frameWidth` wide frame. */
    fun read(pixels: IntArray, frameWidth: Int, frameHeight: Int, box: BoundingBox): SignalLamp {
        val left = max(0, box.x1.toInt())
        val right = min(frameWidth, box.x2.toInt())
        val top = max(0, box.y1.toInt())
        val bottom = min(frameHeight, box.y2.toInt())
        if (right - left < 3 || bottom - top < 6) return SignalLamp.UNKNOWN

        val middle = (top + bottom) / 2
        val redOnTop = glowShare(pixels, frameWidth, left, right, top, middle, ::isRedGlow)
        val greenBelow = glowShare(pixels, frameWidth, left, right, middle, bottom, ::isGreenGlow)
        val redBelow = glowShare(pixels, frameWidth, left, right, middle, bottom, ::isRedGlow)
        val greenOnTop = glowShare(pixels, frameWidth, left, right, top, middle, ::isGreenGlow)

        // A lit lamp's halo spills into the other half at night, so a lamp counts only when it
        // clearly outshines the same color in the other half.
        val red = redOnTop >= LIT_SHARE && redOnTop >= redBelow * DOMINANCE
        val green = greenBelow >= LIT_SHARE && greenBelow >= greenOnTop * DOMINANCE
        // A color glowing mostly in the wrong half is not the two-lamp signal we expect.
        val misplaced = (redBelow >= LIT_SHARE && !red) || (greenOnTop >= LIT_SHARE && !green)
        return when {
            misplaced -> SignalLamp.UNKNOWN
            red && !green -> SignalLamp.RED
            green && !red -> SignalLamp.GREEN
            !red && !green && max(redOnTop, redBelow) < LIT_SHARE / 3 && max(greenBelow, greenOnTop) < LIT_SHARE / 3 ->
                SignalLamp.DARK
            else -> SignalLamp.UNKNOWN
        }
    }

    private fun glowShare(
        pixels: IntArray,
        frameWidth: Int,
        left: Int,
        right: Int,
        top: Int,
        bottom: Int,
        glows: (Int) -> Boolean
    ): Double {
        var lit = 0
        var total = 0
        for (y in top until bottom) {
            val row = y * frameWidth
            for (x in left until right) {
                total += 1
                if (glows(pixels[row + x])) lit += 1
            }
        }
        return if (total == 0) 0.0 else lit.toDouble() / total
    }

    /** A bright, saturated red, like a lit red lamp or its halo at night. */
    internal fun isRedGlow(pixel: Int): Boolean {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        return r >= 150 && r >= g * 1.7 && r >= b * 1.5
    }

    /** A bright green to cyan-green: Korean walk signals lean toward blue-green. */
    internal fun isGreenGlow(pixel: Int): Boolean {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        return g >= 130 && g >= r * 1.5 && g >= b * 0.8
    }
}

/** A spoken change of the pedestrian signal, coming from the signal's direction. */
data class SignalAnnouncement(val message: String, val pan: Float)

/**
 * Turns per-frame lamp readings into rare, safe announcements. Times are in seconds.
 * Call `update` on every analyzed live frame, with `lamp` null when no signal is in view.
 */
class PedestrianSignalTracker {
    private class Reading(val time: Double, val lamp: SignalLamp)

    private val readings = ArrayDeque<Reading>()
    private var announced: SignalLamp? = null
    private var announcedFlashing = false
    /** The last confirmed state, to tell a fresh green (seen turning from red) from one already running. */
    private var lastConfirmed: SignalLamp? = null
    private var lastSeenAt = Double.NEGATIVE_INFINITY

    fun reset() {
        readings.clear()
        announced = null
        announcedFlashing = false
        lastConfirmed = null
        lastSeenAt = Double.NEGATIVE_INFINITY
    }

    fun update(lamp: SignalLamp?, pan: Float, now: Double): SignalAnnouncement? {
        if (lamp == null) {
            // Out of view for a while: start over, so a new crossing is judged afresh.
            if (now - lastSeenAt > FORGET_AFTER) reset()
            return null
        }
        lastSeenAt = now
        readings.addLast(Reading(now, lamp))
        while (readings.isNotEmpty() && now - readings.first().time > WINDOW) readings.removeFirst()

        if (isFlashingGreen()) {
            if (announcedFlashing) return null
            announcedFlashing = true
            announced = SignalLamp.GREEN
            lastConfirmed = SignalLamp.GREEN
            return SignalAnnouncement("보행 신호 초록불이 깜빡여요. 건너지 말고 다음 신호를 기다려 주세요.", pan)
        }

        val confirmed = confirmedLamp(now) ?: return null
        val previous = lastConfirmed
        lastConfirmed = confirmed
        if (confirmed == announced) return null
        announced = confirmed
        announcedFlashing = false

        val message = when (confirmed) {
            SignalLamp.RED -> "보행 신호가 빨간불이에요. 기다려 주세요."
            SignalLamp.GREEN -> if (previous == SignalLamp.RED) {
                "보행 신호가 초록불로 바뀌었어요. 차 소리를 확인하고 건너세요."
            } else {
                "보행 신호가 초록불이에요. 언제 켜졌는지 몰라요. 다음 신호를 기다리면 더 안전해요."
            }
            else -> return null
        }
        return SignalAnnouncement(message, pan)
    }

    /** Red or green when the last few readings agree and span long enough to be no glitch. */
    private fun confirmedLamp(now: Double): SignalLamp? {
        val recent = readings.filter { now - it.time <= CONFIRM_WINDOW }
        if (recent.size < CONFIRM_COUNT) return null
        val lamp = recent.last().lamp
        if (lamp != SignalLamp.RED && lamp != SignalLamp.GREEN) return null
        val agreeing = recent.filter { it.lamp == lamp }
        val span = agreeing.last().time - agreeing.first().time
        return if (agreeing.size >= recent.size - 1 && agreeing.size >= CONFIRM_COUNT && span >= MINIMUM_SPAN) lamp else null
    }

    /** Green and dark readings switching back and forth, as a walk signal flashes before red. */
    private fun isFlashingGreen(): Boolean {
        var switches = 0
        var previous: SignalLamp? = null
        for (reading in readings) {
            val lamp = reading.lamp
            if (lamp != SignalLamp.GREEN && lamp != SignalLamp.DARK) continue
            if (previous != null && lamp != previous) switches += 1
            previous = lamp
        }
        return switches >= FLASH_SWITCHES && readings.any { it.lamp == SignalLamp.GREEN }
    }

    private companion object {
        /** Readings kept for flash detection. */
        const val WINDOW = 3.0
        const val CONFIRM_WINDOW = 1.2
        const val CONFIRM_COUNT = 3
        const val MINIMUM_SPAN = 0.4

        /** Green-dark-green-dark: three switches is two flashes. */
        const val FLASH_SWITCHES = 3
        const val FORGET_AFTER = 5.0
    }
}
