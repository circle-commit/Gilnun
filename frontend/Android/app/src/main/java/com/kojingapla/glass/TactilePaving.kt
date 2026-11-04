package com.kojingapla.glass

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Finds the yellow tactile paving (점자블록) on the ground ahead and says where it runs.
 *
 * Korean tactile paving is vivid yellow, with domes (stop/attention) or bars (go straight).
 * The ground half of the frame is split into a grid; a cell counts when most of it is that
 * yellow and the yellow itself is textured, which the bumps' light and shadow make it.
 * That keeps out faded yellow road paint (low saturation, smooth), yellowish grass (hue)
 * and wide yellow areas (too broad to be a path). Rows of cells from the bottom up give
 * where the path is at the user's feet and where it heads.
 *
 * Experimental: tested on real sidewalk photos without paving (no false paths) and on
 * paving drawn into such photos, not yet on real paving. A trained segmentation model (AI
 * Hub's sidewalk surface masks label tactile blocks) is the better long-term path.
 */
data class TactilePath(
    /** Path center at the user's feet, 0 (left edge) to 1 (right edge). */
    val nearX: Double,
    /** Path center farther ahead. */
    val farX: Double,
    /** Grid rows the path crosses, out of `TactilePavingDetector.ROWS`. */
    val rows: Int
) {
    val position: HorizontalPosition
        get() = when {
            nearX < 0.36 -> HorizontalPosition.LEFT
            nearX > 0.64 -> HorizontalPosition.RIGHT
            else -> HorizontalPosition.CENTER
        }

    /** Which way the path bends ahead, or null when it runs straight. */
    val turn: HorizontalPosition?
        get() = when {
            farX - nearX > TURN -> HorizontalPosition.RIGHT
            nearX - farX > TURN -> HorizontalPosition.LEFT
            else -> null
        }

    private companion object {
        const val TURN = 0.15
    }
}

object TactilePavingDetector {
    const val COLUMNS = 12
    const val ROWS = 8

    /** Only the lower half of the frame is ground near enough to follow. */
    const val GROUND_TOP = 0.5
    const val YELLOW_SHARE = 0.45

    /** Mean luma step between neighboring yellow pixels; flat paint is smoother than bumps. */
    const val MINIMUM_TEXTURE = 3.0

    /** A path is a strip: wider runs of yellow cells are a yellow area, not paving to follow. */
    const val MAXIMUM_RUN = 6
    const val MINIMUM_ROWS = 3

    /** Every second pixel is plenty for color and texture. */
    private const val STEP = 2

    fun find(pixels: IntArray, width: Int, height: Int): TactilePath? {
        if (width < COLUMNS * STEP || height < ROWS * STEP * 2) return null
        val top = (height * GROUND_TOP).toInt()
        val cellWidth = width / COLUMNS
        val cellHeight = (height - top) / ROWS

        // Row 0 is the bottom of the frame, nearest the user.
        val centers = arrayOfNulls<Double>(ROWS)
        for (row in 0 until ROWS) {
            val y0 = height - (row + 1) * cellHeight
            val paving = BooleanArray(COLUMNS) { column ->
                isPavingCell(pixels, width, column * cellWidth, y0, cellWidth, cellHeight)
            }
            centers[row] = longestRun(paving)?.let { (start, end) ->
                if (end - start + 1 > MAXIMUM_RUN) null else (start + end + 1) / 2.0 / COLUMNS
            }
        }

        // The path must start near the feet and continue upward without jumps.
        val present = (0 until ROWS).filter { centers[it] != null }
        if (present.size < MINIMUM_ROWS || present.first() > 2) return null
        for (index in 1 until present.size) {
            val jump = abs(centers[present[index]]!! - centers[present[index - 1]]!!)
            if (present[index] - present[index - 1] > 2 || jump > 3.0 / COLUMNS) return null
        }
        val near = present.take(3).map { centers[it]!! }.average()
        val far = present.takeLast(3).map { centers[it]!! }.average()
        return TactilePath(near, far, present.size)
    }

    private fun isPavingCell(pixels: IntArray, width: Int, x0: Int, y0: Int, cellWidth: Int, cellHeight: Int): Boolean {
        var samples = 0
        var yellow = 0
        var texture = 0.0
        var pairs = 0
        for (y in y0 until y0 + cellHeight - STEP step STEP) {
            for (x in x0 until x0 + cellWidth - STEP step STEP) {
                val pixel = pixels[y * width + x]
                samples += 1
                if (!isPavingYellow(pixel)) continue
                yellow += 1
                val right = pixels[y * width + x + STEP]
                val below = pixels[(y + STEP) * width + x]
                if (isPavingYellow(right)) {
                    texture += abs(luma(pixel) - luma(right)); pairs += 1
                }
                if (isPavingYellow(below)) {
                    texture += abs(luma(pixel) - luma(below)); pairs += 1
                }
            }
        }
        if (samples == 0 || yellow.toDouble() / samples < YELLOW_SHARE || pairs == 0) return false
        return texture / pairs >= MINIMUM_TEXTURE
    }

    /** The saturated yellow of tactile blocks: hue 35-58°, saturation 0.4 or more, not dark. */
    internal fun isPavingYellow(pixel: Int): Boolean {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        val high = max(r, max(g, b))
        val low = min(r, min(g, b))
        if (high < 80 || high != r && high != g) return false
        val chroma = high - low
        if (chroma < high * 0.40) return false
        // Yellow sits between red (0°) and green (120°), with blue the smallest channel.
        if (low != b) return false
        val hue = if (high == r) 60.0 * (g - b) / chroma else 60.0 * (2.0 + (b - r).toDouble() / chroma)
        return hue in 35.0..58.0
    }

    private fun luma(pixel: Int): Double =
        0.299 * ((pixel shr 16) and 0xFF) + 0.587 * ((pixel shr 8) and 0xFF) + 0.114 * (pixel and 0xFF)

    /** First and last column of the longest run of true cells, allowing one-cell gaps. */
    private fun longestRun(cells: BooleanArray): Pair<Int, Int>? {
        var best: Pair<Int, Int>? = null
        var start = -1
        var lastTrue = -1
        for (column in cells.indices) {
            if (!cells[column]) continue
            if (start < 0 || column - lastTrue > 2) start = column
            lastTrue = column
            val length = lastTrue - start
            if (best == null || length > best.second - best.first) best = start to lastTrue
        }
        return best
    }
}

/**
 * Turns per-frame paths into rare sentences: where the paving is when it is found, when it
 * bends ahead, when it is no longer underfoot, and when it ends. Times are in seconds.
 */
class TactilePavingGuide {
    private class Seen(val time: Double, val path: TactilePath?)

    private val recent = ArrayDeque<Seen>()
    private var announced: TactilePath? = null
    private var lastSpokenAt = Double.NEGATIVE_INFINITY
    private var lastSeenAt = Double.NEGATIVE_INFINITY

    fun reset() {
        recent.clear()
        announced = null
        lastSpokenAt = Double.NEGATIVE_INFINITY
        lastSeenAt = Double.NEGATIVE_INFINITY
    }

    fun update(path: TactilePath?, now: Double): SpokenSentence? {
        recent.addLast(Seen(now, path))
        while (recent.size > CONFIRM_FRAMES) recent.removeFirst()
        if (path != null) lastSeenAt = now

        val previous = announced
        if (previous != null && now - lastSeenAt >= LOST_AFTER) {
            announced = null
            return say("점자블록이 더 이상 보이지 않아요.", previous, now)
        }

        // Several of the last frames must agree on where the paving is.
        val found = recent.mapNotNull { it.path }
        if (found.size < CONFIRM_FRAMES - 1 || path == null) return null
        if (found.map { it.position }.distinct().size != 1) return null
        if (now - lastSpokenAt < MINIMUM_GAP) return null

        val turn = found.map { it.turn }.distinct().singleOrNull()
        val message = when {
            previous == null -> "점자블록이 ${path.position.koreanName}에 있어요." +
                (turn?.let { " 앞에서 ${it.koreanName}으로 꺾여요." } ?: "")
            path.position != previous.position -> "점자블록이 ${path.position.koreanName}에 있어요."
            turn != null && previous.turn != turn -> "앞에서 점자블록이 ${turn.koreanName}으로 꺾여요."
            else -> return null
        }
        announced = path.copy(farX = if (turn == null) path.nearX else path.farX)
        return say(message, path, now)
    }

    private fun say(message: String, path: TactilePath, now: Double): SpokenSentence {
        lastSpokenAt = now
        val pan = ((path.nearX - 0.5) * 2).coerceIn(-1.0, 1.0) * SpatialAudio.MAX_PAN
        return SpokenSentence(message, pan.toFloat())
    }

    private companion object {
        const val CONFIRM_FRAMES = 4
        const val MINIMUM_GAP = 4.0
        const val LOST_AFTER = 2.0
    }
}
