package com.kojingapla.glass

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Where braille guide blocks (점자블록) are in an upright frame, as a coarse grid.
 * Port of iOS `BrailleMask`: about 24 px square cells on a 1080x1920 portrait frame.
 */
class BrailleMask(val cells: BooleanArray = BooleanArray(COLUMNS * ROWS)) {
    operator fun get(column: Int, row: Int): Boolean = cells[row * COLUMNS + column]

    val isEmpty: Boolean get() = cells.none { it }

    override fun equals(other: Any?): Boolean = other is BrailleMask && cells.contentEquals(other.cells)

    override fun hashCode(): Int = cells.contentHashCode()

    companion object {
        const val COLUMNS = 45
        const val ROWS = 80
        val EMPTY = BrailleMask()
    }
}

/**
 * Runs the bundled braille block segmentation model (`assets/braille_block.onnx`, the same
 * YOLO11n-seg as iOS `BrailleBlockModel.mlpackage`, trained on AI Hub 인도보행 surface
 * polygons) on camera frames. Port of iOS `BrailleBlockSegmenter`.
 *
 * The model returns raw predictions (1x37xN: box, score, 32 mask coefficients) and mask
 * prototypes (1x32xHxW). Overlapping candidates are merged into one mask anyway, so no NMS
 * is needed. Pure Kotlin plus ONNX Runtime, so the JVM tests run the same code. Not thread-safe.
 */
class BrailleBlockSegmenter(modelBytes: ByteArray, threads: Int = 2) : AutoCloseable {
    /** Same threshold the validation numbers were measured at. */
    val confidenceThreshold = 0.25f

    /** One block yields many overlapping candidates; fewer than this missed parts of blocks. */
    val maxCandidates = 50

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val inputWidth: Int
    private val inputHeight: Int
    private val input: FloatArray

    init {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(threads)
        }
        session = environment.createSession(modelBytes, options)
        inputName = session.inputNames.first()
        val shape = (session.inputInfo.getValue(inputName).info as TensorInfo).shape
        require(shape.size == 4 && shape[1] == 3L) { "unexpected model input ${shape.contentToString()}" }
        inputHeight = shape[2].toInt()
        inputWidth = shape[3].toInt()
        input = FloatArray(3 * inputWidth * inputHeight)
    }

    /** Braille blocks in an upright frame given as ARGB pixels. */
    fun segment(pixels: IntArray, frameWidth: Int, frameHeight: Int): BrailleMask {
        if (frameWidth <= 0 || frameHeight <= 0) return BrailleMask.EMPTY
        val letterbox = Letterbox(frameWidth, frameHeight, inputWidth, inputHeight)
        letterbox.render(pixels, input)

        var predictions: Output? = null
        var prototypes: Output? = null
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()))
            .use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    // Tell the outputs apart by rank, like iOS: 1x37xN and 1x32xHxW.
                    for (entry in result) {
                        val value = entry.value as? OnnxTensor ?: continue
                        val shape = value.info.shape
                        val buffer = value.floatBuffer
                        val output = Output(FloatArray(buffer.remaining()).also { buffer.get(it) }, shape)
                        if (shape.size == 3) predictions = output else if (shape.size == 4) prototypes = output
                    }
                }
            }
        val prediction = predictions ?: error("missing prediction output")
        val prototype = prototypes ?: error("missing prototype output")

        val maskCount = prototype.shape[1].toInt()
        val protoHeight = prototype.shape[2].toInt()
        val protoWidth = prototype.shape[3].toInt()
        require(prediction.shape[1].toInt() == 5 + maskCount) { "expected one class and $maskCount mask coefficients" }

        val candidates = candidates(prediction, maskCount)
        if (candidates.isEmpty()) return BrailleMask.EMPTY

        val protoPlane = protoHeight * protoWidth
        val protoScaleX = inputWidth.toDouble() / protoWidth
        val protoScaleY = inputHeight.toDouble() / protoHeight
        fun logit(candidate: Candidate, protoX: Int, protoY: Int): Float {
            val base = protoY * protoWidth + protoX
            var sum = 0f
            for (k in 0 until maskCount) sum += candidate.coefficients[k] * prototype.values[k * protoPlane + base]
            return sum
        }

        val cells = BooleanArray(BrailleMask.COLUMNS * BrailleMask.ROWS)
        for (row in 0 until BrailleMask.ROWS) {
            // Cell center in model input pixels.
            val y = (row + 0.5) / BrailleMask.ROWS * frameHeight * letterbox.gain + letterbox.padY
            // Bilinear upsampling as in `F.interpolate(align_corners=False)`.
            val sourceY = min(max(y / protoScaleY - 0.5, 0.0), (protoHeight - 1).toDouble())
            val y0 = sourceY.toInt()
            val y1 = min(y0 + 1, protoHeight - 1)
            val weightY = (sourceY - y0).toFloat()
            for (column in 0 until BrailleMask.COLUMNS) {
                val x = (column + 0.5) / BrailleMask.COLUMNS * frameWidth * letterbox.gain + letterbox.padX
                val sourceX = min(max(x / protoScaleX - 0.5, 0.0), (protoWidth - 1).toDouble())
                val x0 = sourceX.toInt()
                val x1 = min(x0 + 1, protoWidth - 1)
                val weightX = (sourceX - x0).toFloat()

                // Ultralytics `process_mask`: inside the box and a positive mask logit.
                cells[row * BrailleMask.COLUMNS + column] = candidates.any { candidate ->
                    if (!candidate.contains(x, y)) return@any false
                    val top = (1 - weightX) * logit(candidate, x0, y0) + weightX * logit(candidate, x1, y0)
                    val bottom = (1 - weightX) * logit(candidate, x0, y1) + weightX * logit(candidate, x1, y1)
                    (1 - weightY) * top + weightY * bottom > 0
                }
            }
        }
        return BrailleMask(cells)
    }

    override fun close() {
        session.close()
    }

    private class Output(val values: FloatArray, val shape: LongArray)

    private class Candidate(val score: Float, val x1: Double, val y1: Double, val x2: Double, val y2: Double, val coefficients: FloatArray) {
        fun contains(x: Double, y: Double): Boolean = x >= x1 && x < x2 && y >= y1 && y < y2
    }

    private fun candidates(prediction: Output, maskCount: Int): List<Candidate> {
        val anchors = prediction.shape[2].toInt()
        fun value(channel: Int, anchor: Int) = prediction.values[channel * anchors + anchor]
        return (0 until anchors)
            .filter { value(4, it) >= confidenceThreshold }
            .sortedByDescending { value(4, it) }
            .take(maxCandidates)
            .map { anchor ->
                val width = value(2, anchor).toDouble()
                val height = value(3, anchor).toDouble()
                val centerX = value(0, anchor).toDouble()
                val centerY = value(1, anchor).toDouble()
                Candidate(
                    score = value(4, anchor),
                    x1 = centerX - width / 2,
                    y1 = centerY - height / 2,
                    x2 = centerX + width / 2,
                    y2 = centerY + height / 2,
                    coefficients = FloatArray(maskCount) { value(5 + it, anchor) }
                )
            }
    }

    companion object {
        const val MODEL_ASSET = "braille_block.onnx"
    }
}

/**
 * Decides when to say where the braille blocks are. Port of iOS `BrailleBlockTracker`:
 * speaks when blocks show up in the walking area (the lower part of the frame) for a few
 * frames in a row, and again when they move to another side; stays quiet while they stay put.
 * Times are in seconds.
 */
class BrailleBlockTracker {
    private var candidate: HorizontalPosition? = null
    private var candidateFrames = 0
    private var missedFrames = 0
    private var announced: HorizontalPosition? = null
    private var lastAnnouncement = Double.NEGATIVE_INFINITY

    fun reset() {
        candidate = null
        candidateFrames = 0
        missedFrames = 0
        announced = null
        lastAnnouncement = Double.NEGATIVE_INFINITY
    }

    /**
     * @param canSpeak false while more urgent guidance is being spoken; the announcement then
     *   waits for a later frame.
     * @return the sentence to speak, or null.
     */
    fun update(mask: BrailleMask, now: Double, canSpeak: Boolean): String? {
        val position = position(mask) ?: run {
            missedFrames += 1
            candidate = null
            candidateFrames = 0
            if (missedFrames >= FRAMES_TO_LOSE) announced = null
            return null
        }

        missedFrames = 0
        if (position == candidate) {
            candidateFrames += 1
        } else {
            candidate = position
            candidateFrames = 1
        }

        if (candidateFrames < FRAMES_TO_CONFIRM || position == announced || !canSpeak) return null
        if (announced != null && now - lastAnnouncement < SIDE_CHANGE_INTERVAL) return null
        announced = position
        lastAnnouncement = now
        return message(position)
    }

    companion object {
        /** Rows below this share of the frame height are the ground ahead of the user. */
        const val WALKING_AREA_TOP = 0.4

        /** Blocks must cover this share of the whole frame inside the walking area. */
        const val MINIMUM_SHARE = 0.01

        /** A single frame wrongly shows blocks about 7% of the time on validation crops. */
        const val FRAMES_TO_CONFIRM = 3

        /** About two seconds at five analyses a second before blocks count as gone. */
        const val FRAMES_TO_LOSE = 10
        const val SIDE_CHANGE_INTERVAL = 4.0

        /** Side of the blocks in the walking area, or null when there are too few there. */
        fun position(mask: BrailleMask): HorizontalPosition? = centerColumn(mask)?.let {
            HorizontalPosition.of(it, BrailleMask.COLUMNS.toDouble())
        }

        /** Mean column (in cells, centers at +0.5) of the blocks in the walking area. */
        fun centerColumn(mask: BrailleMask): Double? {
            val firstRow = Math.ceil(BrailleMask.ROWS * WALKING_AREA_TOP).toInt()
            var count = 0
            var columnSum = 0
            for (row in firstRow until BrailleMask.ROWS) {
                for (column in 0 until BrailleMask.COLUMNS) {
                    if (mask[column, row]) {
                        count += 1
                        columnSum += column
                    }
                }
            }
            if (count < MINIMUM_SHARE * BrailleMask.COLUMNS * BrailleMask.ROWS) return null
            return columnSum.toDouble() / count + 0.5
        }

        fun message(position: HorizontalPosition): String = when (position) {
            HorizontalPosition.LEFT -> "왼쪽에 점자블록이 있어요."
            HorizontalPosition.CENTER -> "앞에 점자블록이 있어요."
            HorizontalPosition.RIGHT -> "오른쪽에 점자블록이 있어요."
        }
    }
}
