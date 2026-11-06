package com.kojingapla.glass

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Runs the bundled sidewalk YOLO model (see `DetectorModel`) on camera frames.
 *
 * Android counterpart of iOS `ObjectDetector.swift`. Frames are letterboxed the way
 * Ultralytics does it (scale to fit with bilinear filtering, centered, gray padding),
 * class-aware NMS follows Ultralytics' `non_max_suppression`, and boxes are mapped back
 * with its `scale_boxes` math, so the results match the PyTorch checkpoint.
 * Pure Kotlin plus ONNX Runtime, so the JVM tests run the same code. Not thread-safe.
 */
class ObjectDetector(modelBytes: ByteArray, threads: Int = 4) : AutoCloseable {
    /**
     * Same as iOS: lower than the backend's 0.35, because for safety recall matters more
     * than precision, and low-confidence objects still need two consecutive frames before
     * they are announced (`GuidanceEventTracker`).
     */
    val confidenceThreshold = 0.25
    val iouThreshold = 0.5
    val maxDetections = 20

    val labels: List<String>
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

        labels = classNames(session.metadata.customMetadata["names"].orEmpty())
        require(labels.isNotEmpty()) { "model metadata has no class names" }
    }

    /**
     * Detects objects in an upright frame given as ARGB pixels (as from `Bitmap.getPixels`).
     * Boxes are in the frame's pixel coordinates (origin top-left), sorted by confidence.
     */
    fun detect(pixels: IntArray, frameWidth: Int, frameHeight: Int): List<RawDetection> {
        if (frameWidth <= 0 || frameHeight <= 0) return emptyList()
        require(pixels.size >= frameWidth * frameHeight) { "pixel buffer is smaller than the frame" }

        val letterbox = Letterbox(frameWidth, frameHeight, inputWidth, inputHeight)
        letterbox.render(pixels, input)

        val output = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(input),
            longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val values = result.get(0) as OnnxTensor
                val shape = values.info.shape
                val buffer = values.floatBuffer
                Prediction(FloatArray(buffer.remaining()).also { buffer.get(it) }, shape[1].toInt(), shape[2].toInt())
            }
        }

        // Ultralytics `scale_boxes`: undo the padding, then the scale, then clip to the frame.
        val gain = letterbox.gain
        val padX = letterbox.padX
        val padY = letterbox.padY
        return suppress(output).map { candidate ->
            val box = candidate.box
            RawDetection(
                label = labels[candidate.classIndex],
                confidence = candidate.score.toDouble(),
                bbox = BoundingBox(
                    x1 = clamp((box.x1 - padX) / gain, frameWidth.toDouble()),
                    y1 = clamp((box.y1 - padY) / gain, frameHeight.toDouble()),
                    x2 = clamp((box.x2 - padX) / gain, frameWidth.toDouble()),
                    y2 = clamp((box.y2 - padY) / gain, frameHeight.toDouble())
                )
            )
        }
    }

    override fun close() {
        session.close()
    }

    /** Raw model output: rows are cx, cy, w, h, then one score per class; columns are anchors. */
    private class Prediction(val values: FloatArray, val rows: Int, val anchors: Int) {
        operator fun get(row: Int, anchor: Int): Float = values[row * anchors + anchor]
    }

    private class Candidate(val box: BoundingBox, val score: Float, val classIndex: Int)

    /** Ultralytics `non_max_suppression` with one label per box and class-aware NMS. */
    private fun suppress(prediction: Prediction): List<Candidate> {
        val classCount = min(labels.size, prediction.rows - 4)
        val candidates = ArrayList<Candidate>()
        for (anchor in 0 until prediction.anchors) {
            var bestClass = 0
            var bestScore = Float.NEGATIVE_INFINITY
            for (classIndex in 0 until classCount) {
                val score = prediction[4 + classIndex, anchor]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = classIndex
                }
            }
            if (bestScore <= confidenceThreshold) continue

            val centerX = prediction[0, anchor].toDouble()
            val centerY = prediction[1, anchor].toDouble()
            val halfWidth = prediction[2, anchor] / 2.0
            val halfHeight = prediction[3, anchor] / 2.0
            val box = BoundingBox(centerX - halfWidth, centerY - halfHeight, centerX + halfWidth, centerY + halfHeight)
            candidates.add(Candidate(box, bestScore, bestClass))
        }

        candidates.sortByDescending { it.score }
        val kept = ArrayList<Candidate>()
        for (candidate in candidates) {
            val overlaps = kept.any {
                it.classIndex == candidate.classIndex && it.box.intersectionOverUnion(candidate.box) > iouThreshold
            }
            if (!overlaps) {
                kept.add(candidate)
                if (kept.size >= maxDetections) break
            }
        }
        return kept
    }

    private fun clamp(value: Double, upperBound: Double): Double = max(0.0, min(upperBound, value))

    companion object {

        /** Reads class names from the export metadata, e.g. "{0: 'person', 1: 'car', ...}". */
        fun classNames(names: String): List<String> =
            Regex("""(\d+):\s*'([^']*)'""").findAll(names)
                .map { it.groupValues[1].toInt() to it.groupValues[2] }
                .sortedBy { it.first }
                .map { it.second }
                .toList()
    }
}
