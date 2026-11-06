package com.kojingapla.glass

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Runs the ONNX detector on the golden portrait crops and checks that it finds the same
 * objects as the PyTorch checkpoint (iOS `Tests/make_detector_golden.py`), with the same
 * tolerances as the iOS Core ML test.
 */
class DetectorTest {
    @Test
    fun matchesPyTorchReference() {
        val modelPath = System.getProperty("gilnun.model") ?: fail("run through tests/run_tests.sh (gilnun.model is not set)")
        val result = compare(modelPath)
        println("detector: ${result.summary}")
        if (result.largestConfidenceGap > 0.05) result.failures.add("confidence differs by ${result.largestConfidenceGap}")
        assertTrue(result.failures.isEmpty(), result.failures.joinToString("\n"))
    }

    /**
     * The int8 "speed first" model trades a little accuracy: calibrated on the repository's
     * few photos it may miss faint objects near the threshold and shift confidences a bit
     * more, so it gets looser bounds. These pin today's quality so it does not get worse.
     */
    @Test
    fun fastModelStaysClose() {
        val modelPath = System.getProperty("gilnun.fastModel") ?: fail("run through tests/run_tests.sh (gilnun.fastModel is not set)")
        val result = compare(modelPath, borderline = 0.35)
        println("fast detector: ${result.summary}")
        assertTrue(result.matched >= result.expected - 2, "fast model matches ${result.matched}/${result.expected}")
        assertTrue(result.largestConfidenceGap <= 0.1, "fast model confidence differs by ${result.largestConfidenceGap}")
        assertTrue(result.failures.isEmpty(), result.failures.joinToString("\n"))
    }

    private class Comparison {
        val failures = ArrayList<String>()
        var matched = 0
        var expected = 0
        var largestConfidenceGap = 0.0
        var smallestIoU = 1.0
        var slowest = 0L
        val summary: String
            get() = "matched $matched/$expected PyTorch detections, max confidence gap " +
                "%.4f, min IoU %.3f, slowest frame %d ms on this machine".format(largestConfidenceGap, smallestIoU, slowest)
    }

    /** Objects with a reference confidence below `borderline` may flip between runtimes. */
    private fun compare(modelPath: String, borderline: Double? = null): Comparison {
        val golden = JSONArray(File(System.getProperty("gilnun.golden")).readText())
        val repoRoot = File(System.getProperty("gilnun.repoRoot"))
        val result = Comparison()

        ObjectDetector(File(modelPath).readBytes()).use { detector ->
            val flipBelow = borderline ?: (detector.confidenceThreshold + 0.05)
            for (index in 0 until golden.length()) {
                val case = golden.getJSONObject(index)
                val name = case.getString("image")
                val crop = case.getJSONArray("crop")
                val image = ImageIO.read(File(repoRoot, name))
                val width = crop.getInt(2)
                val height = crop.getInt(3)
                val pixels = image.getRGB(crop.getInt(0), crop.getInt(1), width, height, null, 0, width)

                val started = System.nanoTime()
                val detections = detector.detect(pixels, width, height)
                result.slowest = max(result.slowest, (System.nanoTime() - started) / 1_000_000)
                val used = HashSet<Int>()

                val references = case.getJSONArray("detections").let { array -> (0 until array.length()).map { array.getJSONObject(it) } }
                for (reference in references) {
                    result.expected += 1
                    val referenceBox = box(reference.getJSONArray("bbox"))
                    val label = reference.getString("label")
                    val confidence = reference.getDouble("confidence")
                    val best = detections.indices
                        .filter { it !in used && detections[it].label == label }
                        .maxByOrNull { detections[it].bbox.intersectionOverUnion(referenceBox) }
                    val iou = best?.let { detections[it].bbox.intersectionOverUnion(referenceBox) } ?: 0.0

                    if (best != null && iou >= 0.8) {
                        used.add(best)
                        result.matched += 1
                        result.largestConfidenceGap = max(result.largestConfidenceGap, abs(detections[best].confidence - confidence))
                        result.smallestIoU = min(result.smallestIoU, iou)
                    } else if (confidence >= flipBelow) {
                        result.failures.add("$name: missed $label ($confidence)")
                    }
                }
                for (other in detections.indices) {
                    if (other !in used && detections[other].confidence >= flipBelow) {
                        result.failures.add("$name: unexpected ${detections[other].label} (${detections[other].confidence})")
                    }
                }
            }
        }
        return result
    }

    @Test
    fun classNamesFromMetadata() {
        val names = ObjectDetector.classNames("{0: 'person', 2: 'bus', 1: 'car'}")
        assertTrue(names == listOf("person", "car", "bus"), "names are ordered by index: $names")
    }

    private fun box(values: JSONArray) = BoundingBox(values.getDouble(0), values.getDouble(1), values.getDouble(2), values.getDouble(3))

}
