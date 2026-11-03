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
        val golden = JSONArray(File(System.getProperty("gilnun.golden")).readText())
        val repoRoot = File(System.getProperty("gilnun.repoRoot"))
        val failures = ArrayList<String>()
        var matched = 0
        var expected = 0
        var largestConfidenceGap = 0.0
        var smallestIoU = 1.0
        var slowest = 0L

        ObjectDetector(File(modelPath).readBytes()).use { detector ->
            // Objects this close to the threshold may legitimately flip between runtimes.
            val borderline = detector.confidenceThreshold + 0.05
            for (index in 0 until golden.length()) {
                val case = golden.getJSONObject(index)
                val name = case.getString("image")
                val crop = case.getJSONArray("crop")
                val image = ImageIO.read(File(repoRoot, name))
                val x = crop.getInt(0)
                val y = crop.getInt(1)
                val width = crop.getInt(2)
                val height = crop.getInt(3)
                val pixels = image.getRGB(x, y, width, height, null, 0, width)

                val started = System.nanoTime()
                val detections = detector.detect(pixels, width, height)
                slowest = max(slowest, (System.nanoTime() - started) / 1_000_000)
                val used = HashSet<Int>()

                for (reference in case.getJSONArray("detections").let { array -> (0 until array.length()).map { array.getJSONObject(it) } }) {
                    expected += 1
                    val referenceBox = box(reference.getJSONArray("bbox"))
                    val label = reference.getString("label")
                    val confidence = reference.getDouble("confidence")
                    val best = detections.indices
                        .filter { it !in used && detections[it].label == label }
                        .maxByOrNull { detections[it].bbox.intersectionOverUnion(referenceBox) }
                    val iou = best?.let { detections[it].bbox.intersectionOverUnion(referenceBox) } ?: 0.0

                    if (best != null && iou >= 0.8) {
                        used.add(best)
                        matched += 1
                        largestConfidenceGap = max(largestConfidenceGap, abs(detections[best].confidence - confidence))
                        smallestIoU = min(smallestIoU, iou)
                    } else if (confidence >= borderline) {
                        failures.add("$name: missed $label ($confidence)")
                    }
                }
                for (index2 in detections.indices) {
                    if (index2 !in used && detections[index2].confidence >= borderline) {
                        failures.add("$name: unexpected ${detections[index2].label} (${detections[index2].confidence})")
                    }
                }
            }
        }

        println(
            "detector: matched $matched/$expected PyTorch detections, max confidence gap " +
                "%.4f, min IoU %.3f, slowest frame %d ms on this machine".format(largestConfidenceGap, smallestIoU, slowest)
        )
        if (largestConfidenceGap > 0.05) failures.add("confidence differs by $largestConfidenceGap")
        assertTrue(failures.isEmpty(), failures.joinToString("\n"))
    }

    @Test
    fun classNamesFromMetadata() {
        val names = ObjectDetector.classNames("{0: 'person', 2: 'bus', 1: 'car'}")
        assertTrue(names == listOf("person", "car", "bus"), "names are ordered by index: $names")
    }

    private fun box(values: JSONArray) = BoundingBox(values.getDouble(0), values.getDouble(1), values.getDouble(2), values.getDouble(3))

}
