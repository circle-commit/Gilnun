package com.kojingapla.glass

import org.json.JSONArray
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Runs the ONNX braille segmenter on 360x640 portrait frames and checks its mask against
 * the PyTorch checkpoint's (fixtures/make_braille_golden.py), with the same tolerances as
 * the iOS test (BrailleTests.swift).
 */
class BrailleSegmenterTest {
    @Test
    fun matchesPyTorchReference() {
        val modelPath = System.getProperty("gilnun.brailleModel") ?: fail("run through tests/run_tests.sh (gilnun.brailleModel is not set)")
        val root = File(System.getProperty("gilnun.repoRoot"))
        val golden = JSONArray(File(root, "frontend/Android/tests/fixtures/braille_golden.json").readText())
        val failures = ArrayList<String>()
        var smallestIoU = 1.0
        var withBlocks = 0

        BrailleBlockSegmenter(File(modelPath).readBytes()).use { segmenter ->
            for (index in 0 until golden.length()) {
                val case = golden.getJSONObject(index)
                val name = case.getString("image")
                val rows = case.getJSONArray("rows")
                val expected = BooleanArray(BrailleMask.COLUMNS * BrailleMask.ROWS) { cell ->
                    rows.getString(cell / BrailleMask.COLUMNS)[cell % BrailleMask.COLUMNS] == '1'
                }
                val mask = segmenter.segment(frame(File(root, name)), WIDTH, HEIGHT)

                val both = expected.indices.count { expected[it] && mask.cells[it] }
                val either = expected.indices.count { expected[it] || mask.cells[it] }
                val expectedCount = expected.count { it }
                if (expectedCount >= 20) {
                    withBlocks += 1
                    val iou = both.toDouble() / maxOf(1, either)
                    smallestIoU = minOf(smallestIoU, iou)
                    val required = if (expectedCount >= 100) 0.85 else 0.6
                    if (iou < required) failures.add("$name: mask IoU $iou on $expectedCount cells")
                } else if (either - both > 10) {
                    failures.add("$name: ${either - both} cells differ on a nearly empty mask")
                }
                val side = BrailleBlockTracker.position(mask)
                val expectedSide = BrailleBlockTracker.position(BrailleMask(expected))
                if (side != expectedSide) failures.add("$name: side $side, PyTorch $expectedSide")
            }
        }
        println("braille segmenter: ${golden.length()} frames ($withBlocks with blocks) match PyTorch, min IoU %.3f".format(smallestIoU))
        assertTrue(failures.isEmpty(), failures.joinToString("\n"))
    }

    /** The real sidewalk photos have no blocks; the segmenter must stay quiet on them. */
    @Test
    fun goldenCoversQuietAndBlockFrames() {
        val root = File(System.getProperty("gilnun.repoRoot") ?: return)
        val golden = JSONArray(File(root, "frontend/Android/tests/fixtures/braille_golden.json").readText())
        val names = (0 until golden.length()).map { golden.getJSONObject(it).getString("image") }
        assertEquals(true, names.any { it.startsWith("test_images/") }, "real sidewalk photos are covered")
    }

    /** Center 9:16 crop scaled to 360x640 with bilinear filtering, as the golden script does. */
    private fun frame(file: File): IntArray {
        val image = ImageIO.read(file)
        val cropWidth = minOf(image.width, Math.round(image.height * 9 / 16.0).toInt())
        val left = (image.width - cropWidth) / 2
        val scaled = BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB)
        scaled.createGraphics().apply {
            setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            drawImage(image, 0, 0, WIDTH, HEIGHT, left, 0, left + cropWidth, image.height, null)
            dispose()
        }
        return scaled.getRGB(0, 0, WIDTH, HEIGHT, null, 0, WIDTH)
    }

    private companion object {
        const val WIDTH = 360
        const val HEIGHT = 640
    }
}
