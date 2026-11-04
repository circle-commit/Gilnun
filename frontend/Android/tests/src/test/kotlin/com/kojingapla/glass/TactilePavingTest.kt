package com.kojingapla.glass

import java.io.File
import javax.imageio.ImageIO
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TactilePavingTest {
    private val root = File(System.getProperty("gilnun.repoRoot") ?: "../../..")
    private val fixtures = File(root, "frontend/Android/tests/fixtures")

    private fun find(file: File): TactilePath? {
        val image = ImageIO.read(file)
        // Portrait crop and 360 px wide, like a downscaled camera frame.
        val cropWidth = minOf(image.width, Math.round(image.height * 9 / 16.0).toInt())
        val left = (image.width - cropWidth) / 2
        val scaled = java.awt.image.BufferedImage(360, 640, java.awt.image.BufferedImage.TYPE_INT_RGB)
        scaled.createGraphics().apply {
            setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            drawImage(image, 0, 0, 360, 640, left, 0, left + cropWidth, image.height, null)
            dispose()
        }
        val pixels = scaled.getRGB(0, 0, 360, 640, null, 0, 360)
        return TactilePavingDetector.find(pixels, 360, 640)
    }

    @Test
    fun noPavingInRealSidewalkPhotos() {
        // Yellow crosswalks, yellow curbs and center lines, red paving and yellowish grass.
        val photos = File(root, "test_images").listFiles()!!.toList() +
            File(root, "datasets/yolo_sidewalk/images/train").listFiles()!!.toList()
        for (photo in photos.sorted()) {
            assertNull(find(photo), "${photo.name} has no tactile paving")
        }
    }

    @Test
    fun findsPavingAndWhereItRuns() {
        val straight = assertNotNull(find(File(fixtures, "tactile_straight.jpg")), "straight bars")
        assertEquals(HorizontalPosition.CENTER, straight.position)
        assertNull(straight.turn, "runs straight ahead")

        val turning = assertNotNull(find(File(fixtures, "tactile_turn_right.jpg")), "domes bending right")
        assertEquals(HorizontalPosition.CENTER, turning.position, "at the feet it is ahead")
        assertEquals(HorizontalPosition.RIGHT, turning.turn, "farther on it bends right")

        val left = assertNotNull(find(File(fixtures, "tactile_left.jpg")), "paving off to the left")
        assertEquals(HorizontalPosition.LEFT, left.position)
    }

    @Test
    fun ignoresSmoothPaintAndWideYellowAreas() {
        assertNull(find(File(fixtures, "yellow_paint_strip.jpg")), "same color and shape, but smooth paint")
        assertNull(find(File(fixtures, "yellow_wide_area.jpg")), "too wide to be a path")
    }

    @Test
    fun speaksOnlyChanges() {
        val guide = TactilePavingGuide()
        var now = 0.0
        fun see(path: TactilePath?, frames: Int = 1): List<String> = (0 until frames).mapNotNull {
            now += 0.25
            guide.update(path, now)?.text
        }
        val ahead = TactilePath(0.5, 0.5, 6)
        val bendingRight = TactilePath(0.5, 0.75, 6)
        val onTheLeft = TactilePath(0.2, 0.2, 6)

        assertTrue(see(ahead, 2).isEmpty(), "a couple of frames are not enough")
        assertEquals(listOf("점자블록이 정면에 있어요."), see(ahead, 2), "found ahead")
        assertTrue(see(ahead, 20).isEmpty(), "not repeated while it stays the same")
        assertEquals(listOf("앞에서 점자블록이 오른쪽으로 꺾여요."), see(bendingRight, 4), "a bend ahead")
        assertEquals(listOf("점자블록이 왼쪽에 있어요."), see(onTheLeft, 20), "no longer underfoot, but on the left")
        assertTrue(see(null, 7).isEmpty(), "a moment out of view is fine")
        assertEquals(listOf("점자블록이 더 이상 보이지 않아요."), see(null, 2), "gone for 2 s")

        val panned = TactilePavingGuide().let { fresh ->
            (0 until 4).mapNotNull { fresh.update(onTheLeft, it * 0.25) }.single()
        }
        assertTrue(panned.pan < -0.4f, "spoken from the left: ${panned.pan}")
    }
}
