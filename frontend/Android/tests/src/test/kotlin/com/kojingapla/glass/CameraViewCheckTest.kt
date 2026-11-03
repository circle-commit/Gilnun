package com.kojingapla.glass

import java.io.File
import javax.imageio.ImageIO
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class CameraViewCheckTest {
    /** A gravity reading for a portrait phone whose back camera looks `degrees` above the horizon. */
    private fun aim(degrees: Double): ViewProblem? {
        val radians = Math.toRadians(degrees)
        return CameraViewCheck.aimProblem(0f, (9.81 * cos(radians)).toFloat(), (-9.81 * sin(radians)).toFloat())
    }

    @Test
    fun tellsWhereTheCameraPoints() {
        assertNull(aim(0.0), "held upright, looking ahead")
        assertNull(aim(-30.0), "looking a little down at the sidewalk is fine")
        assertEquals(ViewProblem.POINTING_DOWN, aim(-70.0), "looking at the ground")
        assertEquals(ViewProblem.POINTING_DOWN, CameraViewCheck.aimProblem(0f, 0f, 9.81f), "lying flat, screen up")
        assertEquals(ViewProblem.POINTING_UP, aim(45.0), "looking at the sky")
        assertEquals(ViewProblem.SIDEWAYS, CameraViewCheck.aimProblem(9.81f, 1f, 0f), "held sideways")
        assertNull(CameraViewCheck.aimProblem(0f, 0f, 0f), "no reading yet")
    }

    @Test
    fun tellsACoveredLensFromADarkStreet() {
        assertEquals(ViewProblem.LENS_COVERED, CameraViewCheck.sightProblem(8.0, 0.5), "dark and featureless")
        assertEquals(ViewProblem.TOO_DARK, CameraViewCheck.sightProblem(20.0, 9.0), "dark but with edges and lights")
        assertEquals(ViewProblem.TOO_DARK, CameraViewCheck.sightProblem(30.0, 1.0), "dim but not black")
        assertNull(CameraViewCheck.sightProblem(110.0, 12.0), "a normal street")
    }

    @Test
    fun speaksProblemsOnlyWhenTheyLast() {
        val speaker = ViewProblemSpeaker()
        assertNull(speaker.update(ViewProblem.POINTING_DOWN, 0.0), "a glance down at a curb")
        assertNull(speaker.update(null, 0.8), "is over quickly and says nothing")
        assertNull(speaker.update(ViewProblem.POINTING_DOWN, 5.0))
        assertNull(speaker.update(ViewProblem.POINTING_DOWN, 6.0), "not long enough yet")
        assertEquals(ViewProblem.POINTING_DOWN.message, speaker.update(ViewProblem.POINTING_DOWN, 6.6), "after 1.5 s")
        assertNull(speaker.update(ViewProblem.POINTING_DOWN, 10.0), "not repeated right away")
        assertEquals(ViewProblem.POINTING_DOWN.message, speaker.update(ViewProblem.POINTING_DOWN, 16.7), "repeated after 10 s")
        assertNull(speaker.update(null, 17.0), "fixed, waiting to be sure")
        assertEquals("좋아요. 이제 앞이 잘 보여요.", speaker.update(null, 18.1), "the fix is confirmed once")
        assertNull(speaker.update(null, 25.0), "and only once")
    }

    /** Real sidewalk photos, including the night crossing, must not trigger a warning. */
    @Test
    fun realStreetsAreNotTooDark() {
        val root = File(System.getProperty("gilnun.repoRoot") ?: return)
        for (file in File(root, "test_images").listFiles()!!.sorted()) {
            val image = ImageIO.read(file)
            val values = DoubleArray(LumaSample.COLUMNS * LumaSample.ROWS) { index ->
                val x = (index % LumaSample.COLUMNS) * image.width / LumaSample.COLUMNS
                val y = (index / LumaSample.COLUMNS) * image.height / LumaSample.ROWS
                val rgb = image.getRGB(x, y)
                0.299 * ((rgb shr 16) and 0xFF) + 0.587 * ((rgb shr 8) and 0xFF) + 0.114 * (rgb and 0xFF)
            }
            val sample = LumaSample(values)
            assertNull(CameraViewCheck.sightProblem(sample.brightness, sample.blurScore()), "${file.name}: brightness ${sample.brightness}")
        }
    }
}
