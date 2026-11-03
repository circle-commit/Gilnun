package com.kojingapla.glass

import java.io.File
import javax.imageio.ImageIO
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.fail

class PedestrianSignalTest {
    private val red = 0xFFFF2A20.toInt()
    private val green = 0xFF20E0B0.toInt()
    private val housing = 0xFF202020.toInt()

    /** A 40x100 signal at (100, 50) in a 300x400 gray frame, with the given lamps lit. */
    private fun frame(topLamp: Int?, bottomLamp: Int?): IntArray {
        val width = 300
        val pixels = IntArray(width * 400) { 0xFF808080.toInt() }
        for (y in 50 until 150) for (x in 100 until 140) {
            val lamp = if (y < 100) topLamp else bottomLamp
            val inLamp = (x - 120) * (x - 120) + (y - (if (y < 100) 75 else 125)) * (y - (if (y < 100) 75 else 125)) < 12 * 12
            pixels[y * width + x] = if (inLamp && lamp != null) lamp else housing
        }
        return pixels
    }

    private val box = BoundingBox(100.0, 50.0, 140.0, 150.0)

    @Test
    fun readsTheLitLamp() {
        assertEquals(SignalLamp.RED, PedestrianSignalReader.read(frame(red, null), 300, 400, box), "red on top")
        assertEquals(SignalLamp.GREEN, PedestrianSignalReader.read(frame(null, green), 300, 400, box), "green below")
        assertEquals(SignalLamp.DARK, PedestrianSignalReader.read(frame(null, null), 300, 400, box), "nothing lit")
        assertEquals(SignalLamp.UNKNOWN, PedestrianSignalReader.read(frame(red, green), 300, 400, box), "both lit is unclear")
        assertEquals(SignalLamp.UNKNOWN, PedestrianSignalReader.read(frame(green, red), 300, 400, box), "lamps in the wrong place")
    }

    @Test
    fun picksTheTallSignalAhead() {
        fun light(x1: Double, y1: Double, x2: Double, y2: Double, confidence: Double = 0.8) =
            RawDetection("traffic_light", confidence, BoundingBox(x1, y1, x2, y2))
        val vehicle = light(500.0, 100.0, 620.0, 140.0)
        val pedestrianAhead = light(520.0, 300.0, 545.0, 360.0)
        val pedestrianAside = light(20.0, 300.0, 60.0, 400.0)
        val faint = light(530.0, 500.0, 560.0, 570.0, confidence = 0.2)
        val chosen = PedestrianSignalReader.mainSignal(listOf(vehicle, pedestrianAside, faint, pedestrianAhead), 1080, 1920)
        assertEquals(pedestrianAhead, chosen, "the wide vehicle light, the light off to the side and the faint one are skipped")
        assertNull(PedestrianSignalReader.mainSignal(listOf(vehicle), 1080, 1920), "no pedestrian signal in view")
    }

    @Test
    fun announcesOnlyConfirmedChanges() {
        val tracker = PedestrianSignalTracker()
        var now = 0.0
        fun see(lamp: SignalLamp?, frames: Int = 1): List<String> = (0 until frames).mapNotNull {
            now += 0.25
            tracker.update(lamp, 0.1f, now)?.message
        }

        assertTrue(see(SignalLamp.RED, 2).isEmpty(), "two readings are not enough")
        val red = see(SignalLamp.RED, 2)
        assertEquals(listOf("보행 신호가 빨간불이에요. 기다려 주세요."), red, "red after a few agreeing frames")
        assertTrue(see(SignalLamp.RED, 8).isEmpty(), "red is not repeated")
        assertTrue(see(SignalLamp.GREEN, 1).isEmpty() && see(SignalLamp.UNKNOWN, 1).isEmpty(), "a single green glitch is ignored")
        val green = see(SignalLamp.GREEN, 4)
        assertEquals(listOf("보행 신호가 초록불로 바뀌었어요. 차 소리를 확인하고 건너세요."), green, "a fresh green after red")

        val flashing = (0 until 4).flatMap { see(SignalLamp.DARK, 2) + see(SignalLamp.GREEN, 2) }
        assertEquals(listOf("보행 신호 초록불이 깜빡여요. 건너지 말고 다음 신호를 기다려 주세요."), flashing, "flashing green once")

        now += 10
        assertNull(tracker.update(null, 0f, now), "signal gone")
        val midPhase = see(SignalLamp.GREEN, 4)
        assertEquals(1, midPhase.size, "a green first seen mid-phase is spoken")
        assertTrue(midPhase[0].contains("다음 신호를 기다리면"), "but the user is told to wait for the next one: $midPhase")
    }

    /** A real Korean crosswalk at night from the sidewalk dataset: the signal across is red. */
    @Test
    fun readsARealRedSignal() {
        val modelPath = System.getProperty("gilnun.model") ?: fail("run through tests/run_tests.sh (gilnun.model is not set)")
        val repoRoot = File(System.getProperty("gilnun.repoRoot"))
        val image = ImageIO.read(File(repoRoot, "test_images/Bbox_0110_MP_SEL_013485.jpg"))
        val width = 608
        val height = 1080
        val pixels = image.getRGB(656, 0, width, height, null, 0, width)

        ObjectDetector(File(modelPath).readBytes()).use { detector ->
            val signal = assertNotNull(
                PedestrianSignalReader.mainSignal(detector.detect(pixels, width, height), width, height),
                "the signal across the street is found"
            )
            assertEquals(SignalLamp.RED, PedestrianSignalReader.read(pixels, width, height, signal.bbox), "and read as red")
        }
    }
}
