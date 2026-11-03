package com.kojingapla.glass

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SpatialAudioTest {
    private fun box(left: Double, right: Double) = BoundingBox(left, 1000.0, right, 1500.0)

    @Test
    fun panFollowsTheBoxCenter() {
        assertEquals(0f, SpatialAudio.pan(box(440.0, 640.0), 1080.0), "a centered box sounds from the front")
        assertEquals(-SpatialAudio.MAX_PAN, SpatialAudio.pan(box(0.0, 0.0), 1080.0), "the left edge is as far left as it goes")
        assertEquals(SpatialAudio.MAX_PAN, SpatialAudio.pan(box(1080.0, 1080.0), 1080.0), "the right edge is as far right as it goes")
        assertEquals(-0.4f, SpatialAudio.pan(box(170.0, 370.0), 1080.0), 1e-6f, "a quarter from the left is halfway left")
        assertEquals(0f, SpatialAudio.pan(box(0.0, 100.0), 0.0), "no frame, no pan")
    }

    @Test
    fun eachSentenceComesFromItsObject() {
        // No cooldowns, so both objects are spoken together on the second frame.
        val analyzer = SceneAnalyzer(GuidanceEventTracker(GuidanceEventTracker.Cooldowns(0.0, 0.0, 0.0, 0.0)) { 0.0 })
        val car = RawDetection("car", 0.9, BoundingBox(800.0, 900.0, 1060.0, 1400.0))
        val bollard = RawDetection("bollard", 0.9, BoundingBox(480.0, 1500.0, 600.0, 1800.0))
        // The bollard needs a second sighting; a car is spoken on its first.
        analyzer.analyze(listOf(bollard), 1080.0, 1920.0, 10.0)
        val result = analyzer.analyze(listOf(car, bollard), 1080.0, 1920.0, 10.2)

        assertEquals(result.voiceGuide, result.sentences.joinToString(" ") { it.text }, "the sentences are the spoken guide")
        assertEquals(2, result.sentences.size, "both objects are spoken: ${result.sentences}")
        val bollardSentence = result.sentences.first { it.text.contains("볼라드") }
        assertEquals(0f, bollardSentence.pan, 0.05f, "the bollard ahead sounds from the front")
        val carSentence = result.sentences.first { it.text.contains("차량") }
        assertTrue(carSentence.pan > 0.5f, "the car on the right sounds from the right: ${carSentence.pan}")
    }
}
