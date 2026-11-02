package com.kojingapla.glass

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/** Port of iOS `GuidanceTests.swift`: backend unit tests, calmer speech and backend parity. */
class GuidanceTest {
    private var now = 100.0

    // ── Ports of backend/tests ────────────────────────────────

    @Test
    fun distanceEstimation() {
        assertEquals(DistanceLevel.NEAR, GuidanceRules.distanceLevel("car", 0.05, 0.65), "car at 5% area is near")
        assertEquals(DistanceLevel.CLOSE, GuidanceRules.distanceLevel("car", 0.08, 0.65), "car at 8% area is close")
        assertEquals(DistanceLevel.CLOSE, GuidanceRules.distanceLevel("bollard", 0.011, 0.61), "low bollard is close")
        assertEquals(DistanceLevel.NEAR, GuidanceRules.distanceLevel("bollard", 0.011, 0.43), "higher bollard is near")
        assertEquals(DistanceLevel.NEAR, GuidanceRules.distanceLevel("traffic_sign", 0.02, 0.55), "sign is near")
        assertEquals(DistanceLevel.FAR, GuidanceRules.distanceLevel("traffic_sign", 0.01, 0.55), "small sign is far")

        val front = GuidanceRules.enrichAndPrioritize(
            listOf(unitDetection("car", areaRatio = 0.05, verticalRatio = 0.65), unitDetection("bollard", areaRatio = 0.011, verticalRatio = 0.61))
        ).associate { it.label to it.frontDangerZone }
        assertEquals(false, front["car"], "car is not in the front danger zone")
        assertEquals(true, front["bollard"], "bollard is in the front danger zone")
    }

    @Test
    fun temporalConfidence() {
        fun events(tracker: GuidanceEventTracker, label: String, confidence: Double = 0.8): Int {
            now += 0.2
            val detections = GuidanceRules.enrichAndPrioritize(listOf(unitDetection(label, confidence, 0.05, 0.7)))
            return tracker.chooseEvents(detections, now).size
        }
        fun zeroCooldownTracker() = GuidanceEventTracker(GuidanceEventTracker.Cooldowns(0.0, 0.0, 0.0, 0.0))

        val person = zeroCooldownTracker()
        assertEquals(0, events(person, "person"), "person needs a second frame")
        assertEquals(1, events(person, "person"), "person speaks on the second frame")

        val pole = zeroCooldownTracker()
        assertEquals(0, events(pole, "pole"), "pole frame 1")
        assertEquals(0, events(pole, "pole"), "pole frame 2")
        assertEquals(1, events(pole, "pole"), "pole speaks on the third frame")

        val noisyPole = zeroCooldownTracker()
        repeat(3) { assertEquals(0, events(noisyPole, "pole", confidence = 0.4), "low-confidence pole frame ${it + 1} is ignored") }
        assertEquals(0, events(noisyPole, "pole"), "confident pole frame 1")
        assertEquals(0, events(noisyPole, "pole"), "confident pole frame 2")
        assertEquals(1, events(noisyPole, "pole"), "confident pole speaks on the third frame")

        assertEquals(1, events(zeroCooldownTracker(), "car"), "car speaks on the first frame")
    }

    // ── Calmer speech (the iOS and Android apps) ──────────────

    private fun obj(
        label: String,
        meters: Double,
        position: HorizontalPosition = HorizontalPosition.LEFT,
        right: Double = 300.0,
        approaching: Boolean = false
    ) = SceneDetection(
        label = label,
        koreanLabel = SceneAnalyzer.KOREAN_LABELS[label] ?: label,
        confidence = 0.8,
        bbox = BoundingBox(100.0, 900.0, right, 1300.0),
        frameWidth = 1080.0,
        position = position,
        areaRatio = 0.05,
        verticalRatio = 0.7,
        distanceMeters = meters,
        approaching = approaching
    )

    private fun spoken(tracker: GuidanceEventTracker, detection: SceneDetection, frames: Int): List<String> =
        (0 until frames).flatMap {
            now += 0.2
            tracker.chooseEvents(GuidanceRules.enrichAndPrioritize(listOf(detection)), now).map { it.message }
        }

    private fun tracker(calmer: Boolean) =
        GuidanceEventTracker(GuidanceEventTracker.Cooldowns(0.0, 0.0, 0.0, 0.0), calmerSpeech = calmer) { 0.0 }

    @Test
    fun calmerSpeech() {
        assertEquals(1, spoken(tracker(false), obj("pole", 3.0), 4).size, "backend: pole beside the path is spoken")
        assertTrue(spoken(tracker(true), obj("pole", 3.0), 4).isEmpty(), "calmer: pole beside the path at 3 m is quiet")
        assertEquals(1, spoken(tracker(true), obj("pole", 1.0), 4).size, "calmer: pole within reach is spoken")
        assertEquals(1, spoken(tracker(true), obj("pole", 3.0, HorizontalPosition.CENTER), 4).size, "calmer: pole in the path is spoken")
        assertEquals(
            1, spoken(tracker(true), obj("movable_signage", 2.0, right = 600.0), 4).size,
            "calmer: a sign beside the path whose box crosses the middle reaches into the path and is spoken"
        )
        val reach = tracker(true)
        assertTrue(spoken(reach, obj("pole", 2.0), 4).isEmpty(), "calmer: pole at 2 m beside the path is quiet")
        assertEquals(1, spoken(reach, obj("pole", 1.4), 1).size, "calmer: spoken on its first frame within 1.5 m")
        val longWalk = tracker(true)
        assertTrue(spoken(longWalk, obj("pole", 2.0), 65).isEmpty(), "calmer: pole beside the path for 13 s is quiet")
        assertEquals(1, spoken(longWalk, obj("pole", 1.4), 1).size, "calmer: still spoken on its first frame within reach")
        assertEquals(1, spoken(tracker(true), obj("car", 6.0), 1).size, "calmer: vehicles beside the path are still spoken")

        val gap = tracker(true)
        assertEquals(1, spoken(gap, obj("person", 4.0), 2).size, "calmer: person beside the path is spoken")
        assertTrue(spoken(gap, obj("dog", 4.0, HorizontalPosition.RIGHT), 3).isEmpty(), "calmer: another calm sentence waits")
        now += 4
        assertEquals(1, spoken(gap, obj("dog", 4.0, HorizontalPosition.RIGHT), 1).size, "calmer: spoken after the calm gap")
        assertEquals(1, spoken(gap, obj("bollard", 2.0, HorizontalPosition.CENTER), 3).size, "calmer: something in the path does not wait")
        assertTrue(spoken(gap, obj("car", 6.0, HorizontalPosition.RIGHT), 1).isEmpty(), "calmer: a car beside the path and not coming waits")
        assertEquals(1, spoken(gap, obj("person", 4.0, approaching = true), 1).size, "calmer: a person coming closer does not wait")
    }

    @Test
    fun displayRisk() {
        // The screen keeps "위험" (85 and up) for what is in the way or coming at the user.
        val parkedCar = GuidanceRules.enrichAndPrioritize(listOf(obj("car", 6.0)))[0]
        val personAhead = GuidanceRules.enrichAndPrioritize(listOf(obj("person", 2.0, HorizontalPosition.CENTER)))[0]
        assertTrue(parkedCar.riskScore >= 85 && GuidanceRules.displayRisk(parkedCar) == 84, "display: a car beside the path shows 주의")
        assertTrue(GuidanceRules.displayRisk(personAhead) >= 85, "display: a person in the way shows 위험")

        val signs = GuidanceRules.enrichAndPrioritize(
            listOf(obj("movable_signage", 2.0, approaching = true), obj("movable_signage", 2.0, right = 600.0, approaching = true))
        )
        assertTrue(signs.all { it.riskScore >= 85 }, "display: both signs score high")
        assertEquals(84, GuidanceRules.displayRisk(signs.first { !GuidanceRules.reachesPath(it) }), "display: a sign beside the path shows 주의")
        assertTrue(GuidanceRules.displayRisk(signs.first { GuidanceRules.reachesPath(it) }) >= 85, "display: a sign reaching into the path shows 위험")

        val pole1m = GuidanceRules.enrichAndPrioritize(listOf(obj("pole", 1.0, approaching = true)))[0]
        val backendWords = GuidanceRules.detectionMessage(pole1m, GuidanceEventType.APPROACHING) { 0.0 }
        val calmerWords = GuidanceRules.detectionMessage(pole1m, GuidanceEventType.APPROACHING, calmerSpeech = true) { 0.0 }
        assertTrue(backendWords.contains("다가오고"), "backend: $backendWords")
        assertTrue(!calmerWords.contains("다가오고") && calmerWords.contains("가까워지고"), "calmer: a pole does not come closer: $calmerWords")
    }

    @Test
    fun liveSceneWithoutDistance() {
        // Android has no LiDAR: a centered, close obstacle is still announced from its box.
        val analyzer = SceneAnalyzer(GuidanceEventTracker(calmerSpeech = true) { 0.0 })
        val bollard = RawDetection("bollard", 0.9, BoundingBox(480.0, 1500.0, 600.0, 1800.0))
        assertEquals("", analyzer.analyze(listOf(bollard), 1080.0, 1920.0, 10.0).voiceGuide, "first sighting waits for a second frame")
        val result = analyzer.analyze(listOf(bollard), 1080.0, 1920.0, 10.2)
        assertEquals("앞에 볼라드 / 차단봉이 있어요. 천천히 이동해 주세요.", result.voiceGuide)
        assertEquals(RiskLevel.CRITICAL, result.voiceUrgency)
    }

    // ── Parity with the Python backend ────────────────────────

    @Test
    fun parityWithBackend() {
        val path = System.getProperty("gilnun.fixtures") ?: fail("run through tests/run_tests.sh (gilnun.fixtures is not set)")
        val fixtures = JSONObject(File(path).readText())
        val frameWidth = fixtures.getDouble("frame_width")
        val frameHeight = fixtures.getDouble("frame_height")
        val failures = ArrayList<String>()
        fun expect(condition: Boolean, message: () -> String) {
            if (!condition) failures.add(message())
        }

        fixtures.getJSONArray("rules").objects().forEachIndexed { index, case ->
            val detection = GuidanceRules.enrichAndPrioritize(listOf(sceneDetection(case.getJSONObject("detection"), frameWidth)))[0]
            val name = "rules[$index] ${detection.label}"
            expect(detection.frontDangerZone == case.getBoolean("front_danger_zone")) { "$name front danger zone" }
            expect(detection.riskScore == case.getInt("risk_score")) { "$name risk score ${detection.riskScore} != ${case.getInt("risk_score")}" }
            expect(detection.riskLevel.rawValue == case.getString("risk_level")) { "$name risk level" }
            expect(GuidanceRules.distanceLevel(detection).rawValue == case.getString("distance_level")) { "$name distance level" }
            expect(GuidanceRules.objectGroup(detection.label).rawValue == case.getString("object_group")) { "$name object group" }
            expect(GuidanceRules.particle(detection.koreanLabel) == case.getString("particle")) { "$name particle" }
        }

        fixtures.getJSONArray("priority").objects().forEachIndexed { index, case ->
            // The Korean label carries the input index; it does not affect ordering.
            val detections = case.getJSONArray("detections").objects().mapIndexed { offset, detection ->
                sceneDetection(detection, frameWidth, koreanLabel = offset.toString())
            }
            val order = GuidanceRules.enrichAndPrioritize(detections).map { it.koreanLabel.toInt() }
            val expected = case.getJSONArray("order").ints()
            expect(order == expected) { "priority[$index] order $order != $expected" }
        }

        fixtures.getJSONArray("templates").objects().forEachIndexed { index, case ->
            val detection = GuidanceRules.enrichAndPrioritize(listOf(sceneDetection(case.getJSONObject("detection"), frameWidth)))[0]
            val eventType = GuidanceEventType.fromRaw(case.getString("event_type"))
            if (eventType == null) {
                expect(false) { "templates[$index] unknown event ${case.getString("event_type")}" }
                return@forEachIndexed
            }
            val templates = GuidanceRules.weightedTemplates(GuidanceRules.messageContext(detection, eventType))
            val expected = case.getJSONArray("templates").arrays().map { GuidanceRules.WeightedTemplate(it.getString(0), it.getInt(1)) }
            expect(templates == expected) { "templates[$index] candidates $templates != $expected" }
            val message = GuidanceRules.detectionMessage(detection, eventType) { 0.0 }
            expect(message == case.getString("message")) { "templates[$index] message $message != ${case.getString("message")}" }
        }

        fixtures.getJSONArray("event_sequences").objects().forEachIndexed { sequenceIndex, sequence ->
            val cooldowns = sequence.getJSONObject("cooldowns")
            val tracker = GuidanceEventTracker(
                GuidanceEventTracker.Cooldowns(
                    cooldowns.getDouble("global"),
                    cooldowns.getDouble("object"),
                    cooldowns.getDouble("situation"),
                    cooldowns.getDouble("signature"),
                    cooldowns.getDouble("stale_after")
                )
            ) { 0.0 }
            sequence.getJSONArray("frames").objects().forEachIndexed { frameIndex, frame ->
                val detections = GuidanceRules.enrichAndPrioritize(
                    frame.getJSONArray("detections").objects().map { sceneDetection(it, frameWidth) }
                )
                val actual = tracker.chooseEvents(detections, frame.getDouble("now")).map {
                    "${it.type.rawValue}|${it.detection.label}|${it.detection.position.rawValue}|${it.message}"
                }
                val expected = frame.getJSONArray("events").objects().map {
                    "${it.getString("event")}|${it.getString("label")}|${it.getString("position")}|${it.getString("message")}"
                }
                expect(actual == expected) { "event_sequences[$sequenceIndex][$frameIndex] $actual != $expected" }
            }
        }

        var approachAlerts = 0
        fixtures.getJSONArray("approach_sequences").objects().forEachIndexed { sequenceIndex, sequence ->
            val tracker = ApproachTracker(minGrowthRatio = 1.28)
            sequence.getJSONArray("frames").objects().forEachIndexed { frameIndex, frame ->
                val detections = frame.getJSONArray("detections").objects().map { it.getString("label") to box(it.getJSONArray("bbox")) }
                val actual = tracker.update(detections, frame.getDouble("now")).map { "${it.detectionIndex}|${it.trackId}|${it.growthRatio}" }
                val expected = frame.getJSONArray("alerts").objects().map {
                    "${it.getInt("index")}|${it.getInt("track_id")}|${it.getDouble("growth_ratio")}"
                }
                approachAlerts += expected.size
                expect(actual == expected) { "approach_sequences[$sequenceIndex][$frameIndex] $actual != $expected" }
            }
        }

        var sceneFrames = 0
        var spokenFrames = 0
        fixtures.getJSONArray("scene_sequences").objects().forEachIndexed { sequenceIndex, sequence ->
            val analyzer = SceneAnalyzer(GuidanceEventTracker { 0.0 })
            sequence.getJSONArray("frames").objects().forEachIndexed { frameIndex, frame ->
                val raw = frame.getJSONArray("raw").objects().map {
                    RawDetection(it.getString("label"), it.getDouble("confidence"), box(it.getJSONArray("bbox")))
                }
                val result = analyzer.analyze(raw, frameWidth, frameHeight, frame.getDouble("now"))
                val name = "scene_sequences[$sequenceIndex][$frameIndex]"
                val voiceGuide = frame.getString("voice_guide")
                sceneFrames += 1
                if (voiceGuide.isNotEmpty()) spokenFrames += 1
                expect(result.voiceGuide == voiceGuide) { "$name voice guide \"${result.voiceGuide}\" != \"$voiceGuide\"" }

                val actual = result.detections.map {
                    "${it.label}|${it.position.rawValue}|${it.approaching}|${it.riskScore}|${it.riskLevel.rawValue}"
                }
                val expected = frame.getJSONArray("detections").objects().map {
                    "${it.getString("label")}|${it.getString("position")}|${it.getBoolean("approaching")}|" +
                        "${it.getInt("risk_score")}|${it.getString("risk_level")}"
                }
                expect(actual == expected) { "$name detections $actual != $expected" }
            }
        }

        println(
            "guidance parity: ${fixtures.getJSONArray("rules").length()} rule cases, " +
                "${fixtures.getJSONArray("priority").length()} orderings, ${fixtures.getJSONArray("templates").length()} template cases, " +
                "${fixtures.getJSONArray("event_sequences").length()} event sequences, $approachAlerts approach alerts, " +
                "$sceneFrames scene frames ($spokenFrames with speech)"
        )
        assertTrue(failures.isEmpty(), "${failures.size} parity failures:\n" + failures.take(40).joinToString("\n"))
    }

    private fun unitDetection(label: String, confidence: Double = 0.8, areaRatio: Double, verticalRatio: Double) = SceneDetection(
        label = label,
        koreanLabel = label,
        confidence = confidence,
        bbox = BoundingBox(100.0, 100.0, 300.0, 400.0),
        frameWidth = 1080.0,
        position = HorizontalPosition.CENTER,
        areaRatio = areaRatio,
        verticalRatio = verticalRatio
    )

    private fun sceneDetection(json: JSONObject, frameWidth: Double, koreanLabel: String? = null) = SceneDetection(
        label = json.getString("label"),
        koreanLabel = koreanLabel ?: json.getString("korean_label"),
        confidence = json.getDouble("confidence"),
        bbox = box(json.getJSONArray("bbox_xyxy")),
        frameWidth = frameWidth,
        position = HorizontalPosition.fromRaw(json.getString("position")),
        areaRatio = json.getDouble("area_ratio"),
        verticalRatio = json.getDouble("vertical_ratio"),
        approaching = json.getBoolean("approaching")
    )

    private fun box(values: JSONArray) = BoundingBox(values.getDouble(0), values.getDouble(1), values.getDouble(2), values.getDouble(3))
}

internal fun JSONArray.objects(): List<JSONObject> = (0 until length()).map { getJSONObject(it) }

internal fun JSONArray.arrays(): List<JSONArray> = (0 until length()).map { getJSONArray(it) }

internal fun JSONArray.ints(): List<Int> = (0 until length()).map { getInt(it) }
