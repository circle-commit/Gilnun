package com.kojingapla.glass

import kotlin.math.max
import kotlin.math.min

data class LiveSceneResult(
    /** Sentence to speak now; empty when nothing new is worth announcing. */
    val voiceGuide: String,
    /** Highest risk level among the announced objects, used to decide whether to interrupt speech. */
    val voiceUrgency: RiskLevel,
    /** Detections sorted by guidance priority (riskiest first). */
    val detections: List<SceneDetection>,
    /** `voiceGuide` split into sentences, each panned toward its object. */
    val sentences: List<SpokenSentence> = emptyList()
) {
    companion object {
        val EMPTY = LiveSceneResult("", RiskLevel.LOW, emptyList())
    }
}

/**
 * Turns detector output into prioritized detections and spoken guidance.
 *
 * Port of iOS `SceneAnalyzer.swift` without its LiDAR parts, which Android phones lack.
 * Call it from one thread; it keeps state across frames to detect approaching objects
 * and to avoid repeating itself.
 */
class SceneAnalyzer(
    private val eventTracker: GuidanceEventTracker = GuidanceEventTracker(calmerSpeech = true)
) {
    private val approachTracker = ApproachTracker(minGrowthRatio = 1.28)

    fun reset() {
        approachTracker.reset()
        eventTracker.reset()
    }

    /**
     * @param rawDetections detector output in `frameWidth` x `frameHeight` pixel coordinates.
     * @param now monotonic timestamp in seconds.
     */
    fun analyze(rawDetections: List<RawDetection>, frameWidth: Double, frameHeight: Double, now: Double): LiveSceneResult {
        val frameArea = max(1.0, frameWidth * frameHeight)

        // The backend rounded these values for its JSON response before applying
        // thresholds; rounding the same way keeps decisions identical to it.
        val detections = rawDetections.mapNotNull { raw ->
            val areaRatio = roundHalfEven(min(1.0, raw.bbox.area / frameArea), 6)
            if (areaRatio < MINIMUM_AREA_RATIO) return@mapNotNull null

            SceneDetection(
                label = raw.label,
                koreanLabel = KOREAN_LABELS[raw.label] ?: raw.label,
                confidence = roundHalfEven(raw.confidence, 4),
                bbox = BoundingBox(
                    roundHalfEven(raw.bbox.x1, 2),
                    roundHalfEven(raw.bbox.y1, 2),
                    roundHalfEven(raw.bbox.x2, 2),
                    roundHalfEven(raw.bbox.y2, 2)
                ),
                frameWidth = frameWidth,
                position = HorizontalPosition.of(raw.bbox.centerX, frameWidth),
                areaRatio = areaRatio,
                verticalRatio = roundHalfEven(max(0.0, min(1.0, raw.bbox.y2 / max(1.0, frameHeight))), 4)
            )
        }.toMutableList()

        val alerts = approachTracker.update(detections.map { it.label to it.bbox }, now)
        for (alert in alerts) {
            detections[alert.detectionIndex] = detections[alert.detectionIndex].copy(
                approaching = true,
                growthRatio = alert.growthRatio
            )
        }

        val prioritized = GuidanceRules.enrichAndPrioritize(detections)
        if (prioritized.isEmpty()) return LiveSceneResult.EMPTY

        val events = eventTracker.chooseEvents(prioritized, now, limit = 2)
        return LiveSceneResult(
            voiceGuide = events.joinToString(" ") { it.message },
            voiceUrgency = events.maxOfOrNull { it.detection.riskLevel } ?: RiskLevel.LOW,
            detections = prioritized,
            sentences = events.map { SpokenSentence(it.message, SpatialAudio.pan(it.detection.bbox, frameWidth)) }
        )
    }

    companion object {
        val KOREAN_LABELS = mapOf(
            "person" to "사람",
            "car" to "차량",
            "truck" to "트럭",
            "bus" to "버스",
            "bicycle" to "자전거",
            "motorcycle" to "오토바이",
            "scooter" to "전동 킥보드",
            "wheelchair" to "휠체어",
            "stroller" to "유모차",
            "traffic_light" to "신호등",
            "traffic_sign" to "교통 표지판",
            "pole" to "기둥",
            "bollard" to "볼라드 / 차단봉",
            "bench" to "벤치",
            "tree_trunk" to "나무",
            "movable_signage" to "입간판",
            "potted_plant" to "화분",
            "parking_meter" to "주차 정산기",
            "stop" to "버스 정류장",
            "table" to "테이블",
            "barricade" to "바리케이드",
            "chair" to "의자",
            "fire_hydrant" to "소화전",
            "kiosk" to "가판대",
            "carrier" to "카트",
            "dog" to "강아지",
            "traffic_light_controller" to "신호 제어함",
            "power_controller" to "전기 분전함"
        )

        /** Boxes smaller than 1% of the frame are ignored. */
        private const val MINIMUM_AREA_RATIO = 0.01
    }
}
