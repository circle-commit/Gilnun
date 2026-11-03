package com.kojingapla.glass

import java.util.Locale
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Risk scoring, distance estimation and Korean speech templates for live guidance.
 *
 * Port of iOS `GuidanceEngine.swift`, itself a port of
 * `backend/services/guidance_message_service.py`. Keep the three in sync;
 * `frontend/Android/tests/run_tests.sh` checks this file against the Python code.
 */
object GuidanceRules {
    enum class ObjectGroup(val rawValue: String) {
        VEHICLE("vehicle"),
        MOBILITY("mobility"),
        PERSON("person"),
        OBSTACLE("obstacle"),
        TRAFFIC("traffic")
    }

    class DistanceThresholds(
        val veryCloseArea: Double,
        val veryCloseVertical: Double,
        val closeArea: Double,
        val closeVertical: Double,
        val nearArea: Double,
        val nearVertical: Double
    )

    val highestRiskLabels = setOf("car", "truck", "bus")
    val highRiskLabels = setOf("motorcycle", "scooter", "bicycle")
    val mediumHighRiskLabels = setOf("bollard", "pole", "movable_signage", "tree_trunk", "barricade", "fire_hydrant")
    val mediumRiskLabels = setOf("person", "wheelchair", "stroller", "carrier", "dog")
    val lowerRiskLabels = setOf("bench", "potted_plant", "traffic_sign", "traffic_light")
    val immediateSpeechLabels = highestRiskLabels + setOf("motorcycle", "scooter")
    val lowConfidenceSpeechLabels = highestRiskLabels + highRiskLabels + setOf("wheelchair", "stroller")
    val noisySpeechLabels = setOf("pole", "tree_trunk", "traffic_sign", "traffic_light", "movable_signage")

    /** Fixed objects without a risk set of their own. "stop" is a bus stop. */
    val otherObstacleLabels = setOf(
        "parking_meter", "stop", "table", "chair", "kiosk", "traffic_light_controller", "power_controller"
    )
    val obstacleLabels = mediumHighRiskLabels + setOf("bench", "potted_plant") + otherObstacleLabels
    val mobilityLabels = highRiskLabels + setOf("wheelchair", "stroller", "carrier", "dog")
    val guidanceLabels = highestRiskLabels + highRiskLabels + mediumHighRiskLabels +
        mediumRiskLabels + lowerRiskLabels + otherObstacleLabels

    // Small ground obstacles can be close while their boxes are still small.
    val groundObstacleDistanceLabels = setOf(
        "bollard", "pole", "tree_trunk", "movable_signage", "bench",
        "potted_plant", "parking_meter", "table", "fire_hydrant"
    )

    val defaultDistanceThresholds = DistanceThresholds(0.09, 0.86, 0.04, 0.68, 0.018, 0.52)

    // Large objects occupy a lot of pixels even when they are not immediately near.
    val vehicleDistanceThresholds = DistanceThresholds(0.16, 0.90, 0.08, 0.76, 0.035, 0.58)
    val personDistanceThresholds = DistanceThresholds(0.11, 0.88, 0.055, 0.72, 0.025, 0.54)
    val mobilityDistanceThresholds = DistanceThresholds(0.08, 0.84, 0.035, 0.64, 0.016, 0.48)
    val groundObstacleDistanceThresholds = DistanceThresholds(0.035, 0.80, 0.018, 0.60, 0.01, 0.42)
    val trafficDistanceThresholds = DistanceThresholds(0.08, 0.90, 0.035, 0.76, 0.016, 0.60)

    // ── Risk ──────────────────────────────────────────────────

    fun objectGroup(label: String): ObjectGroup = when {
        label in highestRiskLabels -> ObjectGroup.VEHICLE
        label in mobilityLabels -> ObjectGroup.MOBILITY
        label == "person" -> ObjectGroup.PERSON
        label in obstacleLabels -> ObjectGroup.OBSTACLE
        label == "traffic_sign" || label == "traffic_light" -> ObjectGroup.TRAFFIC
        else -> ObjectGroup.OBSTACLE
    }

    /** Things that stay put: the user walks toward them; they never come closer. */
    fun isFixed(label: String): Boolean {
        val group = objectGroup(label)
        return group == ObjectGroup.OBSTACLE || group == ObjectGroup.TRAFFIC
    }

    /**
     * The box spans the frame's middle, so the object reaches into the walking line even
     * when its center sits in a side third.
     */
    fun reachesPath(detection: SceneDetection): Boolean {
        val middle = detection.frameWidth / 2
        return detection.bbox.x1 <= middle && middle <= detection.bbox.x2
    }

    /**
     * Risk shown on screen. "위험" (85 and up) is for what is in the way or coming at the
     * user; anything else shows as "주의" at most. A fixed object only "approaches" because
     * the user walks toward it.
     */
    fun displayRisk(detection: SceneDetection): Int {
        val distance = distanceLevel(detection)
        val near = distance == DistanceLevel.CLOSE || distance == DistanceLevel.VERY_CLOSE
        val inTheWay = detection.frontDangerZone || (reachesPath(detection) && near) ||
            (detection.approaching && !isFixed(detection.label))
        return if (inTheWay) detection.riskScore else min(detection.riskScore, 84)
    }

    fun distanceLevel(label: String, areaRatio: Double, verticalRatio: Double): DistanceLevel {
        val thresholds = distanceThresholds(label)
        val area = clamp01(areaRatio)
        val vertical = clamp01(verticalRatio)

        return when {
            area >= thresholds.veryCloseArea || vertical >= thresholds.veryCloseVertical -> DistanceLevel.VERY_CLOSE
            area >= thresholds.closeArea || vertical >= thresholds.closeVertical -> DistanceLevel.CLOSE
            area >= thresholds.nearArea || vertical >= thresholds.nearVertical -> DistanceLevel.NEAR
            else -> DistanceLevel.FAR
        }
    }

    /** Uses a measured distance when there is one; otherwise estimates from the box like the backend. */
    fun distanceLevel(detection: SceneDetection): DistanceLevel {
        detection.distanceMeters?.let { return distanceLevel(meters = it) }
        return distanceLevel(detection.label, detection.areaRatio, detection.verticalRatio)
    }

    /** At walking speed (about 1.2 m/s) these are roughly 1, 2 and 4 seconds away. */
    fun distanceLevel(meters: Double): DistanceLevel = when {
        meters < 1.2 -> DistanceLevel.VERY_CLOSE
        meters < 2.5 -> DistanceLevel.CLOSE
        meters < 5.0 -> DistanceLevel.NEAR
        else -> DistanceLevel.FAR
    }

    fun isFrontDangerZone(detection: SceneDetection): Boolean {
        if (detection.position != HorizontalPosition.CENTER) return false
        val distance = distanceLevel(detection)
        return distance == DistanceLevel.CLOSE || distance == DistanceLevel.VERY_CLOSE
    }

    fun riskScore(detection: SceneDetection): Int {
        val confidence = clamp01(detection.confidence)
        val areaRatio = clamp01(detection.areaRatio)
        val verticalRatio = clamp01(detection.verticalRatio)

        var score = baseRisk(detection.label)
        score += positionBonus(detection.position)
        score += min(24, (areaRatio * 200).toInt())
        score += min(14, (verticalRatio * 14).toInt())
        score += (confidence * 8).toInt()

        if (detection.approaching) {
            score += if (detection.position == HorizontalPosition.CENTER) 24 else 13
        }
        if (isFrontDangerZone(detection)) {
            score += 10
        }
        return max(0, min(100, score))
    }

    private val priorityOrder: Comparator<SceneDetection> =
        compareByDescending<SceneDetection> { it.riskScore }
            .thenByDescending { it.position == HorizontalPosition.CENTER }
            .thenByDescending { it.areaRatio }
            .thenByDescending { it.confidence }

    /** Attaches safety metadata and sorts by live-guidance priority (riskiest first). */
    fun enrichAndPrioritize(detections: List<SceneDetection>): List<SceneDetection> =
        detections
            .map { detection ->
                val score = riskScore(detection)
                detection.copy(
                    frontDangerZone = isFrontDangerZone(detection),
                    riskScore = score,
                    riskLevel = RiskLevel.of(score)
                )
            }
            .sortedWith(priorityOrder)

    // ── Speech ────────────────────────────────────────────────

    class MessageContext(
        val label: String,
        val koreanLabel: String,
        val particle: String,
        val position: HorizontalPosition,
        val riskLevel: RiskLevel,
        val objectGroup: ObjectGroup,
        val eventType: GuidanceEventType,
        val approaching: Boolean,
        val newlyDetected: Boolean,
        val distanceLevel: DistanceLevel,
        val frontWord: String
    )

    enum class TemplateCondition {
        LOW_LEFT,
        LOW_RIGHT,
        MEDIUM_VISIBLE,
        FRONT_OBSTACLE,
        FRONT_OBSTACLE_NAMED,
        CENTER_APPROACHING,
        CRITICAL_FRONT,
        SIDE_APPROACHING,
        SIDE_PERSON,
        FRONT_VEHICLE,
        FRONT_HIGH_RISK,
        HIGH_RISK,
        HIGH_NEAR,
        ANY_GUIDANCE;

        fun matches(context: MessageContext): Boolean {
            val isCenter = context.position == HorizontalPosition.CENTER
            val isNear = context.distanceLevel == DistanceLevel.CLOSE || context.distanceLevel == DistanceLevel.VERY_CLOSE
            val risk = context.riskLevel

            return when (this) {
                LOW_LEFT -> !context.approaching && context.position == HorizontalPosition.LEFT && risk == RiskLevel.LOW
                LOW_RIGHT -> !context.approaching && context.position == HorizontalPosition.RIGHT && risk == RiskLevel.LOW
                MEDIUM_VISIBLE -> !context.approaching && risk == RiskLevel.MEDIUM && context.distanceLevel != DistanceLevel.VERY_CLOSE
                FRONT_OBSTACLE -> isCenter && context.objectGroup == ObjectGroup.OBSTACLE && risk >= RiskLevel.MEDIUM
                FRONT_OBSTACLE_NAMED -> !context.approaching && isCenter && context.objectGroup == ObjectGroup.OBSTACLE && risk >= RiskLevel.HIGH
                CENTER_APPROACHING -> context.approaching && isCenter
                CRITICAL_FRONT -> risk == RiskLevel.CRITICAL && isCenter && isNear
                SIDE_APPROACHING -> context.approaching && !isCenter
                SIDE_PERSON -> context.objectGroup == ObjectGroup.PERSON && !isCenter
                FRONT_VEHICLE -> !context.approaching && context.objectGroup == ObjectGroup.VEHICLE && isCenter && risk != RiskLevel.CRITICAL
                FRONT_HIGH_RISK -> !context.approaching && isCenter && risk >= RiskLevel.HIGH
                HIGH_RISK -> !context.approaching && risk == RiskLevel.HIGH
                HIGH_NEAR -> !context.approaching && risk == RiskLevel.HIGH && isNear
                ANY_GUIDANCE -> !context.approaching && risk <= RiskLevel.MEDIUM
            }
        }
    }

    class MessageTemplate(val text: String, val baseWeight: Int, val condition: TemplateCondition, val tags: Set<String>)

    data class WeightedTemplate(val text: String, val weight: Int)

    val messageTemplates = listOf(
        MessageTemplate("왼쪽에 {korean_label}{particle} 있어요.", 18, TemplateCondition.LOW_LEFT, setOf("low", "side", "natural")),
        MessageTemplate("오른쪽에 {korean_label}{particle} 있어요.", 18, TemplateCondition.LOW_RIGHT, setOf("low", "side", "natural")),
        MessageTemplate("{position_ko}에 {korean_label}{particle} 보여요.", 13, TemplateCondition.MEDIUM_VISIBLE, setOf("medium", "new")),
        MessageTemplate("{position_ko}에 {korean_label}{particle} 보여요. 조심해서 지나가 주세요.", 12, TemplateCondition.MEDIUM_VISIBLE, setOf("medium", "action")),
        MessageTemplate("{front_word}에 {korean_label}{particle} 있어요. 천천히 이동해 주세요.", 26, TemplateCondition.FRONT_OBSTACLE, setOf("obstacle", "front", "action", "medium")),
        MessageTemplate("{front_word}에 장애물이 있어요. 천천히 이동해 주세요.", 24, TemplateCondition.FRONT_OBSTACLE, setOf("obstacle", "front", "action", "medium")),
        MessageTemplate("정면에 {korean_label}{particle} 있어요. 조심해 주세요.", 24, TemplateCondition.FRONT_OBSTACLE_NAMED, setOf("obstacle", "front", "urgent")),
        MessageTemplate("정면 {korean_label}{particle} 가까워지고 있어요. 잠시 멈춰 주세요.", 34, TemplateCondition.CENTER_APPROACHING, setOf("approaching", "front", "urgent", "action")),
        MessageTemplate("정면의 {korean_label}{particle} 가까워지고 있어요. 주의해 주세요.", 22, TemplateCondition.CENTER_APPROACHING, setOf("approaching", "front", "urgent")),
        MessageTemplate("주의해 주세요. {korean_label}{particle} 바로 앞에 있어요.", 34, TemplateCondition.CRITICAL_FRONT, setOf("critical", "front", "urgent", "action")),
        MessageTemplate("잠시 멈춰 주세요. 정면에 {korean_label}{particle} 가까워요.", 30, TemplateCondition.CRITICAL_FRONT, setOf("critical", "front", "urgent", "action")),
        MessageTemplate("{position_from_ko} {korean_label}{particle} 다가오고 있어요.", 18, TemplateCondition.SIDE_APPROACHING, setOf("approaching", "side", "natural")),
        MessageTemplate("{position_ko}의 {korean_label}{particle} 가까워지고 있어요.", 18, TemplateCondition.SIDE_APPROACHING, setOf("approaching", "side")),
        MessageTemplate("{position_from_ko} {korean_label}{particle} 지나가고 있어요.", 20, TemplateCondition.SIDE_PERSON, setOf("person", "side", "natural")),
        MessageTemplate("{position_ko}에 사람이 있어요. 살짝 주의해 주세요.", 11, TemplateCondition.SIDE_PERSON, setOf("person", "side", "medium")),
        MessageTemplate("{position_ko}에 {korean_label}{particle} 보여요.", 9, TemplateCondition.SIDE_PERSON, setOf("person", "side", "new")),
        MessageTemplate("{front_word}에 {korean_label}{particle} 보여요. 주의해 주세요.", 24, TemplateCondition.FRONT_VEHICLE, setOf("vehicle", "front", "new", "urgent")),
        MessageTemplate("정면에 {korean_label}{particle} 가까워요. 조심해 주세요.", 22, TemplateCondition.FRONT_HIGH_RISK, setOf("front", "urgent", "near")),
        MessageTemplate("{position_ko}에 {korean_label}{particle} 있어요. 주의해 주세요.", 13, TemplateCondition.HIGH_RISK, setOf("urgent")),
        MessageTemplate("{position_ko}에 {korean_label}{particle} 가까이 있어요.", 14, TemplateCondition.HIGH_NEAR, setOf("urgent", "near", "natural")),
        MessageTemplate("{position_ko}에 {korean_label}{particle} 있어요.", 8, TemplateCondition.ANY_GUIDANCE, setOf("natural")),
        MessageTemplate("{position_ko}에 {korean_label}{particle} 보여요.", 7, TemplateCondition.ANY_GUIDANCE, setOf("new", "natural"))
    )

    const val FALLBACK_TEMPLATE = "{position_ko}에 {korean_label}{particle} 있어요."

    fun messageContext(detection: SceneDetection, eventType: GuidanceEventType): MessageContext {
        val koreanLabel = detection.koreanLabel.ifEmpty { "장애물" }
        return MessageContext(
            label = detection.label,
            koreanLabel = koreanLabel,
            particle = particle(koreanLabel),
            position = detection.position,
            riskLevel = detection.riskLevel,
            objectGroup = objectGroup(detection.label),
            eventType = eventType,
            approaching = detection.approaching || eventType == GuidanceEventType.APPROACHING,
            newlyDetected = eventType == GuidanceEventType.NEW_OBJECT,
            distanceLevel = distanceLevel(detection),
            frontWord = if (detection.label in highestRiskLabels) "앞쪽" else "앞"
        )
    }

    /** Templates that fit the situation, with their selection weights. */
    fun weightedTemplates(context: MessageContext): List<WeightedTemplate> {
        val weighted = messageTemplates
            .filter { it.condition.matches(context) }
            .map { WeightedTemplate(it.text, templateWeight(it.baseWeight, context, it.tags)) }
        return weighted.ifEmpty { listOf(WeightedTemplate(FALLBACK_TEMPLATE, 1)) }
    }

    /**
     * Picks a weighted template so repeated guidance does not sound robotic.
     * `random` returns a uniform value in `0..upperBound`.
     */
    fun detectionMessage(
        detection: SceneDetection,
        eventType: GuidanceEventType,
        calmerSpeech: Boolean = false,
        random: (Double) -> Double
    ): String {
        val context = messageContext(detection, eventType)
        var templates = weightedTemplates(context)
        // "다가오고 있어요" made a pole sound like it was moving.
        if (calmerSpeech && isFixed(detection.label)) {
            val still = templates.filter { !it.text.contains("다가오고") }
            if (still.isNotEmpty()) templates = still
        }
        val message = format(chooseWeightedTemplate(templates, random), context)
        val meters = detection.distanceMeters ?: return message
        return "$message ${distancePhrase(meters)}"
    }

    /** Spoken distance: half meters up close, whole meters farther away. */
    fun distancePhrase(meters: Double): String {
        if (meters < 1.0) return "1미터 안이에요."
        // Math.round rounds half up, like Swift's `.rounded()` for these positive values.
        val rounded = if (meters < 3.0) Math.round(meters * 2) / 2.0 else Math.round(meters).toDouble()
        val number = if (rounded == floor(rounded)) rounded.toInt().toString() else String.format(Locale.US, "%.1f", rounded)
        return "약 ${number}미터 거리예요."
    }

    fun particle(label: String): String {
        if (label.isEmpty()) return "이"
        val code = label.codePointAt(label.offsetByCodePoints(label.length, -1))
        if (code in 0xAC00..0xD7A3) {
            return if ((code - 0xAC00) % 28 != 0) "이" else "가"
        }
        return "이"
    }

    // ── Helpers ───────────────────────────────────────────────

    private fun distanceThresholds(label: String): DistanceThresholds {
        if (label in groundObstacleDistanceLabels) return groundObstacleDistanceThresholds
        return when (objectGroup(label)) {
            ObjectGroup.VEHICLE -> vehicleDistanceThresholds
            ObjectGroup.PERSON -> personDistanceThresholds
            ObjectGroup.MOBILITY -> mobilityDistanceThresholds
            ObjectGroup.TRAFFIC -> trafficDistanceThresholds
            ObjectGroup.OBSTACLE -> defaultDistanceThresholds
        }
    }

    private fun baseRisk(label: String): Int = when (label) {
        in highestRiskLabels -> 70
        in highRiskLabels -> 58
        in mediumHighRiskLabels -> 48
        in mediumRiskLabels -> 36
        in lowerRiskLabels -> 22
        else -> 28
    }

    private fun positionBonus(position: HorizontalPosition): Int = when (position) {
        HorizontalPosition.CENTER -> 15
        HorizontalPosition.RIGHT -> 6
        HorizontalPosition.LEFT -> 5
    }

    private fun templateWeight(baseWeight: Int, context: MessageContext, tags: Set<String>): Int {
        var weight = baseWeight
        val isUrgentOrAction = "urgent" in tags || "action" in tags

        weight += when (context.eventType) {
            GuidanceEventType.APPROACHING -> if ("approaching" in tags) 24 else -4
            GuidanceEventType.RISK_INCREASED, GuidanceEventType.CLOSER, GuidanceEventType.ENTERED_FRONT_ZONE ->
                if (isUrgentOrAction) 18 else 2
            GuidanceEventType.NEW_OBJECT -> if ("new" in tags) 12 else 3
        }

        weight += when (context.riskLevel) {
            RiskLevel.CRITICAL -> if ("critical" in tags) 34 else if (isUrgentOrAction) 18 else -8
            RiskLevel.HIGH -> if (isUrgentOrAction) 16 else 0
            RiskLevel.MEDIUM -> if ("medium" in tags || "action" in tags) 12 else 2
            RiskLevel.LOW -> if ("low" in tags || "natural" in tags) 16 else -6
        }

        weight += when (context.position) {
            HorizontalPosition.CENTER -> if ("front" in tags || "action" in tags) 16 else 2
            HorizontalPosition.RIGHT -> if ("side" in tags) 8 else 1
            HorizontalPosition.LEFT -> if ("side" in tags) 7 else 1
        }

        weight += when (context.objectGroup) {
            ObjectGroup.VEHICLE -> if ("vehicle" in tags || "urgent" in tags) 16 else 0
            ObjectGroup.MOBILITY, ObjectGroup.OBSTACLE -> if ("obstacle" in tags || "action" in tags) 13 else 3
            ObjectGroup.PERSON -> if ("person" in tags || "natural" in tags) 12 else 0
            ObjectGroup.TRAFFIC -> 0
        }

        weight += when (context.distanceLevel) {
            DistanceLevel.VERY_CLOSE -> if ("critical" in tags || "near" in tags || "action" in tags) 16 else -4
            DistanceLevel.CLOSE -> if ("near" in tags || isUrgentOrAction) 10 else 0
            DistanceLevel.FAR -> if ("low" in tags || "natural" in tags) 8 else -4
            DistanceLevel.NEAR -> 0
        }

        return max(1, weight)
    }

    private fun chooseWeightedTemplate(templates: List<WeightedTemplate>, random: (Double) -> Double): String {
        val total = templates.sumOf { max(0, it.weight) }
        if (total <= 0) return templates[0].text

        val cursor = random(total.toDouble())
        var running = 0.0
        for (template in templates) {
            running += max(0, template.weight).toDouble()
            if (cursor <= running) return template.text
        }
        return templates.last().text
    }

    private fun format(template: String, context: MessageContext): String =
        template
            .replace("{korean_label}", context.koreanLabel)
            .replace("{particle}", context.particle)
            .replace("{position_ko}", context.position.koreanName)
            .replace("{position_from_ko}", context.position.koreanName + "에서")
            .replace("{front_word}", context.frontWord)

    private fun clamp01(value: Double): Double = max(0.0, min(1.0, value))
}

enum class GuidanceEventType(val rawValue: String) {
    NEW_OBJECT("new_object"),
    CLOSER("closer"),
    RISK_INCREASED("risk_increased"),
    ENTERED_FRONT_ZONE("entered_front_zone"),
    APPROACHING("approaching");

    companion object {
        fun fromRaw(value: String): GuidanceEventType? = entries.firstOrNull { it.rawValue == value }
    }
}

data class GuidanceEvent(
    val type: GuidanceEventType,
    val detection: SceneDetection,
    val priority: Int,
    val message: String = ""
)

/**
 * Event-based speech suppression for live guidance. Times are in seconds.
 *
 * The detector has no stable object IDs, so situations are keyed by object class
 * and a coarse horizontal bucket. That is enough to avoid repeating the same
 * narration without a heavy multi-object tracker.
 */
class GuidanceEventTracker(
    private val cooldowns: Cooldowns = Cooldowns(),
    /**
     * The iOS app's changes after a field test, where guidance changed every 2.3 s and half
     * of it was about trees and poles beside the path: fixed objects beside the path are
     * spoken only within reach, calm guidance waits `CALM_GAP` after the last sentence, and
     * fixed objects are never said to be coming closer. Off, the tracker matches the backend.
     */
    private val calmerSpeech: Boolean = false,
    private val random: (Double) -> Double = { upperBound -> kotlin.random.Random.nextDouble() * upperBound }
) {
    data class Cooldowns(
        val global: Double = 1.5,
        val obj: Double = 6.0,
        val situation: Double = 8.0,
        val signature: Double = 10.0,
        val staleAfter: Double = 12.0
    )

    private data class SituationState(
        val lastSeenAt: Double,
        val lastSpokenAt: Double,
        val riskScore: Int,
        val riskLevel: RiskLevel,
        val position: HorizontalPosition,
        val areaRatio: Double,
        val approaching: Boolean,
        val seenCount: Int,
        /** Measured distance when this object was last announced. */
        val spokenDistanceMeters: Double?
    )

    private val states = HashMap<String, SituationState>()
    private val lastSituationAt = HashMap<String, Double>()
    private val lastSignatureAt = HashMap<String, Double>()
    private var lastSpokenAt = 0.0

    fun reset() {
        states.clear()
        lastSituationAt.clear()
        lastSignatureAt.clear()
        lastSpokenAt = 0.0
    }

    /** Returns up to `limit` events worth announcing now, each with its sentence. */
    fun chooseEvents(detections: List<SceneDetection>, now: Double, limit: Int = 2): List<GuidanceEvent> {
        states.entries.removeAll { now - it.value.lastSeenAt > cooldowns.staleAfter }

        val candidates = ArrayList<GuidanceEvent>()
        for (detection in detections) {
            if (!passesSpeechConfidence(detection)) {
                rememberSeen(detection, now, speechEligible = false)
                continue
            }
            if (!isTemporallyConfirmed(detection)) {
                rememberSeen(detection, now)
                continue
            }
            // Remembered while quiet, so it is spoken on its first frame within reach.
            if (calmerSpeech && isOutOfTheWay(detection)) {
                rememberSeen(detection, now)
                continue
            }

            val event = classifyEvent(detection, now)
            rememberSeen(detection, now)
            if (event != null) candidates.add(event)
        }

        val ordered = candidates.sortedWith(
            compareByDescending<GuidanceEvent> { it.priority }
                .thenByDescending { it.detection.riskScore }
                .thenByDescending { it.detection.confidence }
        )

        val selected = ArrayList<GuidanceEvent>()
        for (candidate in ordered) {
            val event = candidate.copy(
                message = GuidanceRules.detectionMessage(candidate.detection, candidate.type, calmerSpeech, random)
            )
            if (event.message.isEmpty() || !canSpeak(event, now)) continue

            selected.add(event)
            markSpoken(event, now)
            if (selected.size >= limit) break
        }
        return selected
    }

    private fun classifyEvent(detection: SceneDetection, now: Double): GuidanceEvent? {
        if (detection.label !in GuidanceRules.guidanceLabels) return null

        val riskScore = detection.riskScore
        val isCenterFront = detection.frontDangerZone || detection.position == HorizontalPosition.CENTER

        val state = states[objectKey(detection)]
            ?: return if (riskScore >= 35 || isCenterFront) GuidanceEvent(GuidanceEventType.NEW_OBJECT, detection, 45) else null

        if (now - state.lastSeenAt > cooldowns.staleAfter) {
            return GuidanceEvent(GuidanceEventType.NEW_OBJECT, detection, 45)
        }
        if (state.lastSpokenAt <= 0 && (riskScore >= 35 || isCenterFront)) {
            return GuidanceEvent(GuidanceEventType.NEW_OBJECT, detection, 45)
        }
        if (detection.approaching && !state.approaching) {
            return GuidanceEvent(GuidanceEventType.APPROACHING, detection, 95)
        }
        if (isCenterFront && state.position != HorizontalPosition.CENTER) {
            return GuidanceEvent(GuidanceEventType.ENTERED_FRONT_ZONE, detection, 85)
        }

        val scoreDelta = riskScore - state.riskScore
        val levelDelta = detection.riskLevel.rank - state.riskLevel.rank
        val areaDelta = detection.areaRatio - state.areaRatio

        if (scoreDelta >= 18 || (levelDelta >= 1 && scoreDelta >= 8)) {
            return GuidanceEvent(GuidanceEventType.RISK_INCREASED, detection, 80 + max(levelDelta, 0) * 5)
        }

        // With a measured distance, announce again once an object is a meter closer than last time and near.
        val meters = detection.distanceMeters
        val spoken = state.spokenDistanceMeters
        if (meters != null && spoken != null && spoken - meters >= 1.0 && meters < 3.0 && riskScore >= 35) {
            return GuidanceEvent(GuidanceEventType.CLOSER, detection, 72)
        }

        if (areaDelta >= 0.035 && riskScore >= 45) {
            return GuidanceEvent(GuidanceEventType.CLOSER, detection, 72)
        }
        return null
    }

    private fun rememberSeen(detection: SceneDetection, now: Double, speechEligible: Boolean = true) {
        val key = objectKey(detection)
        val previous = states[key]
        val previousSeenCount = previous?.seenCount ?: 0

        states[key] = SituationState(
            lastSeenAt = now,
            lastSpokenAt = previous?.lastSpokenAt ?: 0.0,
            riskScore = detection.riskScore,
            riskLevel = detection.riskLevel,
            position = detection.position,
            areaRatio = detection.areaRatio,
            approaching = detection.approaching,
            seenCount = if (speechEligible) previousSeenCount + 1 else previousSeenCount,
            spokenDistanceMeters = previous?.spokenDistanceMeters
        )
    }

    private fun canSpeak(event: GuidanceEvent, now: Double): Boolean {
        if (now - lastSpokenAt < cooldowns.global) return false
        if (calmerSpeech && !isUrgent(event) && now - lastSpokenAt < CALM_GAP) return false

        val objectCooldown = if (event.type == GuidanceEventType.APPROACHING || event.type == GuidanceEventType.ENTERED_FRONT_ZONE) {
            2.5
        } else {
            cooldowns.obj
        }
        val state = states[objectKey(event.detection)]
        if (state != null && now - state.lastSpokenAt < objectCooldown) return false

        if (now - (lastSituationAt[situationKey(event)] ?: 0.0) < cooldowns.situation) return false
        if (now - (lastSignatureAt[speechSignature(event)] ?: 0.0) < cooldowns.signature) return false
        return true
    }

    private fun markSpoken(event: GuidanceEvent, now: Double) {
        val key = objectKey(event.detection)
        states[key]?.let {
            states[key] = it.copy(lastSpokenAt = now, spokenDistanceMeters = event.detection.distanceMeters)
        }
        lastSituationAt[situationKey(event)] = now
        lastSignatureAt[speechSignature(event)] = now
        lastSpokenAt = now
    }

    private fun passesSpeechConfidence(detection: SceneDetection): Boolean {
        val confidence = max(0.0, min(1.0, detection.confidence))
        return when (detection.label) {
            in GuidanceRules.lowConfidenceSpeechLabels -> confidence >= 0.25
            in GuidanceRules.noisySpeechLabels -> confidence >= 0.55
            else -> confidence >= 0.35
        }
    }

    /** Requires a few consecutive sightings before speaking, except for urgent objects. */
    private fun isTemporallyConfirmed(detection: SceneDetection): Boolean {
        val confidence = max(0.0, min(1.0, detection.confidence))
        if (detection.label in GuidanceRules.immediateSpeechLabels && confidence >= 0.35) return true
        if (detection.approaching) return true

        val seenCount = (states[objectKey(detection)]?.seenCount ?: 0) + 1
        val requiredCount = if (detection.label in GuidanceRules.noisySpeechLabels) 3 else 2
        return seenCount >= requiredCount
    }

    private fun objectKey(detection: SceneDetection): String {
        val scale = if (detection.frameWidth > 0) REFERENCE_FRAME_WIDTH / detection.frameWidth else 1.0
        val bucket = floor(detection.bbox.centerX * scale / OBJECT_BUCKET_WIDTH).toInt()
        return "${detection.label}:$bucket"
    }

    private fun situationKey(event: GuidanceEvent): String =
        "${event.type.rawValue}:${event.detection.label}:${event.detection.position.rawValue}"

    private fun speechSignature(event: GuidanceEvent): String {
        val detection = event.detection
        val distance = GuidanceRules.distanceLevel(detection)
        return "${event.type.rawValue}:${detection.label}:${detection.position.rawValue}:" +
            "${detection.riskLevel.rawValue}:${distance.rawValue}"
    }

    private companion object {
        /** The backend bucketed box centers every 192 px on 1080 px wide frames. */
        const val REFERENCE_FRAME_WIDTH = 1080.0
        const val OBJECT_BUCKET_WIDTH = 192.0
        const val CALM_GAP = 4.0

        /**
         * A fixed object beside the path and out of reach: 1.5 m when measured, the box's
         * "very close" level without. One that reaches into the path is never out of the way.
         */
        fun isOutOfTheWay(detection: SceneDetection): Boolean {
            val outOfReach = detection.distanceMeters?.let { it >= 1.5 }
                ?: (GuidanceRules.distanceLevel(detection) != DistanceLevel.VERY_CLOSE)
            return GuidanceRules.isFixed(detection.label) && detection.position != HorizontalPosition.CENTER &&
                !GuidanceRules.reachesPath(detection) && outOfReach
        }

        /** In the path, coming closer or within 2.5 m: spoken without the calm gap. */
        fun isUrgent(event: GuidanceEvent): Boolean {
            val detection = event.detection
            val distance = GuidanceRules.distanceLevel(detection)
            return detection.position == HorizontalPosition.CENTER || event.type == GuidanceEventType.APPROACHING ||
                detection.approaching || distance == DistanceLevel.CLOSE || distance == DistanceLevel.VERY_CLOSE
        }
    }
}
