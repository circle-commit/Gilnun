package com.kojingapla.glass

import kotlin.math.asin
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Live-mode checks for problems a blind user cannot see: the phone pointing at the
 * ground, the sky or sideways, and the lens covered or the street too dark. Without
 * them, guidance just goes quiet and the user has no way to tell why.
 *
 * Each problem is spoken only after it lasts a moment (a glance down at a curb is
 * fine), at most every few seconds, and once fixed a short sentence says so.
 * Times are in seconds.
 */
enum class ViewProblem(val message: String) {
    POINTING_DOWN("휴대폰이 바닥을 비추고 있어요. 조금 들어서 앞을 비춰 주세요."),
    POINTING_UP("휴대폰이 위를 비추고 있어요. 조금 내려서 앞을 비춰 주세요."),
    SIDEWAYS("휴대폰을 세로로 세워 들어 주세요."),
    LENS_COVERED("카메라가 가려졌어요. 렌즈를 막고 있는 손가락이나 물건을 치워 주세요."),
    TOO_DARK("너무 어두워서 잘 보이지 않아요. 천천히 조심해서 이동해 주세요.")
}

object CameraViewCheck {
    /** The camera may look this far below the horizon; the sidewalk video the model learned from looks slightly down. */
    const val LOWEST_ELEVATION = -50.0
    const val HIGHEST_ELEVATION = 25.0

    /** Mean luma below which a frame with almost no detail is a covered lens. */
    const val COVERED_BRIGHTNESS = 25.0
    const val COVERED_DETAIL = 3.0

    /** Mean luma below which the detector misses too much at night. */
    const val DARK_BRIGHTNESS = 35.0

    /**
     * Where the back camera points, from the accelerometer or gravity sensor reading
     * (x right, y toward the top of the screen, z out of the screen; at rest it points up).
     */
    fun aimProblem(x: Float, y: Float, z: Float): ViewProblem? {
        val length = sqrt(x * x + y * y + z * z)
        if (length < 1f) return null
        // The back camera looks along -z, so its angle above the horizon is asin(-z / g).
        val elevation = Math.toDegrees(asin((-z / length).toDouble().coerceIn(-1.0, 1.0)))
        return when {
            elevation < LOWEST_ELEVATION -> ViewProblem.POINTING_DOWN
            elevation > HIGHEST_ELEVATION -> ViewProblem.POINTING_UP
            // Upright, gravity runs along y; held sideways, along x.
            abs(x) > abs(y) -> ViewProblem.SIDEWAYS
            else -> null
        }
    }

    /** A covered lens is dark and nearly featureless; a dark street still has edges and lights. */
    fun sightProblem(brightness: Double, detail: Double): ViewProblem? = when {
        brightness < COVERED_BRIGHTNESS && detail < COVERED_DETAIL -> ViewProblem.LENS_COVERED
        brightness < DARK_BRIGHTNESS -> ViewProblem.TOO_DARK
        else -> null
    }
}

/** Turns a per-frame problem (or none) into rare, spaced-out sentences. */
class ViewProblemSpeaker(
    private val delay: Double = 1.5,
    private val repeatEvery: Double = 10.0,
    private val fixedFor: Double = 1.0
) {
    private var current: ViewProblem? = null
    private var currentSince = 0.0
    private var spoken: ViewProblem? = null
    private var spokenAt = Double.NEGATIVE_INFINITY

    fun reset() {
        current = null
        spoken = null
        spokenAt = Double.NEGATIVE_INFINITY
    }

    fun update(problem: ViewProblem?, now: Double): String? {
        if (problem != current) {
            current = problem
            currentSince = now
        }
        val lasted = now - currentSince

        if (problem == null) {
            // Confirm the fix once, so the user knows guidance is back.
            if (spoken != null && lasted >= fixedFor) {
                spoken = null
                return "좋아요. 이제 앞이 잘 보여요."
            }
            return null
        }
        if (lasted < delay) return null
        if (problem == spoken && now - spokenAt < repeatEvery) return null
        spoken = problem
        spokenAt = now
        return problem.message
    }
}
