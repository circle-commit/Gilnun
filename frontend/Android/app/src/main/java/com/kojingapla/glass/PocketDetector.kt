package com.kojingapla.glass

/**
 * Notices when the phone goes into a pocket or bag, from the proximity sensor above the
 * screen: covered for a while means put away, so live analysis can pause and save the
 * battery; uncovered again means it is back in hand. Times are in seconds.
 *
 * A brief cover, such as a hand passing over the sensor or the phone against a cheek
 * for a moment, changes nothing.
 */
class PocketDetector(
    private val coveredFor: Double = 1.5,
    private val uncoveredFor: Double = 0.5
) {
    enum class Change { PUT_AWAY, TAKEN_OUT }

    var isPutAway = false
        private set
    private var covered = false
    private var changedAt = 0.0

    fun reset() {
        isPutAway = false
        covered = false
    }

    /** Feeds a sensor reading; returns a change once the new state has lasted long enough. */
    fun update(isCovered: Boolean, now: Double): Change? {
        if (isCovered != covered) {
            covered = isCovered
            changedAt = now
        }
        val lasted = now - changedAt
        if (!isPutAway && covered && lasted >= coveredFor) {
            isPutAway = true
            return Change.PUT_AWAY
        }
        if (isPutAway && !covered && lasted >= uncoveredFor) {
            isPutAway = false
            return Change.TAKEN_OUT
        }
        return null
    }
}
