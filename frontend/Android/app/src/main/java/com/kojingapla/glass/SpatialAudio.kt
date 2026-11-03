package com.kojingapla.glass

import kotlin.math.max
import kotlin.math.min

/** One spoken guidance sentence and the stereo direction it comes from. */
data class SpokenSentence(val text: String, val pan: Float)

/**
 * Places live guidance in the stereo field, so with earphones each sentence comes from
 * the side the object is on, before its words say so.
 *
 * Blind navigation studies found sound that comes from the target's direction guides
 * better than spoken directions alone (Loomis, Golledge & Klatzky 1998, "Navigation
 * System for the Blind: Auditory Display Modes and Guidance"; Microsoft Soundscape).
 * Plain stereo panning is the lightest form of that: no head tracking or HRTF, and the
 * words still name the side for anyone on a single speaker.
 */
object SpatialAudio {
    /**
     * Never fully to one side, so the far ear still hears the sentence and a mono
     * earpiece or a single earbud loses nothing.
     */
    const val MAX_PAN = 0.8f

    /** Pan for a box: -1 is fully left, 1 fully right, following the box center. */
    fun pan(bbox: BoundingBox, frameWidth: Double): Float {
        if (frameWidth <= 0) return 0f
        val offset = (bbox.centerX / frameWidth - 0.5) * 2
        return (max(-1.0, min(1.0, offset)) * MAX_PAN).toFloat()
    }
}
